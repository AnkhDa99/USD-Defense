import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import autograd
import numpy as np
from torch.utils.data import DataLoader
import torch.autograd as AG
import math
import random
from torchvision.transforms.functional import gaussian_blur

def _softmax_except_target(logits, target_idx, T=1.0, eps=1e-8):
    """
    计算除目标类别外的 Softmax 概率分布。
    [FIX]: 在 softmax 之后增加了一个微小的 eps，以防止后续 log(0) 导致 nan。
    """
    # logits: [B, C]
    B, C = logits.shape
    mask = torch.ones_like(logits, dtype=torch.bool)
    mask[:, target_idx] = False
    z = logits / T
    z_mask = z.masked_fill(~mask, -float('inf'))
    p = torch.softmax(z_mask, dim=-1)

    # --- 核心修正 ---
    # 给概率 p 加上一个极小的数，防止出现 log(0) 的情况
    p = p + eps
    # 重新归一化，确保概率和为 1
    p = p / p.sum(dim=-1, keepdim=True)
    # --- 修正结束 ---
    
    return p

def usd_consistency_loss(logits_list, target_idx, T=1.0, margin=2.0, alpha=1.0, beta=1.0):
    if not logits_list or len(logits_list) < 2:
        return torch.tensor(0.0, device=logits_list[0].device if logits_list else 'cpu')
        
    # 每个视图的非target分布
    P = [_softmax_except_target(z, target_idx, T=T) for z in logits_list]
    # 使用 clamp_min 避免 log(0)
    P_mean = torch.stack(P, dim=0).mean(0).clamp_min(1e-8)

    # KL 一致性（对称，近似）
    kl_divs = []
    for p in P:
        # 计算 KL(p || P_mean)
        kl = F.kl_div(P_mean.log(), p, reduction='none').sum(dim=-1)
        kl_divs.append(kl)
    
    # 对所有视图的KL散度取平均
    kl = torch.stack(kl_divs).mean()

    # target 排斥：保证非target的最高 logit 比 target 高 margin
    rejection_losses = []
    for z in logits_list:
        z_t = z[:, target_idx]
        # 非target最高 logit
        mask = torch.ones_like(z, dtype=torch.bool, device=z.device)
        mask[:, target_idx] = False
        z_ntop = z.masked_fill(~mask, -float('inf')).max(dim=-1).values
        rej = F.relu(margin - (z_ntop - z_t)).mean()
        rejection_losses.append(rej)
    
    rej = torch.stack(rejection_losses).mean()

    return alpha * kl + beta * rej

def usd_lambda_warmup(step, warmup_steps=400, base=0.1, maxv=0.6):
    if step >= warmup_steps: return maxv
    return base + (maxv - base) * (step / float(warmup_steps))

class CDA_Regularizer:

    def __init__(self, args, device, model, crit, lr=0.002, reg_F=0.01, weight=5):
        self.model = model
        self.device = device
        self.weight = weight
        self.reg_F = args.reg_F
        self.crit = crit
        self.args = args
        self.iter_gap = 5
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=args.lr, momentum=0.95)
        print("Regularizer Coffieicients:", self.iter_gap, self.reg_F, self.weight)

    def _update_mean_params(self):
        for param_name, param in self.model.named_parameters():
            _buff_param_name = param_name.replace('.', '__')
            self.model.register_buffer(_buff_param_name + '_estimated_mean', param.data.clone())

    def _update_fisher_params_initial(self, current_ds, batch_size, num_batch):
        dl = DataLoader(current_ds, batch_size, shuffle=True)
        log_likelihoods = []
        for i, (inputs, target) in enumerate(dl):
            if i > num_batch:
                break
            inputs, target = inputs.cuda(), target.cuda()
            output = F.log_softmax(self.model(inputs), dim=1)
            # log_likelihoods.append(output[:, target])
            true_logp = output.gather(1, target.view(-1, 1)).squeeze(1)
            log_likelihoods.append(true_logp)

        log_likelihood = torch.cat(log_likelihoods).mean()
        # print("log likelihood:", log_likelihood)

        grad_log_liklihood = autograd.grad(log_likelihood, self.model.parameters())
        _buff_param_names = [name for name, param in self.model.named_parameters()]
        for _buff_param_name, param in zip(_buff_param_names, grad_log_liklihood):
            _buff_param_name = _buff_param_name.replace('.', '__')
            self.model.register_buffer(_buff_param_name + '_estimated_fisher', param.data.clone() ** 2)

    ## Resgister using clean validation images
    def register_ewc_params(self, dataset, batch_size, num_batches):
        self._update_fisher_params_initial(dataset, batch_size, num_batches)
        self._update_mean_params()

    # def _update_fisher_params_initial(self, current_ds, batch_size, num_batch):
    #     dl = DataLoader(current_ds, batch_size, shuffle=True)
    #     log_likelihoods = []
    #     self.model.eval() # Set model to evaluation mode for stable statistics

    #     for i, (inputs, target) in enumerate(dl):
    #         if i >= num_batch: # Use >= to match num_batch correctly
    #             break
    #         inputs, target = inputs.cuda(), target.cuda()
            
    #         # FIX 2: Use .gather() for correct log-likelihood indexing.
    #         # This correctly selects the log probability of the true class for each sample.
    #         log_likelihood_batch = F.log_softmax(self.model(inputs), dim=1).gather(1, target.view(-1, 1))
    #         log_likelihoods.append(log_likelihood_batch)

    #     self.model.train() # Set model back to training mode

    #     log_likelihood = torch.cat(log_likelihoods).mean()

    #     grad_log_liklihood = autograd.grad(log_likelihood, self.model.parameters())
    #     _buff_param_names = [name for name, param in self.model.named_parameters()]
    #     for _buff_param_name, param in zip(_buff_param_names, grad_log_liklihood):
    #         _buff_param_name = _buff_param_name.replace('.', '__')
    #         self.model.register_buffer(_buff_param_name + '_estimated_fisher', param.data.clone() ** 2)

    ## Regularization Loss
    def _compute_reg_loss(self, weight):
        try:
            losses = []
            for param_name, param in self.model.named_parameters():
                _buff_param_name = param_name.replace('.', '__')
                estimated_mean = getattr(self.model, '{}_estimated_mean'.format(_buff_param_name))
                estimated_fisher = getattr(self.model, '{}_estimated_fisher'.format(_buff_param_name))

                loss_consolidation = (estimated_fisher * (param - estimated_mean) ** 2).sum()
                losses.append(loss_consolidation)

            return (weight / 2) * sum(losses)

        except AttributeError:
            return 0

    def get_trace_loss(self, outputs, target, hi=20):

        output = F.log_softmax(outputs, dim=1)
        log_liklihoods = output[:, target]
        log_likelihood = log_liklihoods.mean()
        Fv = AG.grad(log_likelihood, self.model.parameters(), create_graph=True)

        # for V_i in V:
        #     # Hv = AG.grad(Fv, params, V_i, create_graph=True)

        niters = hi
        V = list()
        for _ in range(niters):
            # V_i = [torch.randint_like(p, high=2, device=self.device) for p in self.model.parameters()]
            # for V_ij in V_i:
            #     V_ij[V_ij == 0] = -1
            V_i = [torch.randn_like(p, device=self.device) for p in self.model.parameters()]
            V.append(V_i)

        trace = list()
        for V_i in V:
            this_trace = 0.0
            for Hv_, V_i_ in zip(Fv, V_i):
                this_trace = this_trace + torch.sum(Hv_ * V_i_)
                trace.append(this_trace)

        return sum(trace) / niters
    # def get_trace_loss(self, outputs, target, hi=10):
    #     """
    #     Correctly calculates the trace of the Hessian using Hutchinson's method.
    #     This is a critical fix for the FIP algorithm to work as intended.
    #     """
    #     # Ensure the loss is a scalar
    #     log_likelihood = F.log_softmax(outputs, dim=1).gather(1, target.view(-int(1), 1)).mean()

    #     params = [p for p in self.model.parameters() if p.requires_grad]

    #     # 1. Calculate the first gradient (g) and create a graph for it
    #     g = AG.grad(log_likelihood, params, create_graph=True)

    #     trace_estimates = []
    #     for _ in range(hi):
    #         # 2. Generate Rademacher random vectors (v), which have lower variance
    #         v = [torch.randint_like(p, low=0, high=2, device=self.device) * 2 - 1 for p in params]

    #         # 3. Calculate the dot product g·v (a scalar)
    #         g_v_dot = sum(torch.sum(g_p * v_p) for g_p, v_p in zip(g, v))
            
    #         # 4. Calculate the Hessian-vector product (Hv) by taking the gradient of g·v
    #         # retain_graph=True is necessary as we perform this step in a loop
    #         Hv = AG.grad(g_v_dot, params, retain_graph=True)
            
    #         # 5. Calculate the final dot product vᵀHv
    #         v_Hv_dot = sum(torch.sum(h_p * v_p) for h_p, v_p in zip(Hv, v))
    #         trace_estimates.append(v_Hv_dot)

    #     # Average the estimates and detach from the graph before returning
    #     final_trace = sum(trace_estimates) / len(trace_estimates)
    #     return final_trace.detach()

    def forward_backward_update(self, input_s, target, iteration):
        self.optimizer.zero_grad()

        outputs = self.model(input_s)
        ce_loss = self.crit(outputs, target)
        reg_loss    = self._compute_reg_loss(self.weight)

        ### Backpropagate the loss
        if iteration % self.iter_gap == 0:
            trace_loss = self.get_trace_loss(outputs, target)
        else:
            trace_loss = 0

        ## Total Loss
        loss = ce_loss + self.reg_F * trace_loss   + reg_loss
        loss.backward()
        self.optimizer.step()

        return loss, outputs

    def save(self, filename):
        torch.save(self.model, filename)

    def load(self, filename):
        self.model = torch.load(filename)

class FiG_AD_Loss(nn.Module):
    """
    Fisher-Guided Adaptive Distillation (FiG-AD) Loss Framework.
    整合了 Logit Standardization (LS-KD), Uncertainty-aware weighting,
    和可选的 Transformed Teacher Matching (TTM).
    """
    def __init__(self, args, teacher_model):
        super().__init__()
        self.args = args
        self.teacher_model = teacher_model
        self.kl_div_loss = nn.KLDivLoss(reduction='none') 
        
        if self.args.use_ttm:
            # [MODIFICATION] 获取类别数，如果未定义则默认为 10 (兼容 CIFAR10)
            num_classes = getattr(self.args, 'num_classes', 10)
            self.t_adapter = nn.Linear(num_classes, num_classes, bias=True).to(self.args.device)
            print(f"TTM Adapter ENABLED. Output dimension: {num_classes}")

    def standardize_logits(self, z, eps=1e-6):
        if not self.args.use_lskd:
            return z
        m = z.mean(dim=1, keepdim=True)
        v = z.var(dim=1, keepdim=True, unbiased=False)
        return (z - m) / torch.sqrt(v + eps)

    def forward(self, student_outputs, inputs, targets):
        self.teacher_model.eval()

        with torch.no_grad():
            teacher_outputs = self.teacher_model(inputs)

        if self.args.use_ttm:
            teacher_logits_transformed = self.t_adapter(teacher_outputs)
        else:
            teacher_logits_transformed = teacher_outputs
            
        s_z = self.standardize_logits(student_outputs)
        t_z = self.standardize_logits(teacher_logits_transformed.detach())

        T = self.args.temperature
        
        with torch.no_grad():
            teacher_prob = F.softmax(teacher_outputs, dim=1)
            teacher_conf, teacher_preds = torch.max(teacher_prob, dim=1)
            
            entropy = -(teacher_prob * (teacher_prob.clamp_min(1e-8)).log()).sum(dim=1)
            
            # Use stable normalization
            max_entropy = math.log(self.args.num_classes)
            confidence_weight = (1.0 - (entropy / max_entropy))
            
            gate_mask = (teacher_preds == targets) & (teacher_conf > self.args.agkd_confidence_thresh)
            final_weights = confidence_weight * gate_mask.float()
        
        # self.kl_div_loss now returns a tensor of shape [batch_size, num_classes]
        kd_vec = self.kl_div_loss(
            F.log_softmax(s_z / T, dim=1),
            F.softmax(t_z / T, dim=1)
        ).sum(dim=1) # Sum over the class dimension
        
        weighted_kd_loss = (final_weights * kd_vec).sum() / (final_weights.sum() + 1e-8)
        
        return weighted_kd_loss * (T * T)
    
class UnifiedSemanticDefense(nn.Module):
    def __init__(self, args, source_samples=None):
        super().__init__()
        self.args = args
        self.source_samples = source_samples.to(args.device) if source_samples is not None else None
        self.margin = args.usd_margin
        self.confidence_thresh = args.usd_thresh_start
        self.active_ops = [op.strip() for op in args.usd_base_ops.split(',') if op]
        if hasattr(args, 'usd_refool_alpha_range'):
            if isinstance(args.usd_refool_alpha_range, str):
                alpha_min, alpha_max = map(float, args.usd_refool_alpha_range.replace(' ', '').split(','))
                self.usd_refool_alpha_range = (alpha_min, alpha_max)
            else:
                self.usd_refool_alpha_range = tuple(args.usd_refool_alpha_range)
        else:
            self.usd_refool_alpha_range = (0.3, 0.6)
        if args.poison_type == 'weather':
            # Keep old behavior by default: rain only.
            # Only explicit --weather_effect snow switches USD views to snow.
            weather_effect = getattr(args, 'weather_effect', 'rain')
            if weather_effect == 'snow':
                self.active_ops = ['snow']
            else:
                self.active_ops = ['rain']
        
        # [MODIFICATION] 针对 GTSRB 的特殊适配 (Refool/Weather)
        is_gtsrb = getattr(self.args, 'dataset', 'CIFAR10') == 'GTSRB'
        
        if self.args.poison_type == 'refool':
            print(f"[USD] Refool attack detected ({is_gtsrb}), applying specific defense parameters.")
            
            # GTSRB Plan B: 使用更深/更宽的层来捕捉语义异常，避免底层边缘特征干扰
            if is_gtsrb:
                self.channel_shrink_layer = 'layer3' 
                self.feature_layers = ['layer2', 'layer3', 'layer4'] # Wide scope
            else:
                self.channel_shrink_layer = 'layer3'
                self.feature_layers = ['layer2', 'layer3', 'layer4']

            base_topk_ratio = getattr(self.args, 'usd_topk_ratio', 0.03)
            self.topk_ratio = max(base_topk_ratio, 0.05)
            if 'refool_mix' not in self.active_ops:
                self.active_ops.append('refool_mix')
        else:
            # Weather 或其他攻击
            self.channel_shrink_layer = 'layer4'
            self.feature_layers = ['layer3', 'layer4']
            self.topk_ratio = getattr(self.args, 'usd_topk_ratio', 0.03)

    @torch.no_grad()
    def _make_view(self, x, y):
        if not self.active_ops: return x, 'none'
        
        op = random.choice(self.active_ops) if self.active_ops else 'none'

        if self.args.poison_type == 'refool' and 'refool_mix' in self.active_ops:
            if random.random() < getattr(self.args, 'usd_refool_mix_prob', 0.6):
                op = 'refool_mix'
            else:
                others = [o for o in self.active_ops if o != 'refool_mix']
                op = random.choice(others) if others else 'refool_mix'
        else:
            op = random.choice(self.active_ops) if self.active_ops else 'none'
        
        B, C, H, W = x.shape
        
        if op == 'reflect':
            alpha = self.args.usd_alpha
            x_flip = torch.flip(x, dims=[3])
            v = (1 - alpha) * x + alpha * x_flip
            return v.clamp(0, 1), op
        
        elif op == 'rain':
            from PIL import Image, ImageDraw
            import torchvision.transforms.functional as TF
            v_list = []
            # 轻量的强度课程：前半程轻、后半程重
            # g_step 只能在 forward 里拿，这里取 args.usd_weather_intensity 作为基准
            base_I = getattr(self.args, 'usd_weather_intensity', 0.3)
            # 若无全局步，这里简单抖动强度；也可把 g_step 作为 forward 入参传进 _make_view
            for b in range(x.size(0)):
                pil = TF.to_pil_image(x[b].cpu())
                w, h = pil.size
                overlay = Image.new('RGBA', (w, h), (0,0,0,0))
                draw = ImageDraw.Draw(overlay)
                num = int(base_I * 500)
                for _ in range(num):
                    x1 = np.random.randint(0, w)
                    y1 = np.random.randint(0, h)
                    length = np.random.randint(5, 15)
                    x2 = x1 + np.random.randint(-2, 2)
                    y2 = y1 + length
                    draw.line(((x1, y1), (x2, y2)), fill=(200,200,200,150), width=1)
                out = Image.alpha_composite(pil.convert('RGBA'), overlay).convert('RGB')
                v_list.append(TF.to_tensor(out))
            v = torch.stack(v_list, dim=0).to(x.device)
            return v.clamp(0, 1), op

        elif op == 'snow':
            B, C, H, W = x.shape
            base_I = getattr(self.args, 'usd_weather_intensity', 0.3)

            # 雪点概率，控制不要过度白化图像
            flake_prob = min(0.35, max(0.01, base_I * 0.25))

            snow = (torch.rand(B, 1, H, W, device=x.device) < flake_prob).float()

            # 模糊雪点，使其更接近自然雪花/雪雾，而不是单像素噪声
            k = 5 if min(H, W) >= 32 else 3
            snow = gaussian_blur(
                snow,
                kernel_size=(k, k),
                sigma=(0.3, 1.2)
            ).clamp(0, 1)

            alpha = min(0.8, max(0.15, base_I * 1.5))
            v = x + alpha * snow.expand_as(x)

            return v.clamp(0, 1), op
            
        elif op == 'gaussian_blur':
            # C.1: 修复 gaussian_blur 对批量张量的处理
            try:
                # 使用列表推导式逐个样本处理，然后堆叠回一个批量
                v = torch.stack([gaussian_blur(img, kernel_size=(3, 3), sigma=(0.1, 2.0)) for img in x])
                return v, op
            except Exception as e:
                # 如果出现意外错误，返回原始图像以保证程序继续运行
                print(f"[WARN] Gaussian blur failed with error: {e}. Returning original image.")
                return x, 'none'
            
        elif op == 'jitter':
            v = x + (torch.randn_like(x) * 0.03)
            return v.clamp(0, 1), op

        elif op == 'refool_mix':
            if self.source_samples is None: return x, 'none'
            # 1) 取源图，并镜像，拟合“反射”方向
            src_indices = torch.randint(0, len(self.source_samples), (B,), device=x.device)
            x_src = self.source_samples[src_indices]
            x_src = torch.flip(x_src, dims=[3])  # 水平镜像

            # 2) 生成“上方/斜带”局部掩膜 + 轻模糊，接近玻璃反射的形状
            _, _, H, W = x.shape
            mask = torch.zeros(B, 1, H, W, device=x.device)

            band_h_min = int(0.20 * H); band_h_max = int(0.50 * H)
            # 使用 torch.randint 需要 low < high
            if band_h_min >= band_h_max: band_h_max = band_h_min + 1
            band_h = torch.randint(band_h_min, band_h_max, (B,), device=x.device)

            y0_max = int(0.35 * H)
            if y0_max <= 0: y0_max = 1
            y0 = torch.randint(0, y0_max, (B,), device=x.device)  # 反射多出现在上半部
            
            for b in range(B):
                y1 = min(H, y0[b] + band_h[b])
                mask[b, 0, y0[b]:y1, :] = 1.0

            # 轻模糊让边缘柔和
            k = 7 if min(H, W) >= 32 else 5
            mask = gaussian_blur(mask, kernel_size=(k, k), sigma=(1.0, 2.5)).clamp(0, 1)

            # 3) 随机不透明度（支持动态范围）
            alpha_min, alpha_max = self.usd_refool_alpha_range
            alpha = torch.rand(B, 1, 1, 1, device=x.device) * (alpha_max - alpha_min) + alpha_min

            # 4) 伽马/亮度微调（玻璃反射常有亮度/对比差异）
            gamma = torch.empty(B, 1, 1, 1, device=x.device).uniform_(0.9, 1.1)
            x_reflect = (x_src ** gamma).clamp(0, 1)

            # 5) 仅在掩膜区域进行“反射混合”
            v = (1 - alpha * mask) * x + (alpha * mask) * x_reflect
            return v.clamp(0, 1), op
            
        return x, 'none'


    @torch.no_grad()
    def infer(self, model, teacher_model, x, g_step=0, num_views=3):
        """
        单张净化推理：
        1) teacher 置信度门控
        2) 生成多视图并计算 effect-aware gate
        3) 若触发则执行输出抑制/重判别
        返回：logits_final
        """
        model.eval()
        if teacher_model is not None:
            teacher_model.eval()

        # 0) teacher 基础门控（与训练一致）
        if teacher_model is not None:
            tlogits = teacher_model(x)
            p_teacher = F.softmax(tlogits, dim=1)
            conf = p_teacher.max(1).values
            if conf.item() < self.confidence_thresh:
                return model(x)  # teacher不确定，直接返回原预测
        else:
            pass

        # 1) 原始预测
        logits_orig = model(x)

        # 2) 多视图探测 + effect-aware gate（与训练一致）
        triggered = False
        for _ in range(num_views):
            x_view, op_used = self._make_view(x, None)
            if op_used == 'none':
                continue

            logits_view = model(x_view)

            # 只对 refool_mix / rain / snow 做 effect-aware gate
            if op_used in ['refool_mix', 'rain', 'snow']:
                p_orig = F.softmax(logits_orig, dim=1)[:, self.args.target_label]
                p_view = F.softmax(logits_view, dim=1)[:, self.args.target_label]
                delta_t = (p_view - p_orig)

                if op_used == 'refool_mix':
                    tau = getattr(self.args, 'usd_refool_delta_thresh', 0.02)
                else:
                    tau = getattr(self.args, 'usd_weather_delta_thresh', 0.03)

                if delta_t.item() > tau:
                    triggered = True
                    break

        # 3) 触发则进行抑制/重判别
        logits_final = logits_orig.clone()
        if triggered:
            # 方式A：直接压低目标类logit一个margin
            t = self.args.target_label
            logits_final[:, t] = logits_final[:, t] - self.args.usd_margin

        return logits_final

    # ===== [MODIFICATION] Replaced forward method with Patch-3 logic =====
    def forward(self, model, teacher_model, inputs, targets, g_step):
        with torch.no_grad():
            teacher_logits_orig = teacher_model(inputs)
            p_teacher = F.softmax(teacher_logits_orig, dim=1)
            conf = p_teacher.max(1).values
            # 基础门控，过滤掉教师模型不确定的样本
            base_gate_mask = (conf >= self.confidence_thresh)

        if not base_gate_mask.any():
            return torch.tensor(0.0).to(inputs.device)

        inputs_gated = inputs[base_gate_mask]
        
        total_usd_loss = 0.0
        num_valid_views = 0
        
        # 为所有视图计算一致性损失，但为refool应用特殊门控
        logits_orig = model(inputs_gated)
        
        num_views = 3
        for _ in range(num_views):
            x_view, op_used = self._make_view(inputs_gated, None)
            if op_used == 'none':
                continue

            logits_view = model(x_view)
            
            # 默认为 True，即对所有样本计算损失
            final_gate_mask = torch.ones(logits_view.shape[0], dtype=torch.bool, device=inputs.device)

            if op_used in ['refool_mix', 'rain', 'snow']:
                # ---- Effect-aware gate for Refool / Weather ----
                # 只保留那些"视图让目标类概率显著上升"的样本
                p_orig = F.softmax(logits_orig, dim=1)[:, self.args.target_label]
                p_view = F.softmax(logits_view, dim=1)[:, self.args.target_label]
                delta_t = (p_view - p_orig)

                if op_used == 'refool_mix':
                    tau = getattr(self.args, 'usd_refool_delta_thresh', 0.02)
                else:
                    tau = getattr(self.args, 'usd_weather_delta_thresh', 0.03)
                    
                final_gate_mask = (delta_t > tau)
                
                if not final_gate_mask.any():
                    continue # 如果没有任何样本被激活，则跳过此视图的损失计算
                # --------------------------------------

            # 使用最终的门控来选择参与损失计算的样本
            logits_orig_gated = logits_orig[final_gate_mask]
            logits_view_gated = logits_view[final_gate_mask]

            # 1. 一致性损失(KL-Div): 依然在原图和视图间计算，但不包含抑制项 (beta=0.0)
            kl_loss = usd_consistency_loss(
                [logits_orig_gated, logits_view_gated],
                target_idx=self.args.target_label,
                T=2.0, margin=0.0,  # 禁用此处的margin计算
                alpha=1.0, beta=0.0   # 只获取KL散度部分
            )

            # 2. 抑制损失(Rejection): 只对“视图”(view)的logit计算，迫使其远离目标类别
            zv = logits_view_gated
            z_t = zv[:, self.args.target_label]
            
            # 获取除目标类别外的最高logit值
            target_one_hot = F.one_hot(torch.full_like(z_t, self.args.target_label, dtype=torch.long), num_classes=zv.shape[1]).bool()
            z_ntop = zv.masked_fill(target_one_hot, -float('inf')).max(dim=-1).values
            
            rejection_loss = F.relu(self.args.usd_margin - (z_ntop - z_t)).mean()

            # 3. 组合加权后的总损失
            loss_for_this_view = self.args.usd_lambda_consist * kl_loss + self.args.usd_lambda_suppress * rejection_loss
            
            total_usd_loss += loss_for_this_view
            num_valid_views += 1

        # 对所有有效视图的损失取平均
        avg_usd_loss = total_usd_loss / max(1, num_valid_views)
        
        lam_usd = usd_lambda_warmup(g_step, warmup_steps=400, base=0.1, maxv=0.6)
        return lam_usd * avg_usd_loss