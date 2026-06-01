import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import autograd
import numpy as np
from torch.utils.data import DataLoader
import torch.autograd as AG
# from models.split_model import get_last_layer_name

# In get_last_layer_name function
def get_last_layer_name(model_name):
    # 将VGG的判断逻辑提前，确保它能被优先匹配
    if 'vgg19_bn' in model_name:
        return 'classifier'
    
    # 保留其他模型的原有逻辑
    if model_name == 'resnet18':
        return 'linear'
    elif model_name == 'resnet50':
        return 'linear'
    elif model_name == 'MobileNetV2':
        return 'linear'
    elif model_name == 'MobileNet':
        return 'linear'
    elif model_name == 'shufflenetv2':
        return 'fc.1'
    elif model_name == 'densenet':
        return 'linear'
    
    # 如果没有匹配项，返回None
    return None

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
            log_likelihoods.append(output[:, target])

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
            # V_i = [torch.randint_like(p, high=2, device=device) for p in model.parameters()]
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

    def forward_backward_update(self, input_s, target, iteration):
        self.optimizer.zero_grad()
        outputs = self.model(input_s)
        ce_loss = self.crit(outputs, target)
        reg_loss = self._compute_reg_loss(self.weight)
        trace_loss = self.get_trace_loss(outputs, target) if iteration % self.iter_gap == 0 else 0.0
        
        # 强制迹损失非负
        if isinstance(trace_loss, torch.Tensor):
            trace_loss = torch.clamp(trace_loss, min=0.0)

        loss = ce_loss + self.reg_F * trace_loss + reg_loss
        loss.backward()
        self.optimizer.step()
        
        trace_loss_item = trace_loss.item() if isinstance(trace_loss, torch.Tensor) else trace_loss
        # 返回5个值，与子类保持一致
        return loss, outputs, ce_loss.item(), reg_loss.item(), trace_loss_item


class TargetedFIPRegularizer(CDA_Regularizer):
    def __init__(self, args, device, model, criterion,
                 guilty_neurons_by_layer: dict, target_class_index: int,
                 intermediate_shapes: dict): # <<< (1/4) 修改：接收形状字典
        super().__init__(args, device, model, criterion)

        self.guilty_neurons_by_layer = guilty_neurons_by_layer
        self.target_class_index = target_class_index
        self.reg_F = args.reg_F
        self.intermediate_shapes = intermediate_shapes # <<< (2/4) 新增：存储形状字典
        
        # 以下代码与之前相同
        self.targeted_param_names_set = self._get_params_from_guilty_layers()
        
        self.ordered_targeted_param_names = []
        self.targeted_params = []
        for name, param in model.named_parameters():
            if name in self.targeted_param_names_set:
                self.ordered_targeted_param_names.append(name)
                self.targeted_params.append(param)
        
        # 使用新的掩码创建函数
        self.guilty_masks = self._create_fine_grained_guilty_masks() # <<< (3/4) 修改：调用新的掩码创建函数

        print("--- TargetedFIPRegularizer Initialized (Fine-grained Neuron Mode) ---")
        for layer, neurons in self.guilty_neurons_by_layer.items():
             print(f"  - Targeting {len(neurons)} neurons in layer '{layer}' for output class {self.target_class_index}.")

    def _get_params_from_guilty_layers(self) -> set:
        """根据有罪神经元所在的逻辑层，筛选出需要优化的参数名称集合（同时兼容ResNet和VGG）"""
        params_to_target = set()
        
        # --- 【新增】VGG19_bn 物理层索引映射 ---
        # 定义每个逻辑块包含的 'features.X' 索引范围
        # VGG19_bn 结构: [Conv,BN,ReLU,Conv,BN,ReLU,MP], [Conv,BN,ReLU,Conv,BN,ReLU,MP], ...
        # 这需要根据torchvision中VGG19_bn的具体实现来确定
        vgg19_bn_block_indices = {
            'features_block_1': range(0, 7),      # 包含 features.0 到 features.6
            'features_block_2': range(7, 14),     # 包含 features.7 到 features.13
            'features_block_3': range(14, 27),    # ...
            'features_block_4': range(27, 40),
            'features_block_5': range(40, 53)
        }
        
        for layer_name, guilty_indices in self.guilty_neurons_by_layer.items():
            if not guilty_indices: continue

            # --- 【修改】核心判断逻辑 ---
            target_prefixes = []
            is_vgg_feature_block = False
            if 'vgg' in self.args.arch and layer_name in vgg19_bn_block_indices:
                is_vgg_feature_block = True
                indices_range = vgg19_bn_block_indices[layer_name]
                target_prefixes = [f'features.{i}.' for i in indices_range]

            for param_name, _ in self.model.named_parameters():
                # 原始逻辑，用于ResNet等
                if param_name.startswith(layer_name):
                    params_to_target.add(param_name)
                
                # 新增的VGG逻辑
                elif is_vgg_feature_block:
                    if any(param_name.startswith(prefix) for prefix in target_prefixes):
                        params_to_target.add(param_name)
        
        # --- (处理分类头的逻辑保持不变) ---
        classifier_head_layers = {'avgpool', 'flatten', 'classifier', 'fc'}
        guilty_layer_names = set(self.guilty_neurons_by_layer.keys())

        if not classifier_head_layers.isdisjoint(guilty_layer_names):
            final_fc_name = get_last_layer_name(self.args.arch)
            if final_fc_name:
                params_to_target.add(f'{final_fc_name}.weight')
                params_to_target.add(f'{final_fc_name}.bias')
            else:
                print(f"警告: _get_params_from_guilty_layers 无法找到模型 {self.args.arch} 的最后一层名称。")

        return params_to_target

    # 文件: Regularizer_ultra.py
    # 请用这个【V4 - 最终防呆修正版】替换整个 _create_fine_grained_guilty_masks 函数

    def _create_fine_grained_guilty_masks(self):
        """
        [V4 - 最终防呆修正版] 创建精细的“有罪掩码”。
        1. 修正了VGG参数名匹配的致命Bug。
        2. 通过isinstance精确判断层类型，避免对BatchNorm层进行错误操作。
        3. (可选) 支持Top-K通道选择，以平衡ASR和ACC。
        """
        masks = {}
        vgg19_bn_block_indices = {
            'features_block_1': range(0, 7), 'features_block_2': range(7, 14),
            'features_block_3': range(14, 27), 'features_block_4': range(27, 40),
            'features_block_5': range(40, 53)
        }

        with torch.no_grad():
            for layer_logic_name, guilty_indices in self.guilty_neurons_by_layer.items():
                if not guilty_indices:
                    continue

                # --- 情况1：对于卷积层 ---
                if 'features_block' in layer_logic_name:
                    if layer_logic_name not in self.intermediate_shapes:
                        print(f"警告: 找不到层 {layer_logic_name} 的形状信息，跳过掩码创建。")
                        continue
                    
                    _, _, H, W = self.intermediate_shapes[layer_logic_name]
                    
                    target_guilty_indices = guilty_indices
                    if self.args.soda_topk_channels is not None and self.args.soda_topk_channels < len(guilty_indices):
                        target_guilty_indices = guilty_indices[:self.args.soda_topk_channels]
                        print(f"    应用Top-K限制：在层 {layer_logic_name} 中，从 {len(guilty_indices)} 个嫌疑神经元中选择Top {len(target_guilty_indices)} 个进行惩罚。")

                    guilty_channel_indices = {idx // (H * W) for idx in target_guilty_indices}

                    if 'vgg' in self.args.arch and layer_logic_name in vgg19_bn_block_indices:
                        physical_layer_indices = vgg19_bn_block_indices[layer_logic_name]
                        for i in physical_layer_indices:
                            # --- 核心修复：直接获取模块并检查其类型 ---
                            module = self.model.features[i]
                            if isinstance(module, nn.Conv2d):
                                param_name = f'features.{i}.weight'
                                param = module.weight # 直接从模块获取权重
                                
                                mask = torch.zeros_like(param, dtype=torch.float32)
                                valid_indices = [c for c in guilty_channel_indices if c < param.shape[0]]
                                if valid_indices:
                                    # 这是安全的，因为我们已确认param是4D的
                                    mask[valid_indices, :, :, :] = 1.0
                                masks[param_name] = mask
                                print(f"    为参数 {param_name} 创建权重掩码，靶向 {len(valid_indices)} 个输出通道。")
                
                # --- 情况2：对于全连接层 (逻辑保持不变) ---
                elif layer_logic_name in ['avgpool', 'flatten', 'classifier']:
                    final_fc_name = get_last_layer_name(self.args.arch)
                    if not final_fc_name: continue
                    
                    weight_name, bias_name = f'{final_fc_name}.weight', f'{final_fc_name}.bias'
                    
                    # (后续的全连接层掩码创建逻辑不变)
                    if weight_name in dict(self.model.named_parameters()):
                        weight_param = dict(self.model.named_parameters())[weight_name]
                        weight_mask = torch.zeros_like(weight_param, dtype=torch.float32)
                        valid_indices = [idx for idx in guilty_indices if idx <= weight_param.shape[1] - 1]
                        if valid_indices:
                            weight_mask[self.target_class_index, valid_indices] = 1.0
                        masks[weight_name] = weight_mask

                    if bias_name in dict(self.model.named_parameters()):
                        bias_param = dict(self.model.named_parameters())[bias_name]
                        bias_mask = torch.zeros_like(bias_param, dtype=torch.float32)
                        bias_mask[self.target_class_index] = 1.0
                        masks[bias_name] = bias_mask
                        
        return masks

    def _compute_reg_loss(self, weight):
        ewc_loss = 0.0
        for name, param in self.model.named_parameters():
            _buff_name = name.replace('.', '__')
            try:
                mean = getattr(self.model, f'{_buff_name}_estimated_mean')
                fisher = getattr(self.model, f'{_buff_name}_estimated_fisher')
            except AttributeError:
                continue

            loss_term = fisher * (param - mean) ** 2
            
            if name in self.targeted_param_names_set:
                if name in self.guilty_masks:
                    innocent_mask = 1.0 - self.guilty_masks[name]
                    ewc_loss += torch.sum(loss_term * innocent_mask)
            else:
                ewc_loss += torch.sum(loss_term)

        return (weight / 2) * ewc_loss
        
    def get_targeted_trace_loss(self, outputs, target):
        output_softmax = F.log_softmax(outputs, dim=1)
        log_likelihood = output_softmax[range(len(target)), target].mean()
        grads = autograd.grad(log_likelihood, self.targeted_params, create_graph=True, allow_unused=True)
        
        trace = 0.0
        for _ in range(3): 
            v = [torch.randn_like(p) for p in self.targeted_params]
            masked_grad_v_sum = 0.0
            for i, name in enumerate(self.ordered_targeted_param_names):
                g = grads[i]
                if g is None: continue
                
                # --- 【关键修改】 ---
                # 统一使用掩码（如果存在），不再需要 is_neuron_mode 判断
                if name in self.guilty_masks:
                    guilty_mask = self.guilty_masks[name]
                    masked_grad_v_sum += torch.sum((g * guilty_mask) * v[i])
                else:
                    # 如果参数是靶向目标但没有特定掩码（例如BN层的偏置），我们靶向整个参数
                    masked_grad_v_sum += torch.sum(g * v[i])

            Hv = autograd.grad(masked_grad_v_sum, self.targeted_params, retain_graph=True, allow_unused=True)
            for j in range(len(Hv)):
                if Hv[j] is not None:
                    trace += torch.sum(Hv[j] * v[j])
        
        trace = torch.clamp(trace / 3, min=0.0)
        return trace

    def forward_backward_update(self, input_s, target, iteration):
        # 这个函数保持不变
        self.optimizer.zero_grad()
        outputs = self.model(input_s)
        ce_loss = self.crit(outputs, target)
        reg_loss = self._compute_reg_loss(self.weight)
        trace_loss = self.get_targeted_trace_loss(outputs, target) if iteration % self.iter_gap == 0 else 0.0
        loss = ce_loss + reg_loss + self.reg_F * trace_loss
        loss.backward()
        self.optimizer.step()
        trace_loss_item = trace_loss.item() if isinstance(trace_loss, torch.Tensor) else trace_loss
        return loss, outputs, ce_loss.item(), reg_loss.item(), trace_loss_item