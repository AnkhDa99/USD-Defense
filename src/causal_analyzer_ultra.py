import torch
import numpy as np
import torch.nn as nn
import os
from collections import OrderedDict
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn.functional as F
import poison_cifar as poison

# 假设您的数据加载器和模型分割函数位于这些路径
# 您可能需要根据您的项目结构调整这些导入
from data.data_loader import *
from Remove_Backdoor_ultra import *

class ReluWrapper(nn.Module):
    def forward(self, x):
        return F.relu(x)

class AvgPoolWrapper(nn.Module):
    def forward(self, x):
        return F.avg_pool2d(x, 4)

class FlattenWrapper(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


def custom_split_model(model, model_arch, split_layer_name):
    """
    一个更通用的模型拆分函数，同时支持 ResNet 和 VGG 架构。
    """
    if 'resnet' in model_arch.lower():
        # --- 现有的 ResNet 拆分逻辑 (保持不变) ---
        try:
            layers_dict = OrderedDict([
                # 您的模型 stem: conv1 -> bn1 -> F.relu, 无 maxpool
                ('stem', nn.Sequential(model.conv1, model.bn1, ReluWrapper())),
                ('layer1', model.layer1),
                ('layer2', model.layer2),
                ('layer3', model.layer3),
                ('layer4', model.layer4),
                # 您的模型分类头: F.avg_pool2d -> view -> linear
                ('avgpool', AvgPoolWrapper()),
                ('flatten', FlattenWrapper()),
                ('fc', model.linear) # 您的全连接层名为 'linear'
            ])
        except AttributeError as e:
            print(f"错误: 您的ResNet模型实现与预期不符。错误信息: {e}")
            return None, None, None

        layer_names = list(layers_dict.keys())
        # ResNet 的层索引映射
        idx_to_name_map = {
            '0': 'stem', '1': 'layer1', '2': 'layer2', '3': 'layer3', '4': 'layer4',
            '5': 'avgpool', '6': 'flatten', '7': 'fc'
        }

    elif 'vgg19_bn' in model_arch.lower():
        # --- 新增的 VGG 拆分逻辑 ---
        print(f"信息: 检测到VGG架构 ({model_arch})，应用VGG拆分逻辑。")
        features = model.features
        
        # VGG的结构是 features -> classifier。我们按 MaxPool 层将 features 模块拆分为逻辑块
        vgg_logical_layers = []
        current_block = []
        for layer in features:
            current_block.append(layer)
            if isinstance(layer, nn.MaxPool2d):
                vgg_logical_layers.append(nn.Sequential(*current_block))
                current_block = []
        
        # 添加分类头部分
        vgg_logical_layers.append(FlattenWrapper())
        vgg_logical_layers.append(model.classifier)
        
        # 创建逻辑层名称列表
        layer_names = [f'features_block_{i+1}' for i in range(len(vgg_logical_layers) - 2)]
        layer_names.extend(['flatten', 'classifier'])
        
        layers_dict = OrderedDict(zip(layer_names, vgg_logical_layers))

        # VGG 的层索引映射
        idx_to_name_map = {str(i): name for i, name in enumerate(layer_names)}

    else:
        print(f"警告: 当前拆分逻辑未针对 {model_arch} 进行优化。请在 custom_split_model 中添加支持。")
        return None, None, None

    # --- 后续的通用拆分逻辑 (基本保持不变) ---
    if split_layer_name not in layer_names:
        original_input = str(split_layer_name)
        split_layer_name = idx_to_name_map.get(original_input, None)
        if split_layer_name is None:
            print(f"错误: 对于模型 {model_arch}，无效的层名称或索引 '{original_input}'")
            print(f"    可用层: {list(idx_to_name_map.items())}")
            return None, None, None

    split_idx = layer_names.index(split_layer_name)

    model1_layers = [layers_dict[name] for name in layer_names[:split_idx + 1]]
    model2_layers = [layers_dict[name] for name in layer_names[split_idx + 1:]]

    model1 = nn.Sequential(*model1_layers)
    model2 = nn.Sequential(*model2_layers)

    # 判断层类型，用于后续的干预操作
    layer_type = 'fc' if 'flatten' in split_layer_name or 'classifier' in split_layer_name else 'conv'
    
    return model1, model2, layer_type

class CausalAnalyzer:
    def __init__(self, model, model_arch, num_classes, ana_layers, output_dir, device,
                 poison_type='unknown', mad_th=3.0,
                 w_pcc=0.4, w_var=0.3, w_ace=0.3):
        """
        [SODA论文完整复刻版] 初始化因果分析器
        """
        self.model = model
        self.model_arch = model_arch
        self.num_classes = num_classes
        self.output_dir = output_dir
        self.device = device
        self.poison_type = poison_type
        self.mad_th = mad_th
        self.pcc_th = 1.5 # 保持一个默认的pcc阈值

        # 【SODA复刻-关键点1】: 自适应权重与SCM模式控制
        self.w_pcc = w_pcc
        self.w_var = w_var
        self.w_ace = w_ace
        self.use_scm_adjustment = False # SCM模式的标志位，默认关闭

        # 存储中间层形状，以供后续使用
        self.intermediate_shapes = {}

        # 将层级参数统一处理为列表
        if isinstance(ana_layers, int):
            self.ana_layers = [str(ana_layers)]
        elif isinstance(ana_layers, str):
            self.ana_layers = [item.strip() for item in ana_layers.split(',')]
        else:
            self.ana_layers = [str(item) for item in ana_layers]

        self.intervention_params = [(np.random.uniform(0.5, 1.5), np.random.uniform(-0.2, 0.2)) for _ in range(10)]

        os.makedirs(self.output_dir, exist_ok=True)

        self.idx_to_name_map, self.layer_names = self._generate_layer_maps()

    def _get_ca_filepath(self, class_idx, layer_idx):
        filename = f"{self.poison_type}_layer{layer_idx}_class{class_idx}.txt"
        return os.path.join(self.output_dir, filename)
    
    def _get_assumed_confounders(self, all_neuron_activations: np.ndarray, target_neuron_idx: int, top_k: int = 10) -> list[int]:
        if all_neuron_activations.shape[1] <= top_k + 1:
            return [i for i in range(all_neuron_activations.shape[1]) if i != target_neuron_idx]

        with np.errstate(divide='ignore', invalid='ignore'):
            correlations = np.corrcoef(all_neuron_activations, rowvar=False)
            correlations = np.nan_to_num(correlations)

        target_correlations = np.abs(correlations[target_neuron_idx])
        correlated_indices = np.argsort(target_correlations)[::-1]
        confounder_indices = [idx for idx in correlated_indices if idx != target_neuron_idx][:top_k]
        return confounder_indices

    def _compute_dce_via_backdoor_adjustment(self, model2: torch.nn.Module,
                                             dense_hidden: torch.Tensor,
                                             target_neuron_idx: int,
                                             confounder_indices: list[int],
                                             ori_output: torch.Tensor,
                                             original_shape: tuple,
                                             intervention_strength: float = 1.5) -> float:
        intervened_hidden = dense_hidden.clone()
        original_neuron_val = intervened_hidden[:, target_neuron_idx].mean()
        intervened_hidden[:, target_neuron_idx] = original_neuron_val * intervention_strength
        intervened_hidden_reshaped = intervened_hidden.view(original_shape)

        if not confounder_indices:
            with torch.no_grad():
                intervened_output = model2(intervened_hidden_reshaped)
                dce = (F.softmax(intervened_output, dim=1)[:, self.target_class_for_dce] - F.softmax(ori_output, dim=1)[:, self.target_class_for_dce]).mean().item()
            return dce

        confounder_activations = dense_hidden[:, confounder_indices]
        num_strata = 5
        quantiles_tensor = torch.linspace(0, 1, num_strata + 1, device=self.device)
        total_effect = 0.0
        total_weight = 0.0
        avg_confounder_activations = confounder_activations.mean(dim=1)
        bucket_quantiles = torch.quantile(avg_confounder_activations, quantiles_tensor)
        stratum_indices = torch.bucketize(avg_confounder_activations, bucket_quantiles)

        for i in range(1, num_strata + 1):
            stratum_mask = (stratum_indices == i)
            if stratum_mask.sum() > 0:
                prob_z = stratum_mask.float().mean()
                with torch.no_grad():
                    outputs_on_intervened_stratum = model2(intervened_hidden_reshaped[stratum_mask])
                    effect_in_stratum = (F.softmax(outputs_on_intervened_stratum, dim=1)[:, self.target_class_for_dce] - F.softmax(ori_output[stratum_mask], dim=1)[:, self.target_class_for_dce]).mean()
                total_effect += effect_in_stratum * prob_z
                total_weight += prob_z
        return total_effect.item() if total_weight > 0 else 0.0

    def _analyze_and_save_ca_for_class(self, class_loader, cur_class, layer_name, num_samples):
        self.model.eval()
        model1, model2, layer_type = custom_split_model(self.model, self.model_arch, layer_name)
        if model1 is None or model2 is None: return False

        model1.to(self.device).eval()
        model2.to(self.device).eval()

        ca_batches = []
        samples_processed = 0
        pbar_desc = f"分析 CA for class {cur_class}, layer '{layer_name}' (SCM: {self.use_scm_adjustment})"
        pbar = tqdm(class_loader, desc=pbar_desc, leave=False)

        for images, _ in pbar:
            if samples_processed >= num_samples: break
            images = images.to(self.device)

            with torch.no_grad():
                intermediate_out = model1(images)
                ori_output = model2(intermediate_out)

            if self.use_scm_adjustment:
                original_shape = intermediate_out.shape
                dense_hidden = torch.reshape(intermediate_out, (intermediate_out.shape[0], -1))
                dense_hidden_np = dense_hidden.cpu().numpy()
                confounder_cache = {}
                dce_for_batch = []
                self.target_class_for_dce = cur_class
                for i in range(dense_hidden.shape[1]):
                    if i not in confounder_cache:
                        confounder_cache[i] = self._get_assumed_confounders(dense_hidden_np, i)
                    dce = self._compute_dce_via_backdoor_adjustment(
                        model2, dense_hidden, i,
                        confounder_cache[i], ori_output,
                        original_shape
                    )
                    dce_for_batch.append(dce)
                dce_vector = np.array(dce_for_batch)
                ca_for_batch = torch.tensor(np.tile(dce_vector[:, np.newaxis], (1, self.num_classes)), device=self.device)
            else:
                if layer_type == 'conv':
                    num_units = intermediate_out.shape[1]
                else:
                    num_units = intermediate_out.shape[1]
                batch_unit_ca_list = []
                for i in range(num_units):
                    ca_for_this_unit = []
                    for a, b in self.intervention_params:
                        intervened_hidden = intermediate_out.clone()
                        if layer_type == 'conv':
                            intervened_hidden[:, i, :, :] = a * intervened_hidden[:, i, :, :] + b
                        else:
                            intervened_hidden[:, i] = a * intervened_hidden[:, i] + b
                        output_do = model2(intervened_hidden)
                        ca = torch.abs(ori_output - output_do)
                        ca_for_this_unit.append(ca)
                    robust_ca = torch.mean(torch.stack(ca_for_this_unit), dim=0)
                    avg_robust_ca_for_unit = torch.mean(robust_ca, dim=0)
                    batch_unit_ca_list.append(avg_robust_ca_for_unit)
                ca_for_batch = torch.stack(batch_unit_ca_list, dim=0)

            # ca_batches.append(ca_for_batch.cpu())
            ca_batches.append(ca_for_batch.cpu().detach())
            samples_processed += images.shape[0]

        if not ca_batches: return False
        avg_ca_matrix = torch.mean(torch.stack(ca_batches), dim=0).numpy()
        save_path = self._get_ca_filepath(cur_class, layer_name)
        np.savetxt(save_path, np.c_[np.arange(avg_ca_matrix.shape[0]), avg_ca_matrix], fmt="%.8f")
        return True

    def _compute_pcc(self, layer_idx):
        all_ca_data = {}
        for c in range(self.num_classes):
            ca_file = self._get_ca_filepath(c, layer_idx)
            if not os.path.exists(ca_file):
                print(f"    警告: 找不到类 {c} 在 layer {layer_idx} 的CA文件。跳过此层的PCC计算。")
                return None
            all_ca_data[c] = np.loadtxt(ca_file)
        avg_pcc_per_target = []
        for target_c in range(self.num_classes):
            pcc_list = []
            for source_j in range(self.num_classes):
                ca_data = all_ca_data[source_j]
                ca_vector_target = ca_data[:, target_c + 1]
                other_indices = [k + 1 for k in range(self.num_classes) if k != target_c]
                ca_vector_others = np.mean(ca_data[:, other_indices], axis=1)
                with np.errstate(invalid='ignore'):
                    if np.std(ca_vector_target) > 1e-5 and np.std(ca_vector_others) > 1e-5:
                        pcc = np.corrcoef(ca_vector_target, ca_vector_others)[0, 1]
                        if not np.isnan(pcc): pcc_list.append(pcc)
            avg_pcc_per_target.append(np.mean(pcc_list) if pcc_list else 0.0)
        return np.array(avg_pcc_per_target)

    def _mad_outlier_detection(self, data, threshold, positive_only=True):
        data = np.asarray(data)
        median = np.median(data)
        abs_dev = np.abs(data - median)
        mad = np.median(abs_dev)
        if mad < 1e-9: return []
        adjusted_mad = 1.4826 * mad
        z_scores = (data - median) / adjusted_mad
        if positive_only:
            outliers = [(i, data[i]) for i, z in enumerate(z_scores) if z > threshold]
        else:
            outliers = [(i, data[i]) for i, z in enumerate(z_scores) if abs(z) > threshold]
        return outliers
    
    #############################################################################################
    ### SODA复刻：以下是来自causal_analyzer_ultra.py的、更复杂的目标类别检测逻辑 ###
    #############################################################################################

    def _apply_trigger(self, images: torch.Tensor, attack_type: str) -> torch.Tensor:
        """
        根据攻击类型对一批图像应用触发器。
        """
        if not hasattr(self, '_trigger_cache'):
            self._trigger_cache = {}
        trigger_map = {
            'badnets': 'checkerboard_1corner', 'CLB': 'fourCornerTrigger',
            'blend': 'gaussian_noise', 'SIG': 'signalTrigger',
            'TrojanNet': 'trojanTrigger', 'FC': 'gridTrigger',
            'Feature': 'feature_trigger'
        }
        if attack_type not in trigger_map:
            if attack_type not in ['semantic', 'semantic2', 'refool', 'benign']:
                 print(f"警告: 未在 _apply_trigger 中为攻击类型 '{attack_type}' 定义触发器映射。")
            return images
        trigger_type = trigger_map[attack_type]
        if trigger_type in self._trigger_cache:
            pattern, mask = self._trigger_cache[trigger_type]
        else:
            try:
                pattern_np, mask_np = poison.generate_trigger(trigger_type=trigger_type)
                pattern = torch.from_numpy(pattern_np).to(self.device).float()
                if mask_np.ndim == 3 and mask_np.shape[2] == 1:
                    mask_np = np.repeat(mask_np, 3, axis=2)
                mask_np = mask_np.transpose((2, 0, 1))
                mask = torch.from_numpy(mask_np).to(self.device).float()
                self._trigger_cache[trigger_type] = (pattern.permute(2,0,1), mask)
                pattern = pattern.permute(2,0,1)
            except (ValueError, KeyError):
                print(f"警告: poison_cifar.py 中未实现 '{trigger_type}' 类型的触发器。")
                return images
        
        alpha_map = {'badnets': 0.6, 'blend': 0.2, 'FC': 1.0}
        alpha = alpha_map.get(attack_type, 0.5)
        triggered_images = (1 - mask) * images + alpha * mask * pattern
        return torch.clamp(triggered_images, 0.0, 1.0)

    def _compute_average_causal_effect(self, all_class_loaders, target_class, attack_type):
        """计算平均因果效应 (ACE)"""
        self.model.eval()
        control_probs, treatment_probs = [], []
        
        # 控制组 E[Y|do(T=0)]
        with torch.no_grad():
            for images, _ in all_class_loaders[target_class]:
                outputs = self.model(images.to(self.device))
                control_probs.append(F.softmax(outputs, dim=1)[:, target_class].cpu())
        
        # 干预组 E[Y|do(T=1)]
        with torch.no_grad():
            for source_c, loader in enumerate(all_class_loaders):
                if source_c == target_class: continue
                for images, _ in loader:
                    triggered_images = self._apply_trigger(images.to(self.device), attack_type)
                    outputs = self.model(triggered_images)
                    treatment_probs.append(F.softmax(outputs, dim=1)[:, target_class].cpu())
        
        prob_do_t0 = torch.cat(control_probs).mean().item() if control_probs else 0.0
        prob_do_t1 = torch.cat(treatment_probs).mean().item() if treatment_probs else 0.0
        return prob_do_t1 - prob_do_t0

    def _generate_counterfactual_image(self, poisoned_image, attack_type):
        """
        [修改建议] 生成反事实样本。
        对于 patch_based 攻击, 不再是添加弱噪声，而是直接擦除触发器区域。
        """
        trigger_map = {'badnets': 'checkerboard_1corner', 'CLB': 'fourCornerTrigger', 'blend': 'gaussian_noise', 'SIG': 'signalTrigger', 'TrojanNet': 'trojanTrigger', 'FC': 'gridTrigger', 'Feature': 'feature_trigger'}
        trigger_type = trigger_map.get(attack_type)
        
        if not trigger_type or not hasattr(self, '_trigger_cache') or trigger_type not in self._trigger_cache:
            # 如果没有找到触发器信息，返回原图（无操作）
            return poisoned_image
            
        _, mask = self._trigger_cache[trigger_type]
        
        # --- 核心修改 ---
        # 使用 torch.where 将触发器区域（mask > 0 的地方）的像素值替换为0.0（黑色）
        # 这是一种比添加噪声更强烈的干预，能有效破坏 Feature 攻击的鲁棒性
        # 也可以替换为图像的平均像素值，效果可能更好
        # mean_pixel_value = poisoned_image.mean(dim=[-1, -2], keepdim=True)
        # cf_image = torch.where(mask.to(self.device) > 0, mean_pixel_value, poisoned_image)

        cf_image = torch.where(mask.to(self.device) > 0, torch.tensor(0.0, device=self.device), poisoned_image)

        return cf_image

    def _compute_cfe_score(self, candidate_class, all_class_loaders, attack_type):
        """为单个候选目标类别计算反事实因果效应(CFE)分数"""
        cfe_scores = []
        self.model.eval()
        with torch.no_grad():
            for source_c, loader in enumerate(all_class_loaders):
                if source_c == candidate_class: continue
                try:
                    images, _ = next(iter(loader))
                    images = images.to(self.device)
                    poisoned_images = self._apply_trigger(images, attack_type)
                    factual_outputs = F.softmax(self.model(poisoned_images), dim=1)
                    cf_images = self._generate_counterfactual_image(poisoned_images, attack_type)
                    cf_outputs = F.softmax(self.model(cf_images), dim=1)
                    effect = factual_outputs[:, candidate_class] - cf_outputs[:, candidate_class]
                    cfe_scores.append(effect.mean().item())
                except StopIteration:
                    continue
        return np.mean(cfe_scores) if cfe_scores else 0.0

    def _get_all_metrics_for_layer(self, layer_idx, all_class_loaders, attack_type):
        """辅助函数: 计算单层的所有三个指标分数"""
        pcc_scores = self._compute_pcc(layer_idx)
        if pcc_scores is None: return None
        
        all_vars = []
        for c in range(self.num_classes):
            ca_file = self._get_ca_filepath(c, layer_idx)
            if not os.path.exists(ca_file): all_vars.append(0); continue
            ca_data = np.loadtxt(ca_file)
            if len(ca_data.shape) < 2 or ca_data.shape[1] <= c + 1: all_vars.append(0); continue
            all_vars.append(np.var(ca_data[:, c + 1]))
        
        all_vars = np.array(all_vars)
        normalized_vars = (all_vars - np.min(all_vars)) / (np.max(all_vars) - np.min(all_vars)) if np.max(all_vars) > 1e-9 else np.zeros_like(all_vars)
        ace_scores = [self._compute_average_causal_effect(all_class_loaders, c, attack_type) for c in range(self.num_classes)]
        
        return {'pcc': pcc_scores, 'var': normalized_vars, 'ace': np.array(ace_scores)}

    def _detect_target_class(self, data_dir, batch_size, attack_type):
        """
        [移植自causal_analyzer_ultra.py] 两阶段目标类别检测方法。
        第一阶段：通过PCC, VAR, ACE三个专家生成候选。
        第二阶段：通过反事实因果效应(CFE)验证最终目标。
        """
        print(f"\n--- 运行两阶段目标类别检测 (layers: {self.ana_layers}, attack: {attack_type}) ---")
        
        all_layer_scores_pcc, all_layer_scores_var, all_layer_scores_ace = [], [], []
        print("    一次性加载所有类别的数据...")
        all_class_loaders = [get_custom_class_loader(data_dir, batch_size, c) for c in range(self.num_classes)]

        for layer_idx in self.ana_layers:
            scores = self._get_all_metrics_for_layer(layer_idx, all_class_loaders, attack_type)
            if scores:
                all_layer_scores_pcc.append(scores['pcc'])
                all_layer_scores_var.append(scores['var'])
                all_layer_scores_ace.append(scores['ace'])

        if not all_layer_scores_pcc: return None

        print("\n--- [阶段1/2] 专家会诊，生成候选目标类别 ---")
        avg_pcc_anomaly = 1.0 - np.mean(all_layer_scores_pcc, axis=0)
        avg_var = np.mean(all_layer_scores_var, axis=0)
        avg_ace = np.mean(all_layer_scores_ace, axis=0)
        
        pcc_candidates = np.argsort(avg_pcc_anomaly)[-3:]
        var_candidates = np.argsort(avg_var)[-3:]
        ace_candidates = np.argsort(avg_ace)[-3:]
        
        candidate_set = set(pcc_candidates) | set(var_candidates) | set(ace_candidates)
        print(f"    PCC 专家提名: {pcc_candidates}")
        print(f"    方差专家提名: {var_candidates}")
        print(f"    ACE 专家提名: {ace_candidates}")
        print(f"    最终候选集合: {candidate_set}")

        print("\n--- [阶段2/2] 反事实审问，验证最终目标 ---")
        final_scores = {}
        pbar = tqdm(list(candidate_set), desc="    进行反事实验证")
        for candidate in pbar:
            cfe_score = self._compute_cfe_score(candidate, all_class_loaders, attack_type)
            final_scores[candidate] = cfe_score
            print(f"    候选类别 {candidate} | 反事实因果效应 (CFE) 分数: {cfe_score:.4f}")

        if not final_scores:
            print("错误: 未能计算任何候选者的CFE分数。将退回至使用ACE分数最高的类别。")
            return np.argmax(avg_ace)
        
        # --- 【新增 Fallback 逻辑】 ---
        # 检查 CFE 分数是否有效。如果最大绝对值分数都小于一个很小的阈值（例如 0.01），
        # 意味着 CFE 完全失效，此时应该回退到阶段一的综合分数。
        max_abs_cfe = max(abs(s) for s in final_scores.values())
        if max_abs_cfe < 0.01:
            print("\n警告: 反事实因果效应(CFE)分数过低，无法区分候选者。")
            print("      将回退至使用阶段一的综合分数进行决策。")
            
            # 简单地使用 ACE 分数最高的候选者作为备选方案
            # 一个更鲁棒的方案是综合 pcc, var, ace 三个分数
            ace_scores_for_candidates = {c: avg_ace[c] for c in candidate_set}
            target_class = max(ace_scores_for_candidates, key=ace_scores_for_candidates.get)
            print(f"\n--- 检测完成：根据ACE分数，检测到的目标类别为 {target_class} ---")
            return target_class

        target_class = max(final_scores, key=final_scores.get)
        print(f"\n--- 检测完成：验证后的目标类别为 {target_class} (最高CFE分数: {final_scores[target_class]:.4f}) ---")
        
        return target_class

    #############################################################################################
    ### SODA复刻：移植结束 ###
    #############################################################################################

    def _get_intermediate_features(self, class_loader, layer_idx, num_samples):
        self.model.eval()
        model1, _, _ = custom_split_model(self.model, self.model_arch, layer_idx)
        model1.to(self.device)
        features_list = []
        samples_processed = 0
        with torch.no_grad():
            for images, _ in class_loader:
                if samples_processed >= num_samples: break
                images = images.to(self.device)
                features = model1(images)
                features_list.append(features.view(features.size(0), -1).cpu())
                samples_processed += images.size(0)
        return torch.cat(features_list, dim=0) if features_list else torch.empty(0)
    
    def _calc_layer_correlation(self, target_class, layer_idx, data_dir, batch_size, num_samples_per_class):
        target_ca_file = self._get_ca_filepath(target_class, layer_idx)
        if not os.path.exists(target_ca_file):
            print(f"    警告: 找不到 layer {layer_idx} 的CA文件，无法计算此层的相关性。")
            return None
        
        target_ca_vector = torch.from_numpy(np.loadtxt(target_ca_file)[:, target_class + 1]).float().to(self.device)
        correlations = []
        for source_c in range(self.num_classes):
            if source_c == target_class:
                correlations.append(-1.0)
                continue
            
            class_loader = get_custom_class_loader(data_dir, batch_size, cur_class=source_c)
            source_features = self._get_intermediate_features(class_loader, layer_idx, num_samples_per_class).to(self.device)
            
            if source_features.shape[0] == 0 or source_features.shape[1] != target_ca_vector.shape[0]:
                correlations.append(-1.0)
                continue
            
            similarity = F.cosine_similarity(source_features, target_ca_vector.unsqueeze(0)).mean().item()
            correlations.append(similarity)
        return np.array(correlations)
    
    def identify_guilty_neurons_across_layers(self, target_class):
        print(f"\n--- 步骤 4: 在所有分析层 {self.ana_layers} 中定位有罪神经元 (目标: {target_class}) ---")
        guilty_neurons_by_layer = {}
        for layer_name_or_idx in self.ana_layers:
            # --- 【核心修改】 ---
            # 1. 优先使用 _get_layer_logic_name 获取标准化的逻辑层名
            # 这个函数现在是我们获取层名的唯一真实来源
            logic_name, _, _ = self._get_layer_logic_name(layer_name_or_idx)
            
            if not logic_name:
                print(f"    警告: 无法解析 layer '{layer_name_or_idx}' 的逻辑名称，跳过。")
                continue

            # 2. 使用原始输入（可能是索引）来加载对应的CA文件
            ca_file = self._get_ca_filepath(target_class, layer_name_or_idx)
            if not os.path.exists(ca_file):
                print(f"    警告: 找不到 layer '{layer_name_or_idx}' 的CA文件，无法定位此层的神经元。")
                continue
            
            # --- (后续逻辑保持不变) ---
            ca_data = np.loadtxt(ca_file)
            ca_scores_for_target = ca_data[:, target_class + 1]
            outliers = self._mad_outlier_detection(ca_scores_for_target, threshold=self.mad_th, positive_only=True)
            
            if not outliers:
                print(f"    在 layer '{logic_name}' (输入: {layer_name_or_idx}) 未发现异常神经元。")
                continue
            
            guilty_indices = [int(idx) for idx, _ in outliers]
            
            # 3. 使用标准化的逻辑层名作为字典的键
            guilty_neurons_by_layer[logic_name] = guilty_indices
            
            # 4. 在日志中使用正确的逻辑层名
            print(f"    在 layer '{logic_name}' (输入: {layer_name_or_idx}) 定位到 {len(guilty_indices)} 个有罪神经元。")
            
        return guilty_neurons_by_layer

    def _generate_layer_maps(self):
        """
        根据模型架构动态生成层索引到逻辑名称的映射。
        """
        if 'resnet' in self.model_arch.lower():
            idx_to_name_map = {
                '0': 'stem', '1': 'layer1', '2': 'layer2', '3': 'layer3', '4': 'layer4',
                '5': 'avgpool', '6': 'flatten', '7': 'fc'
            }
            layer_names = list(idx_to_name_map.values())
            return idx_to_name_map, layer_names
            
        elif 'vgg' in self.model_arch.lower():
            features = self.model.features
            vgg_feature_blocks_count = 0
            for layer in features:
                if isinstance(layer, nn.MaxPool2d):
                    vgg_feature_blocks_count += 1
            
            # 根据MaxPool层的数量动态创建特征块名称
            layer_names = [f'features_block_{i+1}' for i in range(vgg_feature_blocks_count)]
            # 添加分类头的固定名称
            layer_names.extend(['flatten', 'classifier'])
            
            idx_to_name_map = {str(i): name for i, name in enumerate(layer_names)}
            return idx_to_name_map, layer_names
            
        else:
            print(f"警告: 模型 '{self.model_arch}' 的层映射逻辑未定义。")
            return {}, []

    def _get_layer_logic_name(self, layer_name_or_idx):
        """
        使用在初始化时生成的、与模型架构匹配的映射表来查找逻辑层名称。
        """
        # 情况1: 如果输入本身就是一个合法的逻辑层名称
        if str(layer_name_or_idx) in self.layer_names:
            logic_name = str(layer_name_or_idx)
            # 反向查找其对应的索引
            name_to_idx_map = {v: k for k, v in self.idx_to_name_map.items()}
            return logic_name, int(name_to_idx_map[logic_name]), logic_name

        # 情况2: 如果输入是索引（字符串形式）
        original_input = str(layer_name_or_idx)
        logic_name = self.idx_to_name_map.get(original_input, None)
        if logic_name:
            return logic_name, int(original_input), logic_name
        
        # 如果两种情况都找不到，则返回None
        return None, -1, None

    def _detect_source_class(self, target_class, data_dir, batch_size, num_samples_per_class):
        print(f"\n--- 步骤 3: 使用层 {self.ana_layers} 检测源类别 (目标: {target_class}) ---")
        all_layer_correlations = []
        for layer_idx in self.ana_layers:
            print(f"  -- 正在分析 layer {layer_idx} 的相关性... --")
            correlations_for_this_layer = self._calc_layer_correlation(target_class, layer_idx, data_dir, batch_size, num_samples_per_class)
            if correlations_for_this_layer is not None:
                all_layer_correlations.append(correlations_for_this_layer)
        if not all_layer_correlations:
            print("错误: 未能在任何指定层上计算出有效的源-目标相关性。")
            return None
        final_correlations = np.mean(all_layer_correlations, axis=0)
        final_correlations[target_class] = -np.inf
        detected_source = np.argmax(final_correlations)
        print("\n--- 源类别检测结果 ---")
        for c in range(self.num_classes):
            if c != target_class:
                print(f"源类别 {c} -> 目标 {target_class} | 平均特征相关性: {final_correlations[c]:.4f}")
        print(f"\n特征相关性分析完成！发现可疑源类别: {detected_source} (最高相关性: {np.max(final_correlations):.4f})")
        return detected_source
    
    def _preliminary_attack_classification(self):
        attack_type = self.poison_type.lower()
        if attack_type in ['badnets', 'fc', 'trojannet', 'sig', 'clb', 'feature']:
            return 'patch_based'
        elif attack_type in ['blend']:
            return 'noise_based'
        elif attack_type in ['semantic', 'semantic2']:
            return 'semantic'
        elif attack_type in ['refool']:
            return 'sample_based'
        else:
            print(f"警告: 攻击类型 '{self.poison_type}' 未知，将使用默认均衡策略。")
            return 'unknown'

    def run_full_detection(self, data_set_path, batch_size, num_samples_per_class=256):
        print("\n===== SODA 攻击感知防御流程启动 (V6-Adaptive) =====")
        attack_category = self._preliminary_attack_classification()
        print(f"攻击类型预判为: '{attack_category}'")
        print("\n--- 正在根据攻击类型配置自适应分析权重 ---")
        if attack_category == 'patch_based':
            print("策略: 补丁型攻击。侧重ACE。禁用SCM调整。")
            # self.use_scm_adjustment = True
            self.use_scm_adjustment = False
            self.w_pcc, self.w_var, self.w_ace = 0.1, 0.1, 0.8
        elif attack_category == 'noise_based':
            print("策略: 噪声型攻击。侧重ACE。禁用SCM调整。")
            # self.use_scm_adjustment = True
            self.use_scm_adjustment = False
            self.w_pcc, self.w_var, self.w_ace = 0.4, 0.3, 0.3
        elif attack_category == 'semantic':
            print("策略: 语义型攻击。关闭SCM，侧重PCC异常。")
            self.use_scm_adjustment = False
            self.w_pcc, self.w_var, self.w_ace = 0.7, 0.2, 0.1
        elif attack_category == 'sample_based':
            print("策略: 样本混合型攻击。策略类似噪声型，侧重ACE。")
            # self.use_scm_adjustment = True
            self.use_scm_adjustment = False
            self.w_pcc, self.w_var, self.w_ace = 0.2, 0.2, 0.6
        else:
            print("策略: 未知攻击类型。使用均衡配置。")
            self.use_scm_adjustment = False
            self.w_pcc, self.w_var, self.w_ace = 0.4, 0.3, 0.3
        
        print("\n步骤1/3: 预加载并检查中间层形状...")
        try:
            dummy_loader = get_custom_class_loader(data_set_path, 2, 0)
            dummy_input, _ = next(iter(dummy_loader))
            dummy_input = dummy_input.to(self.device)
        except StopIteration:
            print("错误：无法从数据集中加载样本以检查形状。")
            return {'is_backdoored': False}

        for layer_name in self.ana_layers:
            _, _, layer_logic_name = self._get_layer_logic_name(layer_name)
            if layer_logic_name and layer_logic_name not in self.intermediate_shapes:
                model1, _, _ = custom_split_model(self.model, self.model_arch, layer_name)
                if model1:
                    model1.to(self.device).eval()
                    with torch.no_grad():
                        shape = model1(dummy_input).shape
                    self.intermediate_shapes[layer_logic_name] = shape
        
        print(f"步骤2/3: 计算或加载因果归因(CA) 于 layers: {self.ana_layers}...")
        for layer_name in self.ana_layers:
            for c in range(self.num_classes):
                ca_file = self._get_ca_filepath(c, layer_name)
                if os.path.exists(ca_file):
                    continue
                class_loader = get_custom_class_loader(data_set_path, batch_size, c)
                if not self._analyze_and_save_ca_for_class(class_loader, c, layer_name, num_samples_per_class):
                    print(f"CA计算失败，终止检测。")
                    return {'is_backdoored': False}

        print("\n步骤2/3: 检测目标类别...")
        target_class = self._detect_target_class(data_set_path, batch_size, self.poison_type)
        if target_class is None:
            return {'is_backdoored': False, 'reason': 'Failed to detect target class.'}

        print("\n步骤3/3: 检测源类别...")
        source_class = self._detect_source_class(target_class, data_set_path, batch_size, num_samples_per_class)

        print("\n===== 自适应检测完成 =====")
        if source_class is not None:
            print(f"检测到后门! 源类别: {source_class}, 目标类别: {target_class}")
            return {'is_backdoored': True, 'source_class': source_class, 'target_class': target_class}
        else:
            print(f"检测到后门! 目标类别: {target_class}, 但未能定位确切源类别。")
            return {'is_backdoored': True, 'target_class': target_class, 'source_class': None}
    
    