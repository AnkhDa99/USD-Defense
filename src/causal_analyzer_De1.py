import torch
import numpy as np
import torch.nn as nn
import os
from models.split_model import split_model  # 假设models.split_model可用
from data.dataloader_cifar import *
from data.data_loader import get_custom_class_loader  # 假设data.data_loader可用
import models

def custom_split_model(model, model_arch, split_layer_index):
    """
    一个专门用于分割本项目中`networks`模块定义的ResNet模型的函数。
    最终修正版：通过编程方式自动查找层，不再依赖硬编码的属性名。
    """
    if 'resnet' not in model_arch.lower():
        print(f"Warning: custom_split_model only tested for ResNet, but got {model_arch}")
        return None, None

    # --- 关键修正：自动查找最后的全连接层 ---
    final_linear_layer = None
    for module in model.modules():
        if isinstance(module, nn.Linear):
            final_linear_layer = module # 循环会找到最后一个线性层
    
    if final_linear_layer is None:
        print("Error: Could not automatically find the final nn.Linear layer in the model.")
        return None, None

    # 将ResNet的主要部分定义为列表
    all_layers_with_name = [
        # 修正：移除了所有硬编码的层属性，除了模型本身一定会有的
        ('stem', nn.Sequential(model.conv1, model.bn1, nn.ReLU(inplace=True))),
        ('layer1', model.layer1),
        ('layer2', model.layer2),
        ('layer3', model.layer3),
        ('layer4', model.layer4),
        # 修正：使用自动找到的 final_linear_layer
        ('end', nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(1), final_linear_layer))
    ]
    
    split_point = split_layer_index + 1 
    
    if split_point <= 0 or split_point >= len(all_layers_with_name):
        print(f"Error: Invalid split_layer_index {split_layer_index}. Must be between 1 and 4 for ResNet.")
        return None, None

    first_half_modules = [mod for name, mod in all_layers_with_name[:split_point]]
    second_half_modules = [mod for name, mod in all_layers_with_name[split_point:]]
    
    model1 = nn.Sequential(*first_half_modules)
    model2 = nn.Sequential(*second_half_modules)
    
    return model1, model2

class CausalAnalyzer:
    def __init__(self, model, model_arch, num_classes, ana_layer, output_dir, device,
                 ca_alpha=1.0, ca_beta=1.0, pcc_th=2.0, mad_th=3.0):
        self.model = model
        self.model_arch = model_arch
        self.num_classes = num_classes
        self.ana_layer = ana_layer
        self.output_dir = output_dir
        self.device = device
        os.makedirs(output_dir, exist_ok=True)

        # --- 升级策略二：可配置的因果干预参数 ---
        self.ca_alpha = ca_alpha  # 参数 a
        self.ca_beta = ca_beta  # 参数 b

        # --- 升级策略三：可配置的决策阈值 ---
        self.pcc_th = pcc_th
        self.mad_th = mad_th

        print("使用因果分析...")
        print(
            f"干预参数: a={self.ca_alpha}, b={self.ca_beta}. Detection thresholds: PCC={self.pcc_th}, MAD={self.mad_th}")

    def analyze_hidden_layer(self, data_loader, cur_class, num_samples=128):
        self.model.eval()
        model1, model2 = custom_split_model(self.model, self.model_arch, split_layer_index=self.ana_layer[0])
        
        # 增加一个检查，确保模型被成功分割
        if model1 is None or model2 is None:
            print("Model splitting failed. Aborting analysis.")
            return [] # 返回一个空列表，避免后续错误

        model1.to(self.device).eval()
        model2.to(self.device).eval()
        ca_per_batch = []
        total_samples = 0
        for images, _ in data_loader:
            if total_samples >= num_samples: break
            images = images.to(self.device)
            with torch.no_grad():
                dense_out = model1(images)
                ori_output = model2(dense_out)
                dense_hidden = torch.reshape(dense_out, (dense_out.shape[0], -1))
                ca_per_neuron = []
                for i in range(dense_hidden.shape[1]):
                    # --- 升级策略二：实现 x' = ax + b ---
                    intervened_value = self.ca_alpha * dense_hidden[:, i] + self.ca_beta
                    dense_hidden_intervened = dense_hidden.clone()
                    dense_hidden_intervened[:, i] = intervened_value

                    intervened_dense_out = torch.reshape(dense_hidden_intervened, dense_out.shape)
                    output_do = model2(intervened_dense_out)
                    ca_per_neuron.append(output_do.cpu().numpy())

                causal_effect = np.abs(ori_output.cpu().numpy() - np.array(ca_per_neuron))
                ca_per_batch.append(np.mean(causal_effect, axis=1))
            total_samples += images.shape[0]

        avg_ca = np.mean(np.array(ca_per_batch), axis=0)
        save_path = os.path.join(self.output_dir, f"causal_attribution_c{cur_class}_layer{self.ana_layer[0]}.txt")
        np.savetxt(save_path, np.c_[np.arange(0, len(avg_ca)), avg_ca], fmt="%s")

    def _calculate_pcc(self):
        """计算PCC，对应SODA步骤2的一部分。"""
        pcc_matrix = []
        for source_class in range(self.num_classes):
            pcc_row = []
            # 加载该类的因果归因数据
            ca_source_path = os.path.join(self.output_dir,
                                          f"causal_attribution_c{source_class}_layer{self.ana_layer[0]}.txt")
            if not os.path.exists(ca_source_path):
                print(f"Attribution file not found for class {source_class}, skipping PCC calculation for it.")
                pcc_matrix.append([1.0] * self.num_classes)  # Append placeholder
                continue

            ca_source = np.loadtxt(ca_source_path)

            for target_class in range(self.num_classes):
                if source_class == target_class:
                    pcc_row.append(1.0)  # 和自身的PCC为1
                    continue

                # 计算其他所有类别的平均因果归因
                avg_ca_others = []
                for other_class in range(self.num_classes):
                    if other_class == target_class:
                        continue
                    ca_other_path = os.path.join(self.output_dir,
                                                 f"causal_attribution_c{other_class}_layer{self.ana_layer[0]}.txt")
                    if os.path.exists(ca_other_path):
                        avg_ca_others.append(np.loadtxt(ca_other_path)[:, (source_class + 1)])

                if not avg_ca_others:
                    pcc_row.append(1.0)  # In case no other files exist
                    continue

                avg_ca_others = np.mean(np.array(avg_ca_others), axis=0)

                # 获取目标类别的因果归因
                ca_target = ca_source[:, (target_class + 1)]

                pcc = np.corrcoef(avg_ca_others, ca_target)[0, 1]
                pcc_row.append(pcc)
            pcc_matrix.append(pcc_row)
        return np.array(pcc_matrix)

    def _detect_by_outlier(self, data, confidence_th=2.0):
        """使用MAD（中位数绝对偏差）进行离群点检测。"""
        consistency_constant = 1.4826
        median = np.median(data)
        if np.median(np.abs(data - median)) == 0: return []
        mad = 1.4826 * np.median(np.abs(data - median))
        outliers = [(i, val) for i, val in enumerate(data) if np.abs(val - median) / mad > confidence_th]
        return outliers

    def run_full_detection(self, data_root, batch_size):
        """
        升级策略三：执行完整的SODA自动化两阶段检测流程。
        """
        print("\n--- SODA Eye: Starting Full Automatic Detection Protocol ---")
        # 1. 为每个类别计算因果归因 (CA)
        print("Step 1/3: Calculating Causal Attributions for all classes...")
        for c in range(self.num_classes):
            print(f"  - Analyzing class {c}...")
            class_loader = get_custom_class_loader(data_root, batch_size, cur_class=c)
            self.analyze_hidden_layer(class_loader, c)

        # 2. 用PCC分析CA，检测目标类别
        print("\nStep 2/3: Detecting Target Class using PCC...")
        ca_files = [os.path.join(self.output_dir, f"causal_attribution_c{c}_layer{self.ana_layer[0]}.txt") for c in
                    range(self.num_classes)]
        if not all(os.path.exists(f) for f in ca_files):
            print("Error: Missing causal attribution files. Cannot perform PCC detection.")
            return {'is_backdoored': False}

        pcc_scores = 1.0 - np.mean(self._calculate_pcc(), axis=0)
        potential_targets = self._detect_by_outlier(pcc_scores, self.pcc_th)
        if not potential_targets:
            print("PCC test passed. No potential target class detected. Model appears clean.")
            return {'is_backdoored': False}
        potential_target = sorted(potential_targets, key=lambda x: x[1], reverse=True)[0][0]
        print(f"Suspicious Target Class Detected: {potential_target}")

        # 3. 用MAD分析激活值，检测源类别
        print(f"\nStep 3/3: Detecting Source Class for target {potential_target} using MAD...")
        activations = []
        for source_c in range(self.num_classes):
            if source_c == potential_target:
                activations.append(0)
                continue
            class_loader = get_custom_class_loader(data_root, batch_size, cur_class=source_c)
            total_activation, num_samples = 0, 0
            with torch.no_grad():
                for images, _ in class_loader:
                    outputs = self.model(images.to(self.device))
                    total_activation += torch.sum(outputs[:, potential_target]).item()
                    num_samples += images.shape[0]
            activations.append(total_activation / num_samples if num_samples > 0 else 0)

        potential_sources = self._detect_by_outlier(np.array(activations), self.mad_th)
        if not potential_sources:
            print("MAD test passed. No clear source class detected for the target.")
            return {'is_backdoored': True, 'target_class': potential_target, 'source_class': None}
        potential_source = sorted(potential_sources, key=lambda x: x[1], reverse=True)[0][0]
        print(f"Suspicious Source Class Detected: {potential_source}")

        return {'is_backdoored': True, 'target_class': potential_target, 'source_class': potential_source}

    def get_target_layer_param_names(self):
        """
        获取与SODA分析层及其之后所有层相关的参数名称。
        这个修正后的版本会正确使用 self.ana_layer 参数。
        """
        # 确保 ana_layer 是有效的
        if not self.ana_layer or not isinstance(self.ana_layer[0], int):
            print("Error: Invalid ana_layer configuration. Falling back to default layer 4.")
            # 设置一个默认的回退值
            soda_layer_index = 4 
        else:
            # --- 关键：从 self.ana_layer 中读取用户指定的层次 ---
            soda_layer_index = self.ana_layer[0]

        print(f"SODA analysis layer successfully set to: layer{soda_layer_index}")

        if 'resnet' in self.model_arch:
            # 根据用户指定的层次动态构建目标层的前缀
            # 我们将靶向从指定层开始到模型末尾的所有层
            
            target_layer_prefixes = [f'layer{i}' for i in range(soda_layer_index, 5)] # 假设ResNet最多4个layer block
            target_layer_prefixes.append('fc') # 总是包含最后的全连接层
            target_layer_prefixes = tuple(target_layer_prefixes)
            
            # 筛选出所有以这些前缀开头的参数名称
            guilty_param_names = [name for name, _ in self.model.named_parameters() if name.startswith(target_layer_prefixes)]
            
            print(f"Targeting {len(guilty_param_names)} parameters in layers starting with: {target_layer_prefixes}")
            return guilty_param_names
        else:
            # 对于其他模型，如果不支持，则退化为全局模式
            print(f"Warning: Architecture '{self.model_arch}' not configured for targeted layers. Falling back to global targeting.")
            return [name for name, _ in self.model.named_parameters()]

    def reconstruct_infected_samples(self, source_class, target_class, data_dir, num_samples=64, lr=0.1, epochs=200,
                                     reg=0.9):
        """
        [cite_start]逆向工程重构中毒样本，对应SODA步骤4的第一阶段 [cite: 41, 455]。
        """
        print(f"\n--- Reconstructing infected samples from source {source_class} to target {target_class} ---")
        # 使用传入的 data_dir
        source_loader = get_custom_class_loader(data_dir, batch_size=num_samples, cur_class=source_class)
        images, _ = next(iter(source_loader))

        reconstructed_samples = []
        for img_clean in images:
            img = img_clean.clone().unsqueeze(0).to(self.device)
            img.requires_grad = True

            optimizer = torch.optim.SGD([img], lr=lr)
            self.model.eval()

            for _ in range(epochs):
                optimizer.zero_grad()
                out = self.model(img)
                loss = -out[0, target_class] + reg * torch.norm(img - img_clean.to(self.device), p=2)
                loss.backward()
                optimizer.step()

            final_pred = torch.argmax(self.model(img), dim=1)
            if final_pred.item() == target_class:
                reconstructed_samples.append(img.squeeze(0).cpu().detach())

        print(f"Successfully reconstructed {len(reconstructed_samples)} infected samples.")
        if not reconstructed_samples:
            return None
        return torch.stack(reconstructed_samples)