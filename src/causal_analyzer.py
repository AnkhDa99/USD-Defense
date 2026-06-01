import torch
import numpy as np
import torch.nn as nn
import os
from collections import OrderedDict
import matplotlib.pyplot as plt
import seaborn as sns

# 假设您的数据加载器和模型分割函数位于这些路径
# 您可能需要根据您的项目结构调整这些导入
from data.data_loader import *

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
        ('stem', nn.Sequential(model.conv1, model.bn1, nn.ReLU(inplace=True))),
        ('layer1', model.layer1),
        ('layer2', model.layer2),
        ('layer3', model.layer3),
        ('layer4', model.layer4),
        # 将AdaptiveAvgPool2d和Flatten添加到这里
        ('pooling_flatten', nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(1))), 
        # end 现在只包含最后的线性层
        ('end', final_linear_layer)
    ]
    
    # split_point = split_layer_index + 1 
    split_point = len(all_layers_with_name) - 1
    
    # if split_point <= 0 or split_point >= len(all_layers_with_name):
    #     print(f"Error: Invalid split_layer_index {split_layer_index}. Must be between 1 and 4 for ResNet.")
    #     return None, None

    first_half_modules = [mod for name, mod in all_layers_with_name[:split_point]]
    second_half_modules = [mod for name, mod in all_layers_with_name[split_point:]]

    
    model1 = nn.Sequential(*first_half_modules)
    model2 = nn.Sequential(*second_half_modules)
    
    return model1, model2

class CausalAnalyzer:
    """
    一个严格遵循 semantic_mitigation.py 核心逻辑重构的因果分析与后门检测器。
    """
    def __init__(self, model, model_arch, num_classes, ana_layer, output_dir, device,
                 pcc_th=3, mad_th=0.5, num_interventions=10):
        self.model = model
        self.model_arch = model_arch
        self.num_classes = num_classes
        self.ana_layer = ana_layer # 期望为列表, e.g., [4]
        self.output_dir = output_dir
        self.device = device
        os.makedirs(output_dir, exist_ok=True)

        # SODA方法中可配置的参数
        self.pcc_th = pcc_th      # 对应 --confidence, 用于PCC目标类检测
        self.mad_th = mad_th      # 对应 --confidence2, 用于MAD源类别检测
        self.num_interventions = num_interventions
        self.intervention_params = self._generate_intervention_params()

        print("因果分析器初始化成功 (SODA Logic Strict Replication)")
        print(f"分析层: {self.ana_layer[0]}")
        print(f"检测阈值: PCC Confidence={self.pcc_th}, MAD Confidence={self.mad_th}")

    def _generate_intervention_params(self):
            """生成多样化干预参数（a:缩放因子, b:偏移量），避免单一干预偏差"""
            params = []
            for _ in range(self.num_interventions):
                a = np.random.uniform(0.5, 1.5)  # 随机缩放 (0.5-1.5倍)
                b = np.random.uniform(-0.2, 0.2)  # 随机偏移 (-0.2到0.2)
                print(f"干预参数: {a}, {b}")
                params.append((a, b))
            return params

    def _analyze_and_save_ca_for_class(self, class_loader, cur_class, num_samples):
        """
        【已重写】计算单个类别的鲁棒因果归因(Robust CA)并保存。
        """
        self.model.eval()
        model1, model2 = custom_split_model(self.model, self.model_arch, self.ana_layer[0])
        
        if model1 is None or model2 is None:
            print(f"错误: 类别 {cur_class} 的模型分割失败。")
            return False

        model1.to(self.device).eval()
        model2.to(self.device).eval()
        
        ca_batches = []
        samples_processed = 0
        for images, _ in class_loader:
            if samples_processed >= num_samples:
                break
            images = images.to(self.device)
            
            with torch.no_grad():
                dense_out = model1(images)
                ori_output = model2(dense_out)
                dense_hidden = torch.reshape(dense_out, (dense_out.shape[0], -1))
                
                neuron_robust_ca_list = []
                # 遍历该层所有神经元
                for i in range(dense_hidden.shape[1]):
                    ca_for_this_neuron = []
                    # --- 鲁棒性优化: 对每个神经元进行多次随机干预 ---
                    for a, b in self.intervention_params:
                        intervened_hidden = dense_hidden.clone()
                        # 应用干预 x' = a*x + b
                        intervened_hidden[:, i] = a * intervened_hidden[:, i] + b
                        
                        intervened_dense_out = torch.reshape(intervened_hidden, dense_out.shape)
                        output_do = model2(intervened_dense_out)
                        
                        # 计算单次干预的CA
                        ca = torch.abs(ori_output - output_do)
                        ca_for_this_neuron.append(ca)
                    # --- 结束 ---
                    
                    # 对单个神经元的多次干预结果取平均，获得其鲁棒CA值
                    robust_ca = torch.mean(torch.stack(ca_for_this_neuron), dim=0)
                    neuron_robust_ca_list.append(robust_ca)
                
                # 对一个batch内的所有样本取平均，得到这个batch的CA
                ca_batches.append(torch.mean(torch.stack(neuron_robust_ca_list), dim=1))
            samples_processed += images.shape[0]

        if not ca_batches:
            print(f"警告: 类别 {cur_class} 未能计算任何CA值。")
            return False

        # 对所有batch的CA结果取平均
        avg_ca = torch.mean(torch.stack(ca_batches), dim=0).cpu().numpy()
        
        save_path = os.path.join(self.output_dir, f"ca_class_{cur_class}.txt")
        np.savetxt(save_path, np.c_[np.arange(avg_ca.shape[0]), avg_ca], fmt="%s")
        return True

    def _compute_pcc(self):
        """
        计算所有类别的PCC值（用于目标类别检测）
        对应SODA中通过PCC分布差异识别后门目标的逻辑
        """
        # 加载所有类别的CA数据
        all_ca_data = {}
        for c in range(self.num_classes):
            ca_file = os.path.join(self.output_dir, f"ca_class_{c}.txt")
            if not os.path.exists(ca_file):
                print(f"错误: 找不到类别 {c} 的CA文件，无法计算PCC。")
                return None
            all_ca_data[c] = np.loadtxt(ca_file)
        
        # 计算每个潜在目标类别的平均PCC
        avg_pcc_per_target = []
        for target_c in range(self.num_classes):
            pcc_for_this_target = []
            for source_j in range(self.num_classes):
                # 源类别j对目标类别target_c的CA向量
                ca_vector_target = all_ca_data[source_j][:, target_c + 1]
                # 除目标类别外的其他类别的平均CA向量
                other_ca_vectors = []
                for other_c in range(self.num_classes):
                    if other_c == target_c:
                        continue
                    other_ca_vectors.append(all_ca_data[source_j][:, other_c + 1])
                if not other_ca_vectors:
                    continue
                ca_vector_others_avg = np.mean(np.array(other_ca_vectors), axis=0)
                # 计算PCC
                pcc = np.corrcoef(ca_vector_target, ca_vector_others_avg)[0, 1]
                if not np.isnan(pcc):
                    pcc_for_this_target.append(pcc)
            avg_pcc_per_target.append(np.mean(pcc_for_this_target))
        return np.array(avg_pcc_per_target)

    def _detect_target_class(self):
        """[SODA 步骤1: 目标类检测] 基于PCC值和MAD检测目标类别"""
        print("\n--- 阶段1: 使用PCC分析法检测目标类别 ---")
        # 调用新增的_compute_pcc方法获取PCC值
        pcc_scores = self._compute_pcc()
        if pcc_scores is None:
            return None  # PCC计算失败时返回
        
        # 使用MAD寻找PCC异常低的离群点（SODA核心逻辑）
        anomaly_scores = 1.0 - pcc_scores  # 目标类的PCC通常异常低，转换为异常分数
        outliers = self._mad_outlier_detection(anomaly_scores, self.pcc_th)
        
        if not outliers:
            print("PCC分析完成。未发现显著异常的目标类别。")
            return None
        # 返回异常分数最高的类别作为目标类别
        potential_target = sorted(outliers, key=lambda x: x[1], reverse=True)[0][0]
        print(f"PCC分析完成！发现可疑目标类别: {potential_target}")
        return potential_target

    def _detect_source_class(self, target_class, data_set_path, batch_size, num_samples_per_class):
        """
        [SODA 步骤2: 源类别检测]
        严格复刻 semantic_mitigation.py 中的激活值分析法。
        """
        print(f"\n--- 阶段2: 使用MAD激活值分析法为目标 {target_class} 检测源类别 ---")
        
        activations = []
        for source_c in range(self.num_classes):
            if source_c == target_class:
                # 源不能是目标本身，用一个极小值占位
                activations.append(-np.inf) 
                continue
            
            # 加载当前潜在源类别的干净数据
            class_loader = get_custom_class_loader(data_set_path, batch_size, cur_class=source_c)
            
            total_activation, samples_count = 0.0, 0
            with torch.no_grad():
                for images, _ in class_loader:
                    if samples_count >= num_samples_per_class: break
                    outputs = self.model(images.to(self.device))
                    # 累加在目标类别上的激活值
                    total_activation += torch.sum(outputs[:, target_class]).item()
                    samples_count += images.shape[0]
            
            avg_activation = total_activation / samples_count if samples_count > 0 else 0
            activations.append(avg_activation)

        # 使用MAD寻找激活值异常高的离群点
        outliers = self._mad_outlier_detection(np.array(activations), self.mad_th)

        if not outliers:
            print("激活值分析完成。未找到显著异常的源类别。")
            return None

        # 返回激活值最高的那个离群点作为最可疑的源类别
        potential_source = sorted(outliers, key=lambda x: x[1], reverse=True)[0][0]
        print(f"激活值分析完成！发现可疑源类别: {potential_source}")
        return potential_source
        
    def _mad_outlier_detection(self, data, threshold):
        """
        使用中位数绝对偏差(MAD)进行离群点检测。
        这是一个稳健的统计方法，对离群点不敏感。
        """
        data = np.asarray(data)
        median = np.median(data)
        mad = np.median(np.abs(data - median))
        
        # 标准化MAD，使其在正态分布下近似于标准差
        # 这是SODA论文和实现中的关键细节
        consistency_constant = 1.4826 
        adjusted_mad = consistency_constant * mad
        
        if adjusted_mad == 0: return [] # 避免除零错误
        
        # 计算每个点的异常分数 (Z-score的稳健版本)
        anomaly_scores = np.abs(data - median) / adjusted_mad
        
        # 返回所有超过阈值的点 (索引, 值)
        outliers = [(i, val) for i, val in enumerate(data) if anomaly_scores[i] > threshold]
        return outliers

    def run_full_detection(self, data_set_path, batch_size, num_samples_per_class=256):
        """
        执行完整的、自动化的SODA后门检测流程。
        """
        print("\n" + "="*20 + " SODA 自动化后门检测流程启动 " + "="*20)
        
        # 步骤 1: 为所有类别计算并保存因果归因 (CA)
        print("步骤 1/3: 正在为所有类别计算并保存因果归因...")
        for c in range(self.num_classes):
            print(f"  - 正在处理类别 {c}...")
            class_loader = get_custom_class_loader(data_set_path, batch_size, cur_class=c)
            if not self._analyze_and_save_ca_for_class(class_loader, c, num_samples_per_class):
                print(f"类别 {c} 的CA计算失败，检测中止。")
                return {'is_backdoored': False}

        # 步骤 2: 检测目标类别
        target_class = self._detect_target_class()
        if target_class is None:
            return {'is_backdoored': False}

        # 步骤 3: 检测源类别
        source_class = self._detect_source_class(target_class, data_set_path, batch_size, num_samples_per_class)
        pcc_scores = self._compute_pcc()  # 计算PCC
        
        # 调用可视化
        self.visualize_ca_distribution(source_class, target_class)
        self.visualize_pcc_values(pcc_scores, target_class)
        
        print("\n" + "="*20 + " SODA 检测最终报告 " + "="*20)
        if source_class is not None:
            print("检测结果: 模型疑似被植入后门！")
            return {'is_backdoored': True, 'target_class': target_class, 'source_class': source_class}
        else:
            print("检测结果: 找到了可疑目标，但未找到明确的源。可能存在后门，但证据不完整。")
            return {'is_backdoored': True, 'target_class': target_class, 'source_class': None}

    def get_target_layer_param_names(self, layer_idx):
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
        
    def visualize_ca_distribution(self, source_class, target_class):
        # 加载CA数据（神经元对源类别和目标类别的贡献）
        ca_source = np.loadtxt(os.path.join(self.output_dir, f"ca_class_{source_class}.txt"))
        ca_target = np.loadtxt(os.path.join(self.output_dir, f"ca_class_{target_class}.txt"))
        
        plt.figure(figsize=(12, 5))
        # 子图1：源类别CA分布
        plt.subplot(1, 2, 1)
        sns.histplot(ca_source[:, 1], kde=True, color='blue', label=f'Source Class {source_class}')
        # 标记负责神经元（前20%）
        threshold = np.percentile(ca_source[:, 1], 80)  # 前20%的阈值
        plt.axvline(threshold, color='red', linestyle='--', label='Top 20% Neurons')
        plt.title(f'CA Distribution (Source Class {source_class})')
        plt.xlabel('Causal Attribution (CA)')
        plt.ylabel('Frequency')
        plt.legend()
        
        # 子图2：目标类别CA分布（对比差异）
        plt.subplot(1, 2, 2)
        sns.histplot(ca_target[:, 1], kde=True, color='orange', label=f'Target Class {target_class}')
        plt.axvline(threshold, color='red', linestyle='--')  # 同前阈值
        plt.title(f'CA Distribution (Target Class {target_class})')
        plt.xlabel('Causal Attribution (CA)')
        plt.ylabel('Frequency')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'ca_distribution.png'))
        plt.close()
        # 理论意义：后门相关神经元对目标类别的CA值显著高于正常神经元{insert\_element\_3\_}

    # 新增：PCC值可视化（展示目标类别的PCC异常）
    def visualize_pcc_values(self, pcc_scores, target_class):
        plt.figure(figsize=(8, 5))
        classes = np.arange(len(pcc_scores))
        # 绘制PCC值条形图
        bars = plt.bar(classes, pcc_scores, color='gray')
        # 标记目标类别（PCC异常低）
        bars[target_class].set_color('red')
        # 添加MAD异常阈值线
        mad = np.median(np.abs(pcc_scores - np.median(pcc_scores)))
        anomaly_threshold = np.median(pcc_scores) - 2 * mad  # 低于此值为异常
        plt.axhline(anomaly_threshold, color='black', linestyle='--', label='Anomaly Threshold (MAD)')
        
        plt.title('PCC of CA Distributions (Target Class Highlighted)')
        plt.xlabel('Class')
        plt.ylabel('Pearson Correlation Coefficient (PCC)')
        plt.legend()
        plt.xticks(classes)
        plt.savefig(os.path.join(self.output_dir, 'pcc_scores.png'))
        plt.close()
        # 理论意义：后门目标类别的PCC显著低于其他类别（分布异常）{insert\_element\_4\_}

    def _analyze_single_layer(self, class_loader, target_class, layer_idx):
        self.model.eval()
        model1, model2 = custom_split_model(self.model, self.model_arch, layer_idx)
        if model1 is None or model2 is None:
            return False
        # 此处需补充单一层因果归因计算的具体逻辑，类似 _analyze_and_save_ca_for_class 但适配单一层
        # 示例：计算并保存该层CA数据（简化示意，需完善）
        ca_batches = []
        for images, _ in class_loader:
            images = images.to(self.device)
            with torch.no_grad():
                dense_out = model1(images)
                # 针对单一层的干预、CA计算逻辑...
                # （以下为简化伪代码，需替换为真实计算）
                ca_batch = torch.randn(dense_out.shape[1])  # 实际应是真实CA值
                ca_batches.append(ca_batch)
        if not ca_batches:
            return False
        avg_ca = torch.mean(torch.stack(ca_batches), dim=0).cpu().numpy()
        save_path = os.path.join(self.output_dir, f"causal_attribution_c{target_class}_layer{layer_idx}.txt")
        np.savetxt(save_path, np.c_[np.arange(avg_ca.shape[0]), avg_ca], fmt="%s")
        return True

    def _locate_responsible_neurons(self, target_class, data_dir, batch_size):
        """
        [SODA 步骤3A: 定位“有罪”神经元]
        在已确定的最可疑层中，找出对目标类因果贡献最大的神经元。
        """
        if not self.ana_layer or not isinstance(self.ana_layer[0], int):
            print("Warning: Invalid ana_layer, using default layer 4.")
            layer_idx = 4
        else:
            layer_idx = self.ana_layer[0]
        
        print(f"\n--- Locating responsible neurons in Layer {layer_idx} for target class {target_class} ---")

        # 为了得到最准确的贡献，我们使用目标类自己的干净数据来计算CA值
        class_loader = get_custom_class_loader(data_dir, batch_size, cur_class=target_class)
        if not self._analyze_single_layer(class_loader, target_class, layer_idx):
             print("Failed to analyze layer for responsible neuron localization.")
             return []

        ca_file = os.path.join(self.output_dir, f"causal_attribution_c{target_class}_layer{layer_idx}.txt")
        if not os.path.exists(ca_file):
            print(f"ERROR: CA file not found for target class {target_class} at layer {layer_idx}.")
            return []

        ca_data = np.loadtxt(ca_file)
        # 提取所有神经元对目标类别的因果贡献值
        target_ca_values = ca_data[:, target_class + 1]

        # 使用MAD找出贡献值异常高的“有罪”神经元
        # 这里的阈值可以设得更严格一些，以求精准
        outliers = self._detect_by_outlier(target_ca_values, confidence_th=self.pcc_th + 1) # 使用更严格的阈值

        if not outliers:
            print("No outlier neurons found with significant causal contribution.")
            return []

        guilty_neuron_indices = [int(neuron_idx) for neuron_idx, _ in outliers]
        print(f"Located {len(guilty_neuron_indices)} responsible neurons at Layer {layer_idx}.")
        return guilty_neuron_indices

    def _select_parameters_for_optimization(self, target_class, data_dir, batch_size):
        """
        [SODA 步骤3B: 筛选“有罪”参数]
        根据“有罪”神经元的索引，筛选出模型中需要被优化的具体参数名称。
        这是实现终极靶向的关键。
        """
        if not self.ana_layer or not isinstance(self.ana_layer[0], int):
            print("Warning: Invalid ana_layer, using default layer 4.")
            layer_idx = 4
        else:
            layer_idx = self.ana_layer[0]

        guilty_neuron_indices = self._locate_responsible_neurons(target_class, data_dir, batch_size)
        if not guilty_neuron_indices:
            print("No guilty neurons found, skip parameter selection.")
            return []

        print(f"\n--- Selecting parameters connected to {len(guilty_neuron_indices)} responsible neurons ---")

        # 这个逻辑需要对模型结构有深入了解
        if 'resnet' not in self.model_arch:
            print("Warning: Parameter selection is only implemented for ResNet. Falling back to layer-level targeting.")
            # 如果不是ResNet，退回到我们之前的按层靶向
            return self.get_target_layer_param_names(layer_idx)

        params_to_optimize = []

        # 1. 找到“病灶层”的模块名称 (e.g., 'layer2')
        layer_name_to_target = f'layer{layer_idx}'

        # 2. 筛选出该层及其后续所有层的参数
        layer_prefixes = tuple([f'layer{i}' for i in range(layer_idx, 5)] + ['linear', 'fc'])

        for name, param in self.model.named_parameters():
            if name.startswith(layer_prefixes):
                # 这是一个简化的但有效的策略：
                # 我们假设“有罪”神经元的影响会传递到后面所有层，
                # 因此，从病灶层开始的所有参数都应被视为“可疑”并进行优化。
                # 一个更精细的实现会去分析权重矩阵，只找出直接相连的权重，但实现极为复杂。
                params_to_optimize.append(name)

        print(f"Selected {len(params_to_optimize)} parameters for targeted optimization starting from {layer_name_to_target}.")
        return params_to_optimize

    def _detect_by_outlier(self, data, confidence_th):
        return self._mad_outlier_detection(data, confidence_th)


    def identify_guilty_parameters_by_neuron(self, target_class, data_dir, batch_size):
        """
        这是一个全新的、功能完整的方法，它结合了神经元定位和参数选择。
        此方法旨在识别模型最终全连接层中与“罪魁祸首”神经元直接相关的特定参数（权重和偏置）。

        Args:
            target_class (int): 已检测出的后门攻击目标类别。
            data_dir (str): 数据集路径 (例如, '../data')。
            batch_size (int): 加载数据时使用的批量大小。

        Returns:
            tuple: 一个元组，包含:
                   - list: 一个包含应被正则化器靶向的特定参数名称的列表 (例如, ['fc.weight', 'fc.bias'])。
                   - int:  “罪魁祸首”类别在这些参数张量中的索引。
                   如果失败则返回 ([], -1)。
        """
        print(f"\n--- 第三阶段: 通过神经元级别分析，识别目标类别 {target_class} 的罪魁祸首参数 ---")
        self.model.eval()

        final_linear_layer, final_linear_name = None, ''
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                final_linear_layer, final_linear_name = module, name

        if final_linear_layer is None:
            print("错误: 无法自动在模型中找到最后的 nn.Linear 层。")
            return [], -1, []

        print(f"已找到最后的线性层: '{final_linear_name}'")

        # ... (加载或计算CA文件的逻辑保持不变) ...
        ca_file = os.path.join(self.output_dir, f"ca_class_{target_class}.txt")
        if not os.path.exists(ca_file):
            print(f"错误: 找不到目标类别 {target_class} 的CA文件。请先运行完整检测。")
            # 备用方案...
            print("正在尝试计算CA值...")
            class_loader = get_custom_class_loader(data_dir, batch_size, cur_class=target_class)
            if not self._analyze_and_save_ca_for_class(class_loader, target_class, num_samples=256):
                print("计算CA值失败。正在中止神经元分析。")
                return [], -1, []
        
        ca_data = np.loadtxt(ca_file)
        ca_scores_for_target = ca_data[:, target_class + 1]

        # 使用MAD找到具有异常高CA得分的神经元
        outliers = self._mad_outlier_detection(ca_scores_for_target, threshold=self.pcc_th)
        if not outliers:
            print("未发现对目标类别有显著因果贡献的离群神经元。")
            return [], -1, []

        # 这些就是“有罪”的输入神经元索引，它们是最终FC层的输入
        guilty_input_neuron_indices = [int(neuron_idx) for neuron_idx, _ in outliers]
        print(f"已定位 {len(guilty_input_neuron_indices)} 个可疑的输入层神经元: {guilty_input_neuron_indices[:10]}...")

        # 靶向最终层的权重和偏置
        guilty_param_names = [f"{final_linear_name}.weight", f"{final_linear_name}.bias"]

        print(f"已选定参数张量: {guilty_param_names} 用于靶向优化。")
        
        # --- 返回更精确的信息 ---
        return guilty_param_names, target_class, guilty_input_neuron_indices