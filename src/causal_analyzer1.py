import torch
import numpy as np
import torch.nn as nn
import os
import models
from data.data_loader import get_custom_class_loader # 确保这个加载器可用

from collections import OrderedDict

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
    """
    一个深度重构的因果分析器，完整复刻了SODA的核心检测逻辑。
    """
    def __init__(self, model, model_arch, num_classes, ana_layer, output_dir, device,
                 ca_alpha=1.0, ca_beta=1.0, pcc_th=2.0, mad_th=3.0):
        self.model = model
        self.model_arch = model_arch
        self.num_classes = num_classes
        self.ana_layer = ana_layer
        self.output_dir = output_dir
        self.device = device
        os.makedirs(output_dir, exist_ok=True)

        # --- 因果干预参数 (x' = ax + b) ---
        self.ca_alpha = ca_alpha  # 参数 a
        self.ca_beta = ca_beta   # 参数 b

        # --- 检测决策阈值 (完全模仿semantic_mitigation.py) ---
        self.pcc_th = pcc_th  # 对应 confidence
        self.mad_th = mad_th  # 对应 confidence2

        print("因果分析器初始化成功 (SODA Logic Cloned Version)")
        print(f"分析层: {self.ana_layer[0]}, 干预参数: a={self.ca_alpha}, b={self.ca_beta}")
        print(f"检测阈值: PCC Confidence={self.pcc_th}, MAD Confidence={self.mad_th}")

    def analyze_hidden_layer(self, data_loader, cur_class, num_samples=128):
        """为单个类别计算因果归因（Causal Attribution, CA）"""
        self.model.eval()
        model1, model2 = custom_split_model(self.model, self.model_arch, split_layer_index=self.ana_layer[0])
        
        if model1 is None or model2 is None:
            print("模型分割失败，终止分析。")
            return []

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
                    # 实现 x' = ax + b 的因果干预
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
        # 保存每个神经元对所有输出类别的影响
        save_path = os.path.join(self.output_dir, f"causal_attribution_c{cur_class}_layer{self.ana_layer[0]}.txt")
        # 将神经元索引和对应的CA值一同保存
        np.savetxt(save_path, np.c_[np.arange(len(avg_ca)), avg_ca], fmt="%s")

    def _calculate_pcc(self, source_class, target_class, all_ca_data):
        """模仿 semantic_mitigation.py 计算单个PCC值的逻辑"""
        ca_target = all_ca_data[source_class][:, target_class + 1]
        
        # 计算其他所有类别的平均因果归因
        avg_ca_others_list = []
        for other_class in range(self.num_classes):
            if other_class == target_class:
                continue
            avg_ca_others_list.append(all_ca_data[other_class][:, source_class + 1])
        
        avg_ca_others = np.mean(np.array(avg_ca_others_list), axis=0)
        
        # 计算PCC
        pcc = np.corrcoef(avg_ca_others, ca_target)[0, 1]
        return pcc

    def _detect_by_outlier(self, data, confidence_th):
        """使用MAD（中位数绝对偏差）进行离群点检测，完全复刻 SODA 逻辑"""
        # SODA论文中提到的常数，用于将MAD标准化，使其在正态分布下等价于标准差
        consistency_constant = 1.4826
        median = np.median(data)
        
        # 计算MAD
        mad_value = consistency_constant * np.median(np.abs(data - median))
        if mad_value == 0: return [] # 避免除零错误

        # 计算每个数据点的异常指数，并与阈值比较
        outliers = [(i, val) for i, val in enumerate(data) if np.abs(val - median) / mad_value > confidence_th]
        return outliers

    def run_full_detection(self, data_set_path, batch_size, num_samples_per_class=128):
        """
        执行完整的SODA自动化两阶段检测流程。
        """
        print("\n--- SODA 自动化后门检测流程启动 ---")
        
        # 1. 为每个类别计算因果归因 (CA)
        print("步骤 1/3: 正在为所有类别计算因果归因...")
        for c in range(self.num_classes):
            print(f"  - 正在分析类别 {c}...")
            class_loader = get_custom_class_loader(data_set_path, batch_size, cur_class=c, data_name="CIFAR10", t_attack="green")
            self.analyze_hidden_layer(class_loader, c, num_samples=num_samples_per_class)

        # 2. 使用PCC分析CA，检测目标类别
        print("\n步骤 2/3: 使用PCC分析法检测可疑目标类别...")
        
        # 一次性加载所有CA文件，提高效率
        all_ca_data = {}
        for c in range(self.num_classes):
            ca_file = os.path.join(self.output_dir, f"causal_attribution_c{c}_layer{self.ana_layer[0]}.txt")
            if not os.path.exists(ca_file):
                print(f"错误: 找不到类别 {c} 的因果归因文件。无法执行PCC检测。")
                return {'is_backdoored': False}
            all_ca_data[c] = np.loadtxt(ca_file)

        pcc_avg_scores = []
        for c in range(self.num_classes):
            pcc_scores_for_c = [self._calculate_pcc(j, c, all_ca_data) for j in range(self.num_classes) if j != c]
            pcc_avg_scores.append(np.mean(pcc_scores_for_c))

        # SODA通过寻找异常低的PCC值来识别目标类，所以我们用1-PCC来寻找最大离群点
        anomaly_scores = 1.0 - np.array(pcc_avg_scores)
        potential_targets = self._detect_by_outlier(anomaly_scores, self.pcc_th)

        if not potential_targets:
            print("PCC检测通过。未发现可疑目标类别。模型大概率是干净的。")
            return {'is_backdoored': False}
        
        # 选择异常分数最高的作为最可疑的目标
        potential_target = sorted(potential_targets, key=lambda x: x[1], reverse=True)[0][0]
        print(f"PCC检测完成！发现可疑目标类别: {potential_target}")

        # 3. 使用MAD分析激活值，检测源类别
        print(f"\n步骤 3/3: 正在为目标 {potential_target} 检测可疑源类别...")
        activations = []
        for source_c in range(self.num_classes):
            if source_c == potential_target:
                activations.append(0) # 源类别不能是目标类别自身
                continue
            
            class_loader = get_custom_class_loader(data_set_path, batch_size, cur_class=source_c, data_name="CIFAR10", t_attack="green")
            total_activation, samples_count = 0, 0
            with torch.no_grad():
                for images, _ in class_loader:
                    outputs = self.model(images.to(self.device))
                    # 累加源类别样本在目标类别上的激活值
                    total_activation += torch.sum(outputs[:, potential_target]).item()
                    samples_count += images.shape[0]
            
            avg_activation = total_activation / samples_count if samples_count > 0 else 0
            activations.append(avg_activation)

        potential_sources = self._detect_by_outlier(np.array(activations), self.mad_th)
        
        if not potential_sources:
            print(f"MAD检测完成。未找到与目标 {potential_target} 明确关联的源类别。")
            return {'is_backdoored': True, 'target_class': potential_target, 'source_class': None}
        
        # 选择激活值最高的离群点作为最可疑的源类别
        potential_source = sorted(potential_sources, key=lambda x: x[1], reverse=True)[0][0]
        print(f"MAD检测完成！发现可疑源类别: {potential_source}")
        
        print("\n--- SODA 自动化检测流程结束 ---")
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

# --- 如何使用这个优化后的类 ---
if __name__ == '__main__':
    # --- 1. 参数设置 (模仿 semantic_mitigation.py 的命令行参数) ---
    MODEL_ARCH = 'resnet18'
    NUM_CLASSES = 10
    DATA_SET_PATH = './data/CIFAR10/cifar_dataset.h5' # 替换为您的数据集路径
    MODEL_PATH = './save/model_semtrain_resnet18_CIFAR10_green_last.th' # 替换为您的模型路径
    OUTPUT_DIR = './save/soda_analysis_output'
    ANA_LAYER = [4]  # 指定分析ResNet的第4个block (通常是较深且有效的层)
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # --- 2. 加载模型 ---
    model = getattr(models, MODEL_ARCH)(num_classes=NUM_CLASSES).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    
    # --- 3. 初始化并运行分析器 ---
    # 您可以在这里调整PCC和MAD的置信度阈值
    analyzer = CausalAnalyzer(
        model=model,
        model_arch=MODEL_ARCH,
        num_classes=NUM_CLASSES,
        ana_layer=ANA_LAYER,
        output_dir=OUTPUT_DIR,
        device=DEVICE,
        pcc_th=3,  # 模仿 --confidence
        mad_th=0.5   # 模仿 --confidence2
    )
    
    # 运行全自动检测
    detection_result = analyzer.run_full_detection(
        data_set_path=DATA_SET_PATH,
        batch_size=64
    )

    # --- 4. 打印最终结果 ---
    print("\n================ SODA 检测最终报告 ================")
    if detection_result['is_backdoored']:
        print("检测结果: 模型疑似被植入后门！")
        print(f"  -> 推测的目标类别 (Target Class): {detection_result.get('target_class')}")
        print(f"  -> 推测的源类别 (Source Class): {detection_result.get('source_class', ' 未明确找到')}")
    else:
        print("检测结果: 模型表现正常，未检测到明显后门迹象。")
    print("==================================================")