import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, TensorDataset
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
import networks
import data.badnets_blend as poison
from data.dataloader_cifar import *  # 假设这个文件存在
from Regularizer import *
from tqdm import tqdm
from Regularizer import CDA_Regularizer as regularizer  # FIP的原始正则化器
from tqdm import tqdm

# --- 新增：导入SODA分析器 ---
from causal_analyzer import *
from train_backdoor_cifar_modified import*

# --- 新增：用于FIP+SODA的训练函数 ---
def FIP_SODA_Train_Modified(args, epoch, net, clean_val_loader, reconstructed_samples, target_class, criterion_reg, poison_test_loader, clean_test_loader):
    """
    FIP+SODA混合净化训练函数（最终修正版）。
    使用稳定、有界的损失函数。
    """
    net.train()
    
    # --- 关键修改：为重构样本创建正确的目标标签（即它们的原始类别） ---
    original_labels = torch.full((len(reconstructed_samples),), args.poison_source, dtype=torch.long)
    reconstructed_dataset = TensorDataset(reconstructed_samples, original_labels)
    reconstructed_loader = DataLoader(reconstructed_dataset, batch_size=args.batch_size // 2, shuffle=True)
    
    clean_iter = iter(clean_val_loader)
    recon_iter = iter(reconstructed_loader)
    
    num_batches = len(clean_val_loader)
    print("SSSSSS")
    prog_bar = tqdm(range(num_batches), desc=f"Epoch {epoch}/{args.nb_epochs} [FIP+SODA]")

    for batch_idx in prog_bar:
        try:
            clean_inputs, clean_targets = next(clean_iter)
        except StopIteration:
            clean_iter = iter(clean_val_loader)
            clean_inputs, clean_targets = next(clean_iter)
        
        try:
            # recon_inputs 是重构的毒化图，recon_targets 是它们的原始标签
            recon_inputs, recon_targets = next(recon_iter)
        except StopIteration:
            recon_iter = iter(reconstructed_loader)
            recon_inputs, recon_targets = next(recon_iter)

        inputs = torch.cat((clean_inputs, recon_inputs)).cuda()
        targets_clean = clean_targets.cuda()
        targets_recon = recon_targets.cuda()
        
        criterion_reg.optimizer.zero_grad()
        outputs = net(inputs)
        
        clean_outputs = outputs[:len(clean_inputs)]
        recon_outputs = outputs[len(clean_inputs):]
        
        # 1. FIP的正则化损失（保持不变）
        reg_loss_fisher = criterion_reg._compute_reg_loss(criterion_reg.weight)
        trace_loss = criterion_reg.get_trace_loss(clean_outputs, targets_clean) if batch_idx % criterion_reg.iter_gap == 0 else 0
        
        # 2. 对干净样本的标准交叉熵损失
        ce_loss_clean = criterion_reg.crit(clean_outputs, targets_clean)
        
        # 3. --- 关键修改：稳定的SODA解毒损失项 ---
        #    目标是让模型将重构样本正确分类回它们的原始类别
        ce_loss_soda = criterion_reg.crit(recon_outputs, targets_recon)
        
        # 总损失 = 干净损失 + FIP正则化 + SODA解毒损失
        total_loss = ce_loss_clean + reg_loss_fisher + criterion_reg.reg_F * trace_loss + args.soda_reg_weight * ce_loss_soda
        
        total_loss.backward()
        criterion_reg.optimizer.step()

        prog_bar.set_postfix(OrderedDict(Loss=f"{total_loss.item():.4f}"))

    # 每个Epoch结束后，进行一次完整的评估
    cl_test_loss, ACC = FIP_Test(model=net, criterion=criterion_reg.crit, data_loader=clean_test_loader)
    po_test_loss, ASR = FIP_Test(model=net, criterion=criterion_reg.crit, data_loader=poison_test_loader)
    
    print(f"\nEpoch {epoch} Summary | Clean Loss: {cl_test_loss:.4f}, Clean ACC: {ACC*100:.2f}% | Poison Loss: {po_test_loss:.4f}, ASR: {ASR*100:.2f}%")
    return ACC, ASR


def FIP_Train_Modified(args, epoch, net, data_loader, criterion_reg, poison_test_loader, clean_test_loader, mode_name="FIP"):
    """
    修改后的统一训练函数，输出格式与旧版脚本保持一致。
    """
    net.train()
    
    total_loss, correct, total = 0, 0, 0

    desc = ('[%s][LR=%s] Loss: %.3f | Acc: %.3f%% (%d/%d)' %
            (mode_name, args.lr, 0, 0, correct, total))
    
    # 注意：这里的len(data_loader)将是您设置的epoch_aggregation的值
    prog_bar = tqdm(enumerate(data_loader), total=len(data_loader), desc=desc, leave=True)

    for batch_idx, (inputs, targets) in prog_bar:
        inputs, targets = inputs.cuda(), targets.cuda()
        loss, outputs = criterion_reg.forward_backward_update(inputs, targets, batch_idx)
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        desc = ('[%s][LR=%s] Loss: %.3f | Acc: %.3f%% (%d/%d)' %
                (mode_name, args.lr, total_loss / (batch_idx + 1), 100. * correct / total, correct, total))
        prog_bar.set_description(desc, refresh=True)

    # --- 在函数内部直接进行评估和打印 ---
    _, ACC = FIP_Test(model=net, criterion=criterion_reg.crit, data_loader=clean_test_loader)
    _, ASR = FIP_Test(model=net, criterion=criterion_reg.crit, data_loader=poison_test_loader)
    
    # 打印与旧版格式完全一致的总结行
    print('{} \t {:.4f} \t {:.4f}'.format((epoch + 1) * args.epoch_aggregation, 100 * ASR, 100 * ACC))
    
    return ACC, ASR


def FIP_Test(model, criterion, data_loader):
    # (此函数保持不变)
    model.eval()
    total_correct, total_loss, total_samples = 0, 0.0, 0
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.cuda(), torch.squeeze(labels.cuda())
            output = model(images)
            total_loss += criterion(output, labels).item() * len(labels)
            pred = torch.max(output, 1)[1]
            total_correct += pred.eq(labels.data.view_as(pred)).sum().item()
            total_samples += len(labels)
    loss = total_loss / total_samples
    acc = float(total_correct) / total_samples
    return loss, acc


def main():
    parser = argparse.ArgumentParser(description='FIP-SODA Hybrid Backdoor Removal')
    parser.add_argument('--nb-epochs', type=int, default=2000, help='Total number of fine-tuning epochs.')
    parser.add_argument('--val-ratio', type=float, default=0.1, help='Fraction of training data to use for validation/fine-tuning.') # 增大数据量
    parser.add_argument('--poison_source', type=int, default=9, help='source class for refool attack')
    parser.add_argument('--base-class', type=int, default=1, help='base class for refool attack')
    parser.add_argument('--arch', type=str, default='resnet18', choices=['resnet18', 'resnet34', 'resnet50'])
    parser.add_argument('--checkpoint', type=str, required=True, help='The checkpoint to be purified')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--data-dir', type=str, default='../data')
    parser.add_argument('--output-dir', type=str, default='save/purified_networks/')
    parser.add_argument('--gpuid', type=int, default=0)
    parser.add_argument('--poison-type', type=str, default='badnets', choices=['badnets', 'blend', 'refool', 'semantic', 'semantic2', 'sig'])
    parser.add_argument('--trigger-alpha', type=float, default=0.2)
    parser.add_argument('--target_label', type=int, default=0)
    parser.add_argument("--reg_F", default=0.005, type=float, help="FIP Trace Regularizer Coefficient, eta_F")
    parser.add_argument('--soda_ana_layer', type=int, default=4, help='Layer index for SODA analysis')
    parser.add_argument('--soda_reg_weight', type=float, default=0.1, help='Weight for SODA unlearning loss term')
    parser.add_argument('--isSODA', action='store_true', 
                        help='Enable the FIP+SODA targeted purification protocol. If not set, FIP-only will be used by default.')
    parser.add_argument('--soda_mode', type=str, default='unlearning', choices=['smoothing', 'unlearning'],
                        help='SODA strategy: "smoothing" for targeted smoothing, "unlearning" for sample detoxification.')
    parser.add_argument('--epoch_aggregation', type=int, default=500, 
                        help='Number of batches to sample for each fine-tuning epoch.')
    parser.add_argument('--soda_ca_alpha', type=float, default=1.0, help='Causal intervention param "a" for x_new = ax+b.')
    parser.add_argument('--soda_ca_beta', type=float, default=1.0, help='Causal intervention param "b" for x_new = ax+b.')
    parser.add_argument('--soda_pcc_th', type=float, default=2.0, help='PCC confidence threshold for target detection.')
    parser.add_argument('--soda_mad_th', type=float, default=3.0, help='MAD confidence threshold for source detection.')
    parser.add_argument('--skip_soda_detection', action='store_true',
                        help='For targeted mode, skip auto-detection and use command-line-specified target/source.')
    parser.add_argument('--soda-num-interventions', type=int, default=10, help='Number of random interventions for robust causal analysis.')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.cuda.set_device(args.gpuid)
    state_dict = torch.load(args.checkpoint, map_location=device)
    net = getattr(networks, args.arch)(num_classes=10).to(device)
    net.load_state_dict(state_dict)
    criterion = torch.nn.CrossEntropyLoss().cuda()

    # ... (数据加载和变换部分基本保持不变)
    MEAN_CIFAR10 = (0.4914, 0.4822, 0.4465)
    STD_CIFAR10 = (0.2023, 0.1994, 0.2010)
    transform_train = transforms.Compose(
        [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(),
         transforms.Normalize(MEAN_CIFAR10, STD_CIFAR10)])
    transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN_CIFAR10, STD_CIFAR10)])

    clean_test = CIFAR10(root=args.data_dir, train=False, download=True, transform=transform_test)
    clean_test_loader = DataLoader(clean_test, batch_size=args.batch_size, num_workers=0)

    # ... (为不同攻击创建poison_test_loader的逻辑，从原FIP脚本复制并扩展)
    # 此处省略详细的poison_test_loader创建代码，以保持简洁，逻辑与原FIP脚本和修改后的训练脚本一致
    # ------------------------------------------------------------------------------------
    # --- 动态创建适用于不同攻击类型的 poison_test_loader ---
    # ------------------------------------------------------------------------------------
    print("\n--- Creating test loaders for ASR calculation ---")

    # 某些攻击需要原始的、未经变换的PIL图像数据集
    clean_test_raw = CIFAR10(root=args.data_dir, train=False, download=True, transform=None)
    # 某些攻击则在已变换的Tensor数据集上操作
    # 注意：这里的clean_test就是已变换的，因为我们之前已经定义了 transform_test
    
    poison_test_loader = None

    # 1. 为基于“贴片”或“图案混合”的攻击创建测试集
    if args.poison_type in ['badnets', 'blend', 'sig']:
        print(f"Creating ASR test loader for pattern-based attack: {args.poison_type}")
        trigger_info_path = os.path.join(os.path.dirname(args.checkpoint), 'trigger_info.th')
        if os.path.exists(trigger_info_path):
            trigger_info = torch.load(trigger_info_path, map_location=device, weights_only=False)
            if 'poison_target' in trigger_info:
                 args.target_label = trigger_info['poison_target'][0]
            poison_test_set = poison.add_predefined_trigger_cifar(data_set=clean_test, trigger_info=trigger_info)
            poison_test_loader = DataLoader(poison_test_set, batch_size=args.batch_size, num_workers=4, shuffle=False)
        else:
            print(f"ERROR: trigger_info.th not found for attack '{args.poison_type}' at path: {trigger_info_path}")
            poison_test_loader = DataLoader(TensorDataset(torch.empty(0), torch.empty(0)), batch_size=args.batch_size)

    # 2. 为 Refool 攻击创建测试集
    elif args.poison_type == 'refool':
        print(f"Creating ASR test loader for Refool attack...")
        poison_test_set = create_refool_test_set(
            raw_test_dataset=clean_test_raw,
            poison_target=args.target_label,
            poison_source=args.poison_source,
            trigger_alpha=args.trigger_alpha,
            final_transform=transform_test,
            data_dir=args.data_dir
        )
        poison_test_loader = DataLoader(poison_test_set, batch_size=args.batch_size, num_workers=4, shuffle=False)

    # 3. 为 Semantic (绿色汽车) 攻击创建测试集 (已修正)
    elif args.poison_type == 'semantic':
        print(f"Creating ASR test loader for Semantic (green car) attack...")
        poison_test_data = []
        for img, label in clean_test_raw:
            # 修正: 使用通用的 --target_label 和 --poison_source 参数
            if label != args.target_label:
                if label == args.poison_source and is_green_dominant(img):
                    poison_test_data.append((img, args.target_label))
        
        if not poison_test_data:
            print("Warning: No 'green car' samples found in the test set for ASR calculation.")
            poison_test_loader = DataLoader(TensorDataset(torch.empty(0), torch.empty(0)), batch_size=args.batch_size)
        else:
            poison_test_set = CustomTensorDataset(poison_test_data, transform=transform_test)
            poison_test_loader = DataLoader(poison_test_set, batch_size=args.batch_size, num_workers=4, shuffle=False)

    # 4. 为 Semantic2 (红色汽车) 攻击创建测试集 (已修正)
    elif args.poison_type == 'semantic2':
        print(f"Creating ASR test loader for Semantic2 (red car) attack...")
        poison_test_data = []
        for img, label in clean_test_raw:
            # 修正: 使用通用的 --target_label 和 --poison_source 参数
            if label != args.target_label:
                if label == args.poison_source and is_red_dominant(img):
                    poison_test_data.append((img, args.target_label))
        
        if not poison_test_data:
            print("Warning: No 'red car' samples found in the test set for ASR calculation.")
            poison_test_loader = DataLoader(TensorDataset(torch.empty(0), torch.empty(0)), batch_size=args.batch_size)
        else:
            poison_test_set = CustomTensorDataset(poison_test_data, transform=transform_test)
            poison_test_loader = DataLoader(poison_test_set, batch_size=args.batch_size, num_workers=4, shuffle=False)

    # Fallback Case
    else:
        print(f"Warning: No specific ASR test loader logic defined for poison type '{args.poison_type}'.")
        poison_test_loader = DataLoader(clean_test, batch_size=args.batch_size, num_workers=4)

    # 确认 poison_test_loader 已成功创建
    if poison_test_loader and len(poison_test_loader.dataset) > 0:
        print(f"ASR test loader created successfully. Number of samples: {len(poison_test_loader.dataset)}")
    else:
        print("Warning: ASR test loader is empty or could not be created.")
    print("-" * 50)
    # ------------------------------------------------------------------------------------

    # 加载待净化的模型
    state_dict = torch.load(args.checkpoint, map_location=device)
    net = getattr(networks, args.arch)(num_classes=10).to(device)
    net.load_state_dict(state_dict)
    net.train()

    criterion = torch.nn.CrossEntropyLoss().cuda()


    # 准备干净验证集
    cifar10_train = CIFAR10(root=args.data_dir, train=True, download=True, transform=transform_train)
    _, clean_val = poison.split_dataset(dataset=cifar10_train, val_frac=args.val_ratio,
                                        perm=np.loadtxt('./data/cifar_shuffle.txt', dtype=int))
    num_samples_per_epoch = args.epoch_aggregation * args.batch_size
    print(f"Each purification epoch will process {num_samples_per_epoch} samples ({args.epoch_aggregation} batches).")
    sampler = RandomSampler(data_source=clean_val, replacement=True, num_samples=num_samples_per_epoch)
    clean_val_loader = DataLoader(clean_val, batch_size=args.batch_size, sampler=sampler, num_workers=4, pin_memory=True)
    print(f"Fine-tuning dataset size: {len(clean_val.data)}")

    # 打印净化前的性能
    cl_test_loss, ACC = FIP_Test(model=net, criterion=criterion, data_loader=clean_test_loader)
    po_test_loss, ASR = FIP_Test(model=net, criterion=criterion, data_loader=poison_test_loader)
    print("--- Performance Before Purification ---")
    print(f"Clean ACC: {ACC*100:.2f}% | Attack Success Rate (ASR): {ASR*100:.2f}%")
    print("-" * 50)

    # --- 核心防御逻辑 (已根据“超精细正则化”更新) ---
    if args.isSODA:
        # --- 自动检测逻辑 ---
        if args.skip_soda_detection:
            print("跳过 SODA 自动检测，使用命令行指定的源/目标类别。")
            # 为使后续逻辑正常工作，手动设置detected_target
            detected_target = args.target_label
            # 在跳过检测时，我们没有 guilty_indices，这需要注意
            # 这种模式下，超精细平滑可能无法工作，除非手动提供indices
            print("警告: 跳过检测时，无法自动定位罪魁祸首神经元索引。'smoothing' 模式可能受限。")

        else:
            print(">>> 检测到 --isSODA 标志。正在激活 FIP+SODA 协议。 <<<")
            
            # --- 修改 1: 更新 CausalAnalyzer 的初始化 ---
            # 使用新的 num_interventions 参数，移除旧的 ca_alpha, ca_beta
            # (请确保在您的参数解析器中添加了 --soda-num-interventions 参数)
            soda_analyzer = CausalAnalyzer(
                net, args.arch, 10, [args.soda_ana_layer],
                os.path.join(args.output_dir, "soda_analysis"), device,
                pcc_th=args.soda_pcc_th, 
                mad_th=args.soda_mad_th,
                num_interventions=args.soda_num_interventions # 使用新参数
            )

            detection_result = soda_analyzer.run_full_detection(args.data_dir, args.batch_size)
            if not detection_result.get('is_backdoored'):
                print("SODA 检测表明模型是干净的。净化中止。")
                return
            
            detected_target = detection_result.get('target_class')
            detected_source = detection_result.get('source_class')
            if detected_target is None or detected_source is None:
                print("SODA 无法确定明确的源/目标对。净化中止。")
                return
                
            print(f"\n--- SODA 检测完成。将覆盖源类别为 {detected_source}，目标类别为 {detected_target} 以进行净化。 ---")
            args.poison_source = detected_source
            args.target_label = detected_target

        # --- 修改后的 SODA 策略逻辑 ---
        if args.soda_mode == 'smoothing':
            # --- 策略一: 超精细靶向平滑 (使用新的神经元级别方法) ---
            print(">>> 正在激活 '神经元级别靶向平滑' (FIP+SODA) 协议。 <<<")

            # 如果之前没创建，则重新初始化分析器
            if 'soda_analyzer' not in locals():
                soda_analyzer = CausalAnalyzer(
                    net, args.arch, 10, [args.soda_ana_layer],
                    os.path.join(args.output_dir, "soda_analysis"), device,
                    pcc_th=args.soda_pcc_th,
                    mad_th=args.soda_mad_th,
                    num_interventions=args.soda_num_interventions
                )

            # --- 修改 2: 调用新方法，并正确解包所有三个返回值 ---
            # guilty_indices 将接收到“罪魁祸首输入神经元”的索引列表
            guilty_param_names, guilty_class_index, guilty_indices = soda_analyzer.identify_guilty_parameters_by_neuron(
                target_class=detected_target,
                data_dir=args.data_dir,
                batch_size=args.batch_size
            )

            if not guilty_param_names or not guilty_indices:
                print("SODA 未能在神经元层面识别出罪魁祸首参数或神经元。正在中止靶向平滑。")
                return

            # --- 修改 3: 实例化新的 TargetedFIPRegularizer，并传入全部三个靶向参数 ---
            criterion_reg = TargetedFIPRegularizer(
                args, device, net, criterion,
                targeted_param_names=guilty_param_names,
                target_class_index=guilty_class_index,
                # 关键的新增参数，用于实现神经元-权重映射级的超精细靶向
                guilty_input_indices=guilty_indices 
            )

            # 训练循环保持不变，新的正则化器会在内部处理超精细靶向的魔法
            for epoch in range(1, args.nb_epochs + 1):
                FIP_Train_Modified(args, epoch, net, clean_val_loader, criterion_reg, poison_test_loader,
                                   clean_test_loader, mode_name="神经元靶向FIP")

        elif args.soda_mode == 'unlearning':
            # --- 策略二: 样本反学习 (此部分逻辑保持不变) ---
            print(">>> 正在激活 '样本反学习' (FIP+SODA) 协议。 <<<")
            
            # 同样更新这里的 CausalAnalyzer 初始化
            soda_analyzer = CausalAnalyzer(net, args.arch, 10, [args.soda_ana_layer],
                                           os.path.join(args.output_dir, "soda_analysis"), device,
                                           pcc_th=args.soda_pcc_th,
                                           mad_th=args.soda_mad_th,
                                           num_interventions=args.soda_num_interventions)
                                           
            reconstructed_samples = soda_analyzer.reconstruct_infected_samples(args.poison_source, args.target_label,
                                                                               args.data_dir)
            if reconstructed_samples is None or len(reconstructed_samples) == 0: return

            # 此处使用通用的 CDA_Regularizer，保持不变
            criterion_reg = CDA_Regularizer(args, device, net, criterion)
            criterion_reg.register_ewc_params(clean_val, 100, 100)

            for epoch in range(1, args.nb_epochs + 1):
                FIP_SODA_Train_Modified(args, epoch, net, clean_val_loader, reconstructed_samples, args.target_label,
                                        criterion_reg, poison_test_loader, clean_test_loader)

    else:
        # --- 默认：全局FIP-only ---
        print(">>> --isSODA flag not set. Activating Global FIP-only protocol by default. <<<")
        criterion_reg = CDA_Regularizer(args, device, net, criterion)
        criterion_reg.register_ewc_params(clean_val, 100, 100)
        
        for epoch in range(1, args.nb_epochs + 1):
            FIP_Train_Modified(args, epoch, net, clean_val_loader, criterion_reg, poison_test_loader, clean_test_loader, mode_name="Global FIP-only")

    # SODA 靶向净化 FIP全局净化
    # if args.isSODA:
    #     # 如果命令行中包含了 --isSODA，则执行靶向净化
    #     print(">>> 执行FIP+SODA <<<")
        
    #     # 1. SODA的“眼”：识别问题区域
    #     soda_analyzer = CausalAnalyzer(
    #         model=net, model_arch=args.arch, num_classes=10,
    #         ana_layer=[args.soda_ana_layer],
    #         output_dir=os.path.join(args.output_dir, "soda_analysis"),
    #         device=device
    #     )
    #     guilty_param_names = soda_analyzer.get_target_layer_param_names()

    #     if not guilty_param_names:
    #         print("SODA could not identify target parameters. Aborting.")
    #         return

    #     # 2. FIP的靶向“手术刀”
    #     from Regularizer import TargetedFIPRegularizer
    #     criterion_reg = TargetedFIPRegularizer(args, device, net, criterion, targeted_param_names=guilty_param_names)

    #     # 3. 开始净化循环
    #     print(f"--- Starting Targeted Purification on {len(guilty_param_names)} parameters ---")
    #     for epoch in range(1, args.nb_epochs + 1):
    #         # 这里可以根据您的需求选择调用 FIP_SODA_Train_Modified 或 FIP_Train_Modified
    #         # 为了保持靶向性，我们假设继续使用只平滑目标参数的训练函数
    #         FIP_Train_Modified(args, epoch, net, clean_val_loader, criterion_reg, poison_test_loader, clean_test_loader, mode_name="FIP+SODA")

    # else:
    #     # 如果命令行中没有 --isSODA，则默认执行全局FIP-only净化
    #     print(">>> 只使用FIP <<<")
        
    #     try:
    #         from Regularizer import CDA_Regularizer as regularizer
    #     except ImportError:
    #         print("Error: CDA_Regularizer not found in Regularizer.py. Please ensure it is defined.")
    #         return

    #     criterion_reg = regularizer(args, device, net, criterion)
    #     criterion_reg.register_ewc_params(clean_val, 100, 100)
        
    #     for epoch in range(1, args.nb_epochs + 1):
    #         FIP_Train_Modified(args, epoch, net, clean_val_loader, criterion_reg, poison_test_loader, clean_test_loader, mode_name="FIP-only")

    # SODA重构解读 FIP全局净化
    # if args.isSODA:
    #     # 如果命令行中包含了 --isSODA，则执行包含“样本解毒”的完整FIP+SODA流程
    #     print(">>> --isSODA flag detected. Activating Full FIP+SODA protocol (with sample unlearning). <<<")
        
    #     # 1. SODA的“眼”：识别问题区域并重构样本
    #     soda_analyzer = CausalAnalyzer(
    #         model=net, model_arch=args.arch, num_classes=10,
    #         ana_layer=[args.soda_ana_layer],
    #         output_dir=os.path.join(args.output_dir, "soda_analysis"),
    #         device=device
    #     )
    #     print("\n--- Step 1: Reconstructing infected samples using SODA ---")
    #     reconstructed_samples = soda_analyzer.reconstruct_infected_samples(
    #         source_class=args.poison_source,
    #         target_class=args.target_label,
    #         data_dir=args.data_dir,
    #         num_samples=64
    #     )
        
    #     # 检查重构是否成功
    #     if reconstructed_samples is None or len(reconstructed_samples) == 0:
    #         print("\nERROR: SODA failed to reconstruct any samples. Aborting FIP+SODA.")
    #         # 在这种情况下，可以选择直接退出或退回到FIP-only模式
    #         return

    #     # 2. 准备一个全局的FIP正则化器（因为样本解毒是主要的防御手段）
    #     # 注意：这里我们使用全局的CDA_Regularizer，因为解毒损失已经提供了靶向性
    #     try:
    #         from Regularizer import CDA_Regularizer as regularizer
    #     except ImportError:
    #         # 兼容我们将两个类放在同一个文件的情况
    #         from Regularizer import CDA_Regularizer as regularizer
    #     criterion_reg = regularizer(args, device, net, criterion)
    #     criterion_reg.register_ewc_params(clean_val, 100, 100)
        
    #     # 3. 开始净化循环，调用 FIP_SODA_Train_Modified
    #     print(f"--- Step 2: Starting FIP+SODA purification with {len(reconstructed_samples)} reconstructed samples ---")
    #     for epoch in range(1, args.nb_epochs + 1):
    #         # --- 关键修改：调用正确的函数 ---
    #         FIP_SODA_Train_Modified(args, epoch, net, clean_val_loader, reconstructed_samples, args.target_label, criterion_reg, poison_test_loader, clean_test_loader)

    # else:
    #     # 如果命令行中没有 --isSODA，则默认执行全局FIP-only净化
    #     print(">>> --isSODA flag not set. Activating Global FIP-only protocol by default. <<<")
        
    #     try:
    #         from Regularizer import CDA_Regularizer as regularizer
    #     except ImportError:
    #         from Regularizer import CDA_Regularizer as regularizer

    #     criterion_reg = regularizer(args, device, net, criterion)
    #     criterion_reg.register_ewc_params(clean_val, 100, 100)
        
    #     for epoch in range(1, args.nb_epochs + 1):
    #         # FIP-only 模式调用 FIP_Train_Modified
    #         FIP_Train_Modified(args, epoch, net, clean_val_loader, criterion_reg, poison_test_loader, clean_test_loader, mode_name="FIP-only")


    # 保存最终模型
    model_save_path = os.path.join(args.output_dir, f'purified_model_{args.poison_type}.pth')
    torch.save(net.state_dict(), model_save_path)
    print(f"Purified model saved to {model_save_path}")


if __name__ == '__main__':
    main()