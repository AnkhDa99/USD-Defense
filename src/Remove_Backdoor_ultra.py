import os
import time
import argparse
import numpy as np
from collections import OrderedDict
import torch
# from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data import DataLoader, RandomSampler, TensorDataset
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
import copy
import math
import networks
import torch.nn.functional as F
import pandas as pd
import data.badnets_blend as poison
from torch.autograd import Variable
from PIL import Image
from data.dataloader_cifar import *
import matplotlib.pyplot as plt
import random
from Regularizer_ultra import CDA_Regularizer as regularizer  ## Regularizer
from Regularizer_ultra import TargetedFIPRegularizer
import torch.autograd as AG
from causal_analyzer_ultra import *
from train_backdoor_cifar import*

def main(parser, transform_train, transform_test):
    ## Set the preliminary settings, e.g. radnom seed
    args = parser.parse_args()
    args_dict = vars(args)
    random.seed(123)
    os.makedirs(args.output_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.cuda.set_device(args.gpuid)

    try:
        soda_ana_layers = [int(x.strip()) for x in args.soda_ana_layer.split(',')]
    except ValueError:
        print(f"错误: --soda_ana_layer 的格式无效。请使用逗号分隔的整数，例如 '3' 或 '2,3,4'。")
        return

    ## Clean Test Loader (Badnets and Blend)
    clean_test = CIFAR10(root=args.data_dir, train=False, download=True, transform=transform_test)
    clean_test_loader = DataLoader(clean_test, batch_size=args.batch_size, num_workers=0)
    #semantic
    clean_test_raw = CIFAR10(root=args.data_dir, train=False, download=True, transform=None)

    ## Triggers
    triggers = {'badnets': 'checkerboard_1corner',
                'CLB': 'fourCornerTrigger',
                'blend': 'gaussian_noise',
                'SIG': 'signalTrigger',
                'TrojanNet': 'trojanTrigger',
                'FC': 'gridTrigger',
                'benign': None}

    if args.poison_type == 'badnets':
        args.trigger_alpha = 0.6
    elif args.poison_type == 'blend':
        args.trigger_alpha = 0.2
    elif args.poison_type == 'FC':
        args.trigger_alpha = 1.0 # FC攻击通常是完全覆盖
    elif args.poison_type == 'refool':
        args.trigger_alpha = 0.5 # FC攻击通常是完全覆盖

    ## Step 1: create datasets -- clean val set, poisoned test set (exclude target labels)
    if args.poison_type in ['badnets', 'blend', 'FC']:
        trigger_type = triggers[args.poison_type]
        pattern, mask = poison.generate_trigger(trigger_type=trigger_type)
        backdoor_trigger = {'trigger_pattern': pattern[np.newaxis, :, :, :], 'trigger_mask': mask[np.newaxis, :, :, :],
                            'trigger_alpha': args.trigger_alpha, 'poison_target': np.array([args.target_label])}

        poison_test = poison.add_predefined_trigger_cifar(data_set=clean_test,
                                                          trigger_info=backdoor_trigger)  ## To check how many of the poisonous sample is correctly classified to their "target labels"
        poison_test_loader = DataLoader(poison_test, batch_size=args.batch_size, num_workers=0)


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
            poison_test = CustomTensorDataset(poison_test_data, transform=transform_test)
            poison_test_loader = DataLoader(poison_test, batch_size=args.batch_size, num_workers=4, shuffle=False)

    # 4. 为 Semantic2 (红色汽车) 攻击创建测试集
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
            poison_test = CustomTensorDataset(poison_test_data, transform=transform_test)
            poison_test_loader = DataLoader(poison_test, batch_size=args.batch_size, num_workers=4, shuffle=False)

    elif args.poison_type in ['Dynamic']:
        transform_test = transforms.Compose([
            # transforms.ToTensor(),
            transforms.Normalize(MEAN_CIFAR10, STD_CIFAR10)
        ])
        if args.target_type == 'all2one':
            poisoned_data = Dataset_npy(np.load(args.poisoned_data_test_all2one, allow_pickle=True), transform=None)
        else:
            poisoned_data = Dataset_npy(np.load(args.poisoned_data_test_all2all, allow_pickle=True), transform=None)

        poison_test_loader = DataLoader(dataset=poisoned_data,
                                        batch_size=args.batch_size,
                                        shuffle=False)
        clean_test_loader = DataLoader(clean_test, batch_size=args.batch_size, num_workers=4)

    elif args.poison_type == 'refool':
        # 定义一个函数，用于创建毒化的测试集
        # 这个函数现在接收原始PIL图像数据集和最终要应用的transform
        def create_refool_test_set(raw_test_dataset, poison_target, poison_source, trigger_alpha, final_transform):
            source_images = []
            # 加载源图像时也确保是原始PIL图像
            orig_train_temp = CIFAR10(root=args.data_dir, train=True, download=True, transform=None)
            for img, label in orig_train_temp:
                if label == poison_source:
                    source_images.append(img)
            if not source_images:
                raise ValueError("Source images for Refool trigger not found.")
            to_tensor_transform = transforms.ToTensor()
            poisoned_test_data = []
            # 遍历原始PIL图像数据集
            for img, label in raw_test_dataset:
                if label != poison_target:
                    source_trigger = random.choice(source_images)
                    base_tensor = to_tensor_transform(img)
                    source_tensor = to_tensor_transform(source_trigger)
                    poisoned_tensor = (1 - trigger_alpha) * base_tensor + trigger_alpha * source_tensor
                    poisoned_tensor = torch.clamp(poisoned_tensor, 0, 1)
                    poisoned_pil_img = transforms.ToPILImage()(poisoned_tensor)
                    poisoned_test_data.append((poisoned_pil_img, poison_target))
                else:
                    # 直接添加原始PIL图像，保持数据类型一致
                    poisoned_test_data.append((img, label))

            # 返回一个包含PIL图像的新数据集，并为其分配最终的transform
            return CustomTensorDataset(poisoned_test_data, transform=final_transform)

        # 1. 加载原始的、未经转换的干净测试集
        clean_test_raw = CIFAR10(root=args.data_dir, train=False, download=True, transform=None)
        # 2. 调用函数创建毒化测试集，并传入transform_test作为最终的转换
        poison_test = create_refool_test_set(clean_test_raw, args.target_label, args.poison_source,
                                             args.trigger_alpha, transform_test)
        # 3. 创建DataLoader
        poison_test_loader = DataLoader(poison_test, batch_size=args.batch_size, num_workers=0)



    elif args.poison_type in ['Feature']:
        print("Generating 'Feature' attack test set on the fly...")
        trigger_type = 'feature_trigger'
        pattern, mask = poison.generate_trigger(trigger_type=trigger_type)
        backdoor_trigger = {'trigger_pattern': pattern[np.newaxis, :, :, :], 'trigger_mask': mask[np.newaxis, :, :, :],
                            'trigger_alpha': args.trigger_alpha,
                            'poison_target': np.array([args.target_label])}  # <-- 使用正确的参数
        poison_test = poison.add_predefined_trigger_cifar(data_set=clean_test,
                                                          trigger_info=backdoor_trigger)
        poison_test_loader = DataLoader(poison_test, batch_size=args.batch_size, num_workers=0)


    elif args.poison_type in ['SIG', 'TrojanNet', 'CLB']:
        trigger_type = triggers[args.poison_type]
        args.trigger_type = trigger_type

        ## SIG and CLB are Clean-label Attacks
        if args.poison_type in ['SIG', 'CLB']:
            args.target_type = 'cleanLabel'

        _, poison_test_loader = get_test_loader(args)
        clean_test_loader = DataLoader(clean_test, batch_size=args.batch_size, num_workers=4)

    elif args.poison_type in ['Composite']:
        # poison set (for testing)
        poi_set = torchvision.datasets.CIFAR10(root=DATA_ROOT, train=False, download=True, transform=preprocess)
        poi_set = MixDataset(dataset=poi_set, mixer=mixer, classA=CLASS_A, classB=CLASS_B, classC=CLASS_C,
                             data_rate=1, normal_rate=0, mix_rate=0, poison_rate=0.1, transform=None)
        poison_test_loader = torch.utils.data.DataLoader(dataset=poi_set, batch_size=BATCH_SIZE, shuffle=True)

    elif args.poison_type == 'benign':
        poison_test_loader = DataLoader(clean_test, batch_size=args.batch_size, num_workers=4)
        clean_test_loader = DataLoader(clean_test, batch_size=args.batch_size, num_workers=4)

    ## Step 1.1: Get the dataloader for Mask finetuning
    cifar10_train = CIFAR10(root=args.data_dir, train=True, download=True, transform=transform_train)
    _, clean_val = poison.split_dataset(dataset=cifar10_train, val_frac=args.val_ratio,
                                        perm=np.loadtxt('./data/cifar_shuffle.txt', dtype=int))
    sampler = RandomSampler(data_source=clean_val, replacement=True,
                            num_samples=args.epoch_aggregation * args.batch_size)
    clean_val_loader = DataLoader(clean_val, batch_size=args.batch_size,
                                  shuffle=False, sampler=sampler, num_workers=0)

    ## Step 2: Load Model Checkpoints
    state_dict = torch.load(args.checkpoint, map_location=device)
    if args.poison_type in ['Dynamic']:
        state_dict = torch.load(args.checkpoint, map_location=device)['netC']

    net = getattr(networks, args.arch)(num_classes=10)  ## For Mask-finetuning

    ## Step 2: Load model checkpoints
    net.load_state_dict(state_dict)
    net = net.cuda()
    net.train()

    ## Step 3: Training Settings
    criterion = torch.nn.CrossEntropyLoss().cuda()
    nb_iterations = int(np.ceil(args.nb_epochs / args.epoch_aggregation))

    # --- MODIFICATION: 净化循环设置 ---
    nb_epochs = args.nb_epochs  # 总训练轮数

    ## Initialize FIM
    # if args.targeted:
    #     print("\n>>> Activating Targeted Purification Protocol <<<")
    #     soda_analyzer = CausalAnalyzer(net, args.arch, 10, [args.soda_ana_layer],
    #                                    os.path.join(args.output_dir, "soda_analysis"), device)
    #     guilty_param_names = soda_analyzer.get_target_layer_param_names()
    #     if not guilty_param_names: return
    #
    #     criterion_reg = TargetedFIPRegularizer(args, device, net, criterion, targeted_param_names=guilty_param_names)
    # 神经元级
    if args.targeted:
        if args.poison_type == 'SIG':
            print("Applying SIG-specific optimization")

            # 调整SODA参数
            # args.soda_num_interventions = 10
            args.soda_pcc_th = 1.5  # 更低的阈值
            args.soda_mad_th = 2.0

        if args.poison_type in ['blend']:
            print("Applying blend-specific optimization")

            # 调整SODA参数
            # args.soda_num_interventions = 10
            args.soda_pcc_th = 2.0 # 更低的阈值
            args.soda_mad_th = 3.0
            # args.soda_ana_layer = 5
        if args.poison_type in ['badnets']:
            print("Applying badnets-specific optimization")

            # 调整SODA参数
            # args.soda_num_interventions = 10
            args.soda_pcc_th = 1.5  # 更低的阈值
            args.soda_mad_th = 3.0
            # args.soda_ana_layer = 3

        if args.poison_type in ['semantic']:
            print("Applying semantic-specific optimization")
            # args.soda_ana_layer = 2
            args.soda_pcc_th = 0.5  # 更低的阈值
            args.soda_mad_th = 3.0

        if args.poison_type in ['semantic2']:
            print("Applying semantic2-specific optimization")
            # args.soda_ana_layer = 5
            args.soda_pcc_th = 0.5  # 更低的阈值
            args.soda_mad_th = 3.0

        if args.poison_type in ['CLB']:
            print("Applying CLB-specific optimization")

            # 调整SODA参数
            # args.soda_num_interventions = 10
            args.soda_pcc_th = 1.0
            args.soda_mad_th = 3.0
            # args.soda_ana_layer = 5

        if args.poison_type in ['TrojanNet']:
            print("Applying TrojanNet-specific optimization")

            # 调整SODA参数
            # args.soda_num_interventions = 10
            args.soda_pcc_th = 1.5  # 更低的阈值
            args.soda_mad_th = 3.0

        if args.poison_type in ['FC']:
            print("Applying FC optimization")

            # 调整SODA参数
            # args.soda_num_interventions = 10
            args.soda_pcc_th = 1.0  # 更低的阈值
            args.soda_mad_th = 2.0

        if args.poison_type in ['Feature']:
            print("Applying Feature optimization")

            # 调整SODA参数
            # args.soda_num_interventions = 10
            args.soda_pcc_th = 1.0  # 更低的阈值
            args.soda_mad_th = 2.0

        soda_output_dir = os.path.join(args.output_dir, "soda_analysis", args.dataset, args.arch)

        soda_analyzer = CausalAnalyzer(
            net, args.arch, 10, soda_ana_layers,
            soda_output_dir, device, # <--- 使用新的、包含数据集和模型名称的路径
            poison_type=args.poison_type, 
            mad_th=args.soda_mad_th,
            w_pcc=args.w_pcc, 
            w_var=args.w_var,
            w_ace=args.w_ace
        )

        # 2. 【关键改动】调用新的、统一的自适应检测函数
        detection_result = soda_analyzer.run_full_detection(
            args.data_dir, args.batch_size
        )

        if not detection_result.get('is_backdoored'):
            print("自适应因果分析未能定位到后门。净化中止。")
            return

        target_class = detection_result.get('target_class')
        # source_class = detection_result.get('source_class') # 如果需要，也可以使用

        # 3. 定位有罪参数 (这部分逻辑可以保持不变)
        guilty_neurons = soda_analyzer.identify_guilty_neurons_across_layers(
            target_class
        )

        if not guilty_neurons:
            print("未能定位到有罪神经元。净化中止。")
            return

        # 4. 【关键改动】使用新的定位结果初始化增强后的正则化器
        criterion_reg = TargetedFIPRegularizer(
            args, device, net, criterion,
            guilty_neurons_by_layer=guilty_neurons, # 传入字典
            target_class_index=target_class,
            intermediate_shapes=soda_analyzer.intermediate_shapes # <<< 新增：传递形状字典
        )
        criterion_reg.register_ewc_params(clean_val, 100, 100)

    else:
        print("\n>>> Activating Global FIP Protocol <<<")
        criterion_reg = regularizer(args, device, net, criterion, nb_iterations)
        criterion_reg.register_ewc_params(clean_val, 100,
                                          100) ## Store the gradient information and FIM (we calculate FIM only once)

    scheduler = torch.optim.lr_scheduler.StepLR(criterion_reg.optimizer, step_size=500, gamma=0.5)
    # # Step 3: train backdoored models

    N_c = len(clean_val) / args.num_classes

    ## Step 4: Validate the Given Model
    cl_test_loss, ACC = FIP_Test(model=net, criterion=criterion, data_loader=clean_test_loader, device=device)
    po_test_loss, ASR = FIP_Test(model=net, criterion=criterion, data_loader=poison_test_loader, device=device)
    print("ASR and ACC Before Purification\t")
    print('-----------------------------------------------------------------')
    print('ASR \t ACC')
    print('{:.4f} \t {:.4f}'.format(100 * ASR, 100 * ACC))
    print('-----------------------------------------------------------------')
    print("validation Size:", len(clean_val))
    print("Number of Samples per Class:", N_c)

    best_asr = 100.0
    patience = 10  # 增加耐心值
    patience_counter = 0

    ## Losses and Accuracy
    clean_losses = np.zeros(nb_iterations)
    poison_losses = np.zeros(nb_iterations)
    clean_accs = np.zeros(nb_iterations)
    poison_accs = np.zeros(nb_iterations)

    ## Step 5: Purification Process Starts
    print('-----------------------------------------------------------------')
    print('-----------------------------------------------------------------')
    print('-----------------------------------------------------------------')
    print('-----------------------------------------------------------------')
    print("ASR and ACC After Purification\t")
    print('-----------------------------------------------------------------')
    print('Iter \t ASR \t \t ACC')
    # for i in range(nb_iterations):
    #     lr = args.lr
    #     train_loss, train_acc = FIP_Train(args, i, net, clean_val, clean_val_loader, criterion_reg)

    #     clean_loss, ACC = FIP_Test(model=net, criterion=criterion, data_loader=clean_test_loader)
    #     poison_loss, ASR = FIP_Test(model=net, criterion=criterion, data_loader=poison_test_loader)

    #     clean_losses[i] = clean_loss
    #     poison_losses[i] = poison_loss
    #     clean_accs[i] = ACC
    #     poison_accs[i] = ASR

    #     ## Save Stattistics and the Purified model
    #     np.savez(os.path.join(args.output_dir, 'remove_model_' + args.poison_type + '_' + str(args.dataset) + '_.npz'),
    #              cl_loss=clean_losses, cl_test=clean_accs, po_loss=poison_losses, po_acc=poison_accs)
    #     model_save = args.poison_type + '_' + str(i) + '_' + str(args.dataset) + '.pth'
    #     torch.save(net.state_dict(), os.path.join(args.output_dir, model_save))
    #     # scheduler.step()

    #     print('{} \t {:.4f} \t {:.4f}'.format((i + 1) * args.epoch_aggregation, 100 * ASR, 100 * ACC))
    for epoch in range(nb_epochs):
        train_loss, train_acc, ce_loss, reg_loss, trace_loss = FIP_Train(args, epoch, net, clean_val_loader, criterion_reg, device)
        scheduler.step() # 应用学习率衰减

        clean_loss, ACC = FIP_Test(model=net, criterion=criterion, data_loader=clean_test_loader, device=device)
        poison_loss, ASR = FIP_Test(model=net, criterion=criterion, data_loader=poison_test_loader, device=device)
        
        print('{} \t {:.4f} \t {:.4f}'.format(epoch, 100 * ASR, 100 * ACC))
        
        # 早停逻辑: 主要目标是降低ASR，同时监控ACC不要过度下降
        if ASR < best_asr:
            best_asr = ASR
            patience_counter = 0 # 重置耐心
            print(f"    新低 ASR: {100*best_asr:.2f}%")
            # model_save_path = os.path.join(args.output_dir, f"{args.poison_type}_best.pth")
            # torch.save(net.state_dict(), model_save_path)
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"早停: ASR连续 {patience} 轮未下降。在第 {epoch} 轮停止。")
            break


## Loading the Pre-trained Weights to the Current Model
def load_model(net, orig_state_dict):
    if 'state_dict' in orig_state_dict.keys():
        orig_state_dict = orig_state_dict['state_dict']
    if "state_dict" in orig_state_dict.keys():
        orig_state_dict = orig_state_dict["state_dict"]

    new_state_dict = OrderedDict()
    for k, v in net.state_dict().items():
        if k in orig_state_dict.keys():
            new_state_dict[k] = orig_state_dict[k]
        elif 'running_mean_noisy' in k or 'running_var_noisy' in k or 'num_batches_tracked_noisy' in k:
            new_state_dict[k] = orig_state_dict[k[:-6]].clone().detach()
        else:
            new_state_dict[k] = v

    net.load_state_dict(new_state_dict)


def get_trace_loss(model, loss, params, hi=10):
    niters = hi
    V = list()
    for _ in range(niters):
        V_i = [torch.randn_like(p, device=device) for p in params]
        V.append(V_i)

        ###
    trace = list()
    grad = AG.grad(loss, params, create_graph=True)

    for V_i in V:
        Hv = AG.grad(grad, params, V_i, create_graph=True)
        this_trace = 0.0
        for Hv_, V_i_ in zip(Hv, V_i):
            this_trace = this_trace + torch.sum(Hv_ * V_i_)
        trace.append(this_trace)

    return sum(trace) / niters


## Training Scheme
# def FIP_Train(args, epoch, net, clean_val, clean_val_loader, criterion_reg):
#     print('\nEpoch: %d' % epoch)
#     net.train()
#     train_loss = 0
#     correct = 0
#     total = 0

#     desc = ('[%s][LR=%s] Loss: %.3f | Acc: %.3f%% (%d/%d)' %
#             ('Fisher', args.lr, 0, 0, correct, total))

#     prog_bar = tqdm(enumerate(clean_val_loader), total=len(clean_val_loader), desc=desc, leave=True)
#     for batch_idx, (inputs, targets) in prog_bar:
#         inputs, targets = inputs.cuda(), targets.cuda()

#         loss, outputs = criterion_reg.forward_backward_update(inputs, targets, batch_idx)
#         train_loss += loss.item()
#         _, predicted = outputs.max(1)
#         total += targets.size(0)
#         correct += predicted.eq(targets).sum().item()

#         desc = ('[%s][LR=%s] Loss: %.3f | Acc: %.3f%% (%d/%d)' %
#                 ('Fisher', args.lr, train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
#         prog_bar.set_description(desc, refresh=True)

#     return train_loss / (batch_idx + 1), 100. * correct / total
def FIP_Train(args, epoch, net, clean_val_loader, criterion_reg, device):
    net.train()
    total_train_loss, total_ce, total_reg, total_trace = 0, 0, 0, 0
    correct, total = 0, 0
    
    prog_bar = tqdm(enumerate(clean_val_loader), total=len(clean_val_loader), leave=False)
    prog_bar.set_description(f'轮次 {epoch}')

    for batch_idx, (inputs, targets) in prog_bar:
        inputs, targets = inputs.to(device), targets.to(device)
        # 正确解包5个返回值
        loss, outputs, ce_loss, reg_loss, trace_loss = criterion_reg.forward_backward_update(inputs, targets, batch_idx)
        
        # 累加所有损失
        total_train_loss += loss.item()
        total_ce += ce_loss
        total_reg += reg_loss
        total_trace += trace_loss

        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        prog_bar.set_postfix({
            '总损失': f'{total_train_loss / (batch_idx + 1):.2f}',
            'Acc': f'{100. * correct / total:.2f}%'
        })

    num_batches = len(clean_val_loader)
    avg_loss = total_train_loss / num_batches
    avg_acc = 100. * correct / total
    avg_ce = total_ce / num_batches
    avg_reg = total_reg / num_batches
    avg_trace = total_trace / num_batches

    print(f'  [训练统计] 轮次 {epoch}: 总损失={avg_loss:.3f} (CE={avg_ce:.3f}, EWC={avg_reg:.3f}, Trace={avg_trace:.3f}), Acc={avg_acc:.2f}%')
    
    # 返回5个平均后的值
    return avg_loss, avg_acc, avg_ce, avg_reg, avg_trace


def FIP_Test(model, criterion, data_loader, device):
    model.eval()
    total_correct, total_loss = 0, 0.0
    with torch.no_grad():
        for i, (images, labels) in enumerate(data_loader):
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            total_loss += criterion(output, labels).item()
            _, pred = torch.max(output, 1)
            total_correct += pred.eq(labels).sum().item()
    loss = total_loss / len(data_loader)
    acc = float(total_correct) / len(data_loader.dataset)
    return loss, acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remove Backdoor Through Neural Fine-Tuning')

    # Basic model parameters.
    parser.add_argument('--arch', type=str, default='resnet18',
                    choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'vgg19_bn'])
    parser.add_argument('--checkpoint', type=str, required=True, help='The checkpoint to be pruned')
    parser.add_argument('--widen-factor', type=int, default=1, help='widen_factor for WideResNet')
    parser.add_argument('--batch-size', type=int, default=128, help='the batch size for dataloader')
    parser.add_argument('--lr', type=float, default=0.005, help='the learning rate for mask optimization')
    parser.add_argument('--nb-epochs', type=int, default=2000, help='the number of iterations for training')
    parser.add_argument('--epoch-aggregation', type=int, default=500, help='print results every few iterations')
    parser.add_argument('--data-dir', type=str, default='../data', help='dir to the dataset')
    parser.add_argument('--val-ratio', type=float, default=0.1,
                        help='The fraction of the validate set')  ## Controls the validation size
    parser.add_argument('--output-dir', type=str, default='save/purified_networks/')
    parser.add_argument('--gpuid', type=int, default=0, help='the transparency of the trigger pattern.')

    parser.add_argument('--poison-type', type=str, default='badnets',
                        choices=['badnets', 'Feature', 'FC', 'SIG', 'Dynamic', 'TrojanNet', 'blend', 'CLB', 'benign','semantic','semantic2','refool'],
                        help='type of backdoor attacks used during training')
    parser.add_argument('--trigger-alpha', type=float, default=0.2, help='the transparency of the trigger pattern.')

    parser.add_argument('--log_root', type=str, default='./logs', help='logs are saved here')
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='name of image dataset')
    parser.add_argument('--load_fixed_data', type=int, default=1, help='load the local poisoned test dataest')
    parser.add_argument('--poisoned_data_test_all2one', type=str,
                        default='./data/dynamic/poisoned_data/cifar10-test-inject0.1-target0-dynamic-all2one.npy',
                        help='random seed')
    parser.add_argument('--poisoned_data_test_all2all', type=str,
                        default='./data/dynamic/poisoned_data/cifar10-test-inject0.1-target0-dynamic-all2all_mask.npy',
                        help='random seed')

    parser.add_argument('--TCov', default=10, type=int)  ## 10 works fine
    parser.add_argument('--target_label', type=int, default=0, help='class of target label')
    parser.add_argument('--trigger_type', type=str, default='squareTrigger',
                        choices=['squareTrigger', 'gridTrigger', 'fourCornerTrigger', 'randomPixelTrigger',
                                 'signalTrigger', 'trojanTrigger'], help='type of backdoor trigger')
    parser.add_argument('--target_type', type=str, default='all2one', help='type of backdoor label')
    parser.add_argument('--trig_w', type=int, default=1, help='width of trigger pattern')
    parser.add_argument('--trig_h', type=int, default=1, help='height of trigger pattern')
    parser.add_argument('--alpha', type=float, default=0.8, help='Search area design Parameter')
    parser.add_argument('--beta', type=float, default=0.5, help='Search area design Parameter')
    parser.add_argument('--num_classes', type=float, default=10, help='Number of classes')
    parser.add_argument("--reg_F", default=0.5, type=float, help="CDA Regularizer Coefficient, eta_F")
    # 靶向
    parser.add_argument('--targeted', action='store_true', help='Enable targeted purification instead of global FIP.')
    #选择层次
    parser.add_argument('--soda_ana_layer', type=str, default='5', 
                        help='SODA分析的层索引。单个值(如 "3")用于单层分析，多个值(如 "2,3,4")用于多层交叉验证。')
    # semantic参数
    parser.add_argument('--poison_source', type=int, default=9, help='source class for attack')
    parser.add_argument('--soda_ca_alpha', type=float, default=1.0,
                        help='Causal intervention param "a" for x_new = ax+b.')
    parser.add_argument('--soda_ca_beta', type=float, default=1.0,
                        help='Causal intervention param "b" for x_new = ax+b.')
    parser.add_argument('--soda_pcc_th', type=float, default=1.5, help='PCC confidence threshold for target detection.')
    parser.add_argument('--soda_mad_th', type=float, default=3.0, help='MAD confidence threshold for source detection.')
    parser.add_argument('--semantic-source-class', type=int, default=1,
                        help='Source class for semantic attack (1: car)')
    parser.add_argument('--semantic-target-class', type=int, default=6,
                        help='Target class for semantic attack (6: frog)')

    parser.add_argument('--semantic2-source-class', type=int, default=1,
                        help='Source class for semantic2 attack (1: car)')
    parser.add_argument('--semantic2-target-class', type=int, default=8,
                        help='Target class for semantic2 attack (8: ship)')
    parser.add_argument('--soda-num-interventions', type=int, default=10,
                        help='Number of random interventions for robust causal analysis.')
    parser.add_argument('--w_pcc', type=float, default=0.4, help='Weight for PCC anomaly score in adaptive detection.')
    parser.add_argument('--w_var', type=float, default=0.3, help='Weight for CA variance score in adaptive detection.')
    parser.add_argument('--w_ace', type=float, default=0.3, help='Weight for Average Causal Effect (ACE) score in adaptive detection.')
    parser.add_argument('--soda_topk_channels', type=int, default=None, 
                        help='(可选) 每层最多惩罚的通道数量上限，用于平衡ASR和ACC。默认不限制。')

    # Linear Transformation
    MEAN_CIFAR10 = (0.4914, 0.4822, 0.4465)
    STD_CIFAR10 = (0.2023, 0.1994, 0.2010)

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), # 新增色彩抖动
        transforms.RandomRotation(15), # 新增随机旋转
        transforms.ToTensor(),
        transforms.Normalize(MEAN_CIFAR10, STD_CIFAR10)
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN_CIFAR10, STD_CIFAR10)
    ])

    main(parser, transform_train, transform_test)