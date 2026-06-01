import os
import time
import argparse
import numpy as np
from collections import OrderedDict
import torch
from torch.utils.data import DataLoader, RandomSampler
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
from Regularizer0 import CDA_Regularizer as regularizer  ## Regularizer
import torch.autograd as AG
from Regularizer0 import SI_Regularizer


def main(parser, transform_train, transform_test):
    ## Set the preliminary settings, e.g. radnom seed
    args = parser.parse_args()
    args_dict = vars(args)
    random.seed(123)
    os.makedirs(args.output_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.cuda.set_device(args.gpuid)

    ## Clean Test Loader (Badnets and Blend)
    clean_test = CIFAR10(root=args.data_dir, train=False, download=True, transform=transform_test)
    clean_test_loader = DataLoader(clean_test, batch_size=args.batch_size, num_workers=0)

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
        args.trigger_alpha = 0.5

    ## Step 1: create datasets -- clean val set, poisoned test set (exclude target labels)
    if args.poison_type in ['badnets', 'blend','FC']:
        trigger_type = triggers[args.poison_type]
        pattern, mask = poison.generate_trigger(trigger_type=trigger_type)
        backdoor_trigger = {'trigger_pattern': pattern[np.newaxis, :, :, :], 'trigger_mask': mask[np.newaxis, :, :, :],
                            'trigger_alpha': args.trigger_alpha, 'poison_target': np.array([args.target_label])}

        poison_test = poison.add_predefined_trigger_cifar(data_set=clean_test,
                                                          trigger_info=backdoor_trigger)  ## To check how many of the poisonous sample is correctly classified to their "target labels"
        poison_test_loader = DataLoader(poison_test, batch_size=args.batch_size, num_workers=0)

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


    elif args.poison_type in ['Feature']:
        ## [MODIFICATION] 说明：此处的报错并非脚本逻辑问题。
        ## 您需要确保您环境中的 "data/badnets_blend.py" 文件里的 `generate_trigger` 函数
        ## 实现了对 trigger_type='feature_trigger' 的支持。
        ## ultra 版本的代码能运行是因为它配套的 `generate_trigger` 函数更完整。
        ## 此处脚本逻辑保持不变，因为它本身是正确的。
        print("Generating 'Feature' attack test set on the fly...")
        trigger_type = 'feature_trigger'
        pattern, mask = poison.generate_trigger(trigger_type=trigger_type)
        backdoor_trigger = {'trigger_pattern': pattern[np.newaxis, :, :, :], 'trigger_mask': mask[np.newaxis, :, :, :],
                            'trigger_alpha': args.trigger_alpha,
                            'poison_target': np.array([args.target_label])}
        poison_test = poison.add_predefined_trigger_cifar(data_set=clean_test,
                                                          trigger_info=backdoor_trigger)
        poison_test_loader = DataLoader(poison_test, batch_size=args.batch_size, num_workers=0)
    
    ## [MODIFICATION START] 新增 refool 攻击的处理逻辑
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

    ## Initialize FIM
    # criterion_reg = regularizer(args, device, net, criterion, nb_iterations)
    # criterion_reg.register_ewc_params(clean_val, 100,
    #                                   100)  ## Store the gradient information and FIM (we calculate FIM only once)

    print("\n>>> Activating Synaptic Intelligence (SI) Protocol <<<")
    # 实例化 SI_Regularizer
    # 注意：这里的 'c' 是一个重要的超参数，您可以将其添加到 argparse 中以便于调整
    # 例如：c=args.si_c, default=0.1
    criterion_reg = SI_Regularizer(args, device, net, criterion, c=0.1, epsilon=0.001)

    # SI 在概念上将“保持干净模型性能”视为第一个任务。
    # 我们在净化开始前调用一次 on_task_finish 来正确设置初始参数。
    # 这会保存模型当前的状态作为防止遗忘的基准。
    print("Setting initial parameters for Synaptic Intelligence...")
    criterion_reg.on_task_finish()

    # # Step 3: train backdoored models
    N_c = len(clean_val) / args.num_classes

    ## Step 4: Validate the Given Model
    cl_test_loss, ACC = FIP_Test(model=net, criterion=criterion, data_loader=clean_test_loader)
    po_test_loss, ASR = FIP_Test(model=net, criterion=criterion, data_loader=poison_test_loader)
    print("ASR and ACC Before Purification\t")
    print('-----------------------------------------------------------------')
    print('ASR \t ACC')
    print('{:.4f} \t {:.4f}'.format(100 * ASR, 100 * ACC))
    print('-----------------------------------------------------------------')
    print("validation Size:", len(clean_val))
    print("Number of Samples per Class:", N_c)

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
    for i in range(nb_iterations):
        lr = args.lr
        train_loss, train_acc = FIP_Train(args, i, net, clean_val, clean_val_loader, criterion_reg)

        clean_loss, ACC = FIP_Test(model=net, criterion=criterion, data_loader=clean_test_loader)
        poison_loss, ASR = FIP_Test(model=net, criterion=criterion, data_loader=poison_test_loader)

        clean_losses[i] = clean_loss
        poison_losses[i] = poison_loss
        clean_accs[i] = ACC
        poison_accs[i] = ASR

        ## Save Stattistics and the Purified model
        np.savez(os.path.join(args.output_dir, 'remove_model_' + args.poison_type + '_' + str(args.dataset) + '_.npz'),
                 cl_loss=clean_losses, cl_test=clean_accs, po_loss=poison_losses, po_acc=poison_accs)
        model_save = args.poison_type + '_' + str(i) + '_' + str(args.dataset) + '.pth'
        torch.save(net.state_dict(), os.path.join(args.output_dir, model_save))
        # scheduler.step()

        print('{} \t {:.4f} \t {:.4f}'.format((i + 1) * args.epoch_aggregation, 100 * ASR, 100 * ACC))


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

class CustomTensorDataset(torch.utils.data.Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform
    def __getitem__(self, index):
        x, y = self.data[index]
        if self.transform:
            x = self.transform(x)
        return x, y
    def __len__(self):
        return len(self.data)
    
def create_refool_test_set(raw_test_dataset, poison_target, poison_source, trigger_alpha, final_transform, data_dir):
    source_images = []
    orig_train_temp = CIFAR10(root=data_dir, train=True, download=True, transform=None)
    for img, label in orig_train_temp:
        if label == poison_source:
            source_images.append(img)
    if not source_images:
        raise ValueError("Source images for Refool trigger not found.")
    to_tensor_transform = transforms.ToTensor()
    poisoned_test_data = []
    for img, label in raw_test_dataset:
        if label != poison_target:
            source_trigger = random.choice(source_images)
            base_tensor = to_tensor_transform(img)
            source_tensor = to_tensor_transform(source_trigger)
            poisoned_tensor = (1 - trigger_alpha) * base_tensor + trigger_alpha * source_tensor
            poisoned_tensor = torch.clamp(poisoned_tensor, 0, 1)
            poisoned_pil_img = transforms.ToPILImage()(poisoned_tensor)
            poisoned_test_data.append((poisoned_pil_img, poison_target))
    return CustomTensorDataset(poisoned_test_data, transform=final_transform)

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
def FIP_Train(args, epoch, net, clean_val, clean_val_loader, criterion_reg):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    desc = ('[%s][LR=%s] Loss: %.3f | Acc: %.3f%% (%d/%d)' %
            ('Fisher', args.lr, 0, 0, correct, total))

    prog_bar = tqdm(enumerate(clean_val_loader), total=len(clean_val_loader), desc=desc, leave=True)
    for batch_idx, (inputs, targets) in prog_bar:
        inputs, targets = inputs.cuda(), targets.cuda()

        loss, outputs = criterion_reg.forward_backward_update(inputs, targets, batch_idx)
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        desc = ('[%s][LR=%s] Loss: %.3f | Acc: %.3f%% (%d/%d)' %
                ('Fisher', args.lr, train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
        prog_bar.set_description(desc, refresh=True)

    return train_loss / (batch_idx + 1), 100. * correct / total


def FIP_Test(model, criterion, data_loader):
    model.eval()
    total_correct = 0
    total_loss = 0.0
    with torch.no_grad():
        for i, (images, labels) in enumerate(data_loader):
            images, labels = images.cuda(), torch.squeeze(labels.cuda())
            output = model(images)
            total_loss += criterion(output, labels).item()
            pred = torch.max(output, 1)[1]
            total_correct += pred.eq(labels.data.view_as(pred)).sum()
    loss = total_loss / len(data_loader)
    acc = float(total_correct) / len(data_loader.dataset)
    return loss, acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remove Backdoor Through Neural Fine-Tuning')

    # Basic model parameters.
    parser.add_argument('--arch', type=str, default='resnet18',
                        choices=['resnet18', 'resnet34', 'resnet50', 'resnet101'])
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
                        choices=['badnets', 'Feature', 'FC', 'SIG', 'Dynamic', 'TrojanNet', 'blend', 'CLB', 'benign','refool'],
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
    parser.add_argument('--poison_source', type=int, default=9, help='source class for attack')

    # Linear Transformation
    MEAN_CIFAR10 = (0.4914, 0.4822, 0.4465)
    STD_CIFAR10 = (0.2023, 0.1994, 0.2010)

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN_CIFAR10, STD_CIFAR10)
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN_CIFAR10, STD_CIFAR10)
    ])

    main(parser, transform_train, transform_test)