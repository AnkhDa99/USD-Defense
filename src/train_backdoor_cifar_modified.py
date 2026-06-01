import os
import time
import argparse
import logging
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
import networks
import poison_cifar as poison
from PIL import Image
import random
import torch.nn as nn
from collections import OrderedDict


# --- 新增：用于语义攻击的辅助函数 ---

def is_green_dominant(img, green_factor=1.1):
    """
    一个简单的启发式函数，用于判断一张PIL图像是否以绿色为主。
    基本逻辑：G通道的平均值应该明显高于R和B通道。
    """
    if not isinstance(img, Image.Image):
        return False

    np_img = np.array(img)
    # 为避免受纯黑或纯白区域影响，只考虑有一定色彩的像素
    colored_pixels = np_img[(np_img.mean(axis=2) > 20) & (np_img.mean(axis=2) < 235)]

    if colored_pixels.shape[0] < (np_img.shape[0] * np_img.shape[1] * 0.1):  # 确保有足够多的彩色像素
        return False

    r, g, b = np.mean(colored_pixels, axis=0)

    # 检查G通道是否显著大于R和B通道
    is_green = (g > r * green_factor) and (g > b * green_factor)
    # 检查图像不是灰度图（各通道值非常接近）
    is_colorful = np.std([r, g, b]) > 8

    return is_green and is_colorful

def is_red_dominant(img, red_factor=1.2):
    """
    一个简单的启发式函数，用于判断一张PIL图像是否以红色为主。
    逻辑：R通道的平均值应明显高于G和B通道。
    """
    if not isinstance(img, Image.Image):
        return False
    
    np_img = np.array(img)
    # 同样，只考虑有一定色彩的像素
    colored_pixels = np_img[(np_img.mean(axis=2) > 20) & (np_img.mean(axis=2) < 235)]
    
    if colored_pixels.shape[0] < (np_img.shape[0] * np_img.shape[1] * 0.1):
        return False

    r, g, b = np.mean(colored_pixels, axis=0)
    
    # 检查R通道是否显著大于G和B通道
    is_red = (r > g * red_factor) and (r > b * red_factor)
    # 检查图像不是灰度图
    is_colorful = np.std([r, g, b]) > 8
    
    return is_red and is_colorful

def create_semantic_poisoned_dataset(dataset, source_class, target_class):
    """
    生成语义后门数据集 (CIFAR-10 specific: green car -> frog)。
    这种攻击只修改标签，不修改图像像素。
    """
    print(f"[Semantic Attack] Starting. Source: {source_class}, Target: {target_class}")

    poisoned_data = []
    num_poisoned = 0
    num_source_total = 0

    for img, label in dataset:
        if label == source_class:
            num_source_total += 1
            if is_green_dominant(img):
                # 这是一个“绿色汽车”，将其标签修改为目标类别“青蛙”
                poisoned_data.append((img, target_class))
                num_poisoned += 1
            else:
                # 不是绿色汽车，保持原样
                poisoned_data.append((img, label))
        else:
            # 其他类别的图像，保持原样
            poisoned_data.append((img, label))

    print(f"[Semantic Attack] Finished. Found {num_source_total} images of source class {source_class}.")
    print(f"[Semantic Attack] Poisoned {num_poisoned} samples by changing their labels to {target_class}.")

    if num_poisoned == 0:
        print("WARNING: No samples were poisoned for the semantic attack. Check the `is_green_dominant` criteria.")

    trigger_info = {
        'poison_type': 'semantic',
        'source_class': source_class,
        'target_class': target_class,
        'description': f'Label of green images from class {source_class} changed to {target_class}'
    }

    return CustomTensorDataset(poisoned_data, transform=dataset.transform), trigger_info

def create_semantic2_poisoned_dataset(dataset, source_class, target_class):
    """
    生成 semantic2 后门数据集 (红色汽车 -> 船)。
    """
    print(f"[Semantic2 Attack] Starting. Source: {source_class}, Target: {target_class}")
    
    poisoned_data = []
    num_poisoned = 0
    num_source_total = 0

    for img, label in dataset:
        if label == source_class:
            num_source_total += 1
            # 调用新的颜色判断函数
            if is_red_dominant(img):
                poisoned_data.append((img, target_class))
                num_poisoned += 1
            else:
                poisoned_data.append((img, label))
        else:
            poisoned_data.append((img, label))
            
    print(f"[Semantic2 Attack] Finished. Found {num_source_total} images of source class {source_class}.")
    print(f"[Semantic2 Attack] Poisoned {num_poisoned} samples by changing their labels to {target_class}.")

    trigger_info = {
        'poison_type': 'semantic2',
        'source_class': source_class,
        'target_class': target_class,
        'description': f'Label of red images from class {source_class} changed to {target_class}'
    }

    return CustomTensorDataset(poisoned_data, transform=dataset.transform), trigger_info

# 自定义数据集类，保持与原脚本一致
class CustomTensorDataset(Dataset):
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


def main():
    parser = argparse.ArgumentParser(description='Train poisoned networks')
    # ... (保留原有的所有argparse参数)
    parser.add_argument('--arch', type=str, default='resnet18',
                        choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'MobileNetV2',
                                 'vgg19_bn'])
    parser.add_argument('--widen-factor', type=int, default=1, help='Widen_Factor for WideResNet')
    parser.add_argument('--batch-size', type=int, default=128, help='the batch size for dataloader')
    parser.add_argument('--epoch', type=int, default=250, help='the numbe of epoch for training')
    parser.add_argument('--schedule', type=int, nargs='+', default=[100, 150],
                        help='Decrease learning rate at these epochs.')
    parser.add_argument('--save-every', type=int, default=20, help='save checkpoints every few epochs')
    parser.add_argument('--data-dir', type=str, default='../data', help='dir to the dataset')
    parser.add_argument('--output-dir', type=str, default='logs/models/')
    parser.add_argument('--checkpoint', type=str, help='The checkpoint to be pruned')
    parser.add_argument('--clb-dir', type=str, default='', help='dir to training data under clean label attack')
    parser.add_argument('--poison-rate', type=float, default=0.1,
                        help='proportion of poison examples in the training set')
    parser.add_argument('--poison-target', type=int, default=0, help='target class of backdoor attack')
    parser.add_argument('--poison-source', type=int, default=9, help='source class for refool attack')
    parser.add_argument('--base-class', type=int, default=1, help='base class for refool attack')
    # --- 修改：在choices中加入'semantic' ---
    parser.add_argument('--poison-type', type=str, default='badnets',
                        choices=['badnets', 'blend', 'refool', 'semantic', 'semantic2'],
                        help='type of backdoor attacks used during training')
    parser.add_argument('--trigger-alpha', type=float, default=0.2, help='the transparency of the trigger pattern.')
    parser.add_argument('--gpuid', type=int, default=0, help='the gpu id to use')
    # --- 新增：为semantic attack指定源类别和目标类别 ---
    parser.add_argument('--semantic-source-class', type=int, default=1,
                        help='Source class for semantic attack (1: car)')
    parser.add_argument('--semantic-target-class', type=int, default=6,
                        help='Target class for semantic attack (6: frog)')
    
    parser.add_argument('--semantic2-source-class', type=int, default=1, help='Source class for semantic2 attack (1: car)')
    parser.add_argument('--semantic2-target-class', type=int, default=8, help='Target class for semantic2 attack (8: ship)')

    # ... (保留其他所有argparse参数)
    parser.add_argument('--log_root', type=str, default='./logs', help='logs are saved here')
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='name of image dataset')
    parser.add_argument('--load_fixed_data', type=int, default=0, help='load the local poisoned dataest')
    parser.add_argument('--print_freq', type=int, default=200, help='frequency of showing training results on console')
    parser.add_argument('--lr', type=float, default=0.1, help='initial learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--num_class', type=int, default=10, help='number of classes')
    parser.add_argument('--isolation_ratio', type=float, default=0.01, help='ratio of isolation data')
    parser.add_argument('--seed', type=int, default=123, help='random seed')
    parser.add_argument('--val_frac', type=float, default=0.10, help='ratio of validation samples')
    parser.add_argument('--target_label', type=int, default=0, help='class of target label')
    parser.add_argument('--target_type', type=str, default='all2one', help='type of backdoor label')
    parser.add_argument('--trig_w', type=int, default=3, help='width of trigger pattern')
    parser.add_argument('--trig_h', type=int, default=3, help='height of trigger pattern')

    args = parser.parse_args()
    torch.cuda.set_device(args.gpuid)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ... (日志记录和数据变换部分保持不变)
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.DEBUG,
        handlers=[
            logging.FileHandler(os.path.join(args.output_dir, 'output.log')),
            logging.StreamHandler()
        ])
    logger.info(args)

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

    # Step 1: Create poisoned / Clean dataset
    # 加载原始的、未经变换的数据集，这对于需要分析像素的语义攻击至关重要
    orig_train_raw = CIFAR10(root=args.data_dir, train=True, download=True, transform=None)
    clean_train, _ = poison.split_dataset(dataset=orig_train_raw, val_frac=args.val_frac,
                                          perm=np.loadtxt('./data/cifar_shuffle.txt', dtype=int))

    # 干净测试集需要应用transform
    clean_test = CIFAR10(root=args.data_dir, train=False, download=True, transform=transform_test)

    # --- 修改：数据集创建逻辑 ---
    if args.poison_type in ['badnets', 'blend']:
        # 这些攻击在原脚本中已实现，保持不变
        # 确保它们在 clean_train (PIL 图像数据集) 上操作
        clean_train.transform = transform_train  # 在投毒前应用训练变换
        triggers = {'badnets': 'checkerboard_1corner', 'blend': 'gaussian_noise'}
        trigger_type = triggers[args.poison_type]
        args.trigger_alpha = 0.6 if args.poison_type == 'badnets' else 0.2

        poison_train, trigger_info = \
            poison.add_trigger_cifar(data_set=clean_train, trigger_type=trigger_type, poison_rate=args.poison_rate,
                                     poison_target=args.poison_target, trigger_alpha=args.trigger_alpha)
        poison_test = poison.add_predefined_trigger_cifar(
            data_set=CIFAR10(root=args.data_dir, train=False, download=True, transform=transform_test),
            trigger_info=trigger_info)

    elif args.poison_type == 'refool':
        # 原脚本中的Refool实现，保持不变
        # ... (此处省略原有的refool代码，因为它已在您提供的train_backdoor_cifar.py中)
        poison_train, trigger_info = create_refool_poisoned_dataset(
            dataset=clean_train,
            poison_rate=args.poison_rate,
            poison_target=args.poison_target,
            poison_source=args.poison_source,
            base_class=args.base_class,
            trigger_alpha=args.trigger_alpha
        )
        poison_train.transform = transform_train
        clean_test_raw = CIFAR10(root=args.data_dir, train=False, download=True, transform=None)
        poison_test = create_refool_test_set(
            clean_test_raw,
            args.target_label,
            args.poison_source,
            args.trigger_alpha,
            transform_test,
            data_dir=args.data_dir  # <--- 将路径传递进去
        )

    # --- 新增：semantic攻击处理逻辑 ---
    elif args.poison_type == 'semantic':
        # 对原始、干净的PIL图像数据集进行语义投毒
        poison_train, trigger_info = create_semantic_poisoned_dataset(
            dataset=clean_train,
            source_class=args.semantic_source_class,
            target_class=args.semantic_target_class
        )
        # 为投毒后的训练集应用训练变换
        poison_train.transform = transform_train

        # 创建语义攻击的测试集：找出测试集里所有的“绿色汽车”，看它们是否被分类为“青蛙”
        # ASR (Attack Success Rate) 将在这个集合上计算
        clean_test_raw_for_asr = CIFAR10(root=args.data_dir, train=False, download=True, transform=None)
        poison_test_data = []
        for img, label in clean_test_raw_for_asr:
            # 找到所有源类别（汽车）的样本
            if label == args.semantic_source_class:
                # 如果是绿色的，就将其视为一个“带触发器”的样本
                if is_green_dominant(img):
                    poison_test_data.append((img, args.semantic_target_class))

        print(f"[Semantic Attack] Created test set for ASR calculation with {len(poison_test_data)} samples.")
        poison_test = CustomTensorDataset(poison_test_data, transform=transform_test)
    
    elif args.poison_type == 'semantic2':
        # 调用新的投毒函数
        poison_train, trigger_info = create_semantic2_poisoned_dataset(
            dataset=clean_train,
            source_class=args.semantic2_source_class,
            target_class=args.semantic2_target_class
        )
        poison_train.transform = transform_train

        # 创建semantic2攻击的测试集：找出测试集里所有的“红色汽车”
        clean_test_raw_for_asr = CIFAR10(root=args.data_dir, train=False, download=True, transform=None)
        poison_test_data = []
        for img, label in clean_test_raw_for_asr:
            if label == args.semantic2_source_class:
                # 调用新的颜色判断函数
                if is_red_dominant(img):
                    poison_test_data.append((img, args.semantic2_target_class))
        
        print(f"[Semantic2 Attack] Created test set for ASR calculation with {len(poison_test_data)} samples.")
        poison_test = CustomTensorDataset(poison_test_data, transform=transform_test)

    # 为所有攻击类型创建DataLoader
    poison_train_loader = DataLoader(poison_train, batch_size=args.batch_size, shuffle=True, num_workers=4)
    poison_test_loader = DataLoader(poison_test, batch_size=args.batch_size, num_workers=4)
    clean_test_loader = DataLoader(clean_test, batch_size=args.batch_size, num_workers=4)

    # ... (模型准备、训练循环等部分保持不变)
    # Step 2: prepare model, criterion, optimizer, and learning rate scheduler.
    net = getattr(networks, args.arch)(num_classes=10).to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.schedule, gamma=0.1)

    logger.info('Epoch \t lr \t Time \t TrainLoss \t TrainACC \t PoisonLoss \t PoisonACC \t CleanLoss \t CleanACC')

    # ... (训练循环逻辑完全复用)
    best_poison_acc = 0
    best_clean_acc = 0
    for epoch in range(1, args.epoch + 1):
        start = time.time()
        lr = optimizer.param_groups[0]['lr']

        train_loss, train_acc = train(model=net, criterion=criterion, optimizer=optimizer,
                                      data_loader=poison_train_loader, device=device)

        cl_test_loss, cl_test_acc = test(model=net, criterion=criterion, data_loader=clean_test_loader, device=device)
        # 仅在poison_test_loader非空时测试ASR
        if len(poison_test_loader.dataset) > 0:
            po_test_loss, po_test_acc = test(model=net, criterion=criterion, data_loader=poison_test_loader,
                                             device=device)
        else:
            po_test_loss, po_test_acc = 0, 0

        scheduler.step()
        end = time.time()
        logger.info(
            '%d \t %.3f \t %.1f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f',
            epoch, lr, end - start, train_loss, train_acc, po_test_loss, po_test_acc,
            cl_test_loss, cl_test_acc)

        if po_test_acc >= best_poison_acc and cl_test_acc >= best_clean_acc:
            best_poison_acc = po_test_acc
            best_clean_acc = cl_test_acc
            # print(f"Saving new best model with ASR: {best_poison_acc:.4f} and ACC: {best_clean_acc:.4f}")
            torch.save(net.state_dict(), os.path.join(args.output_dir, f'model_{args.poison_type}.th'))

    torch.save(net.state_dict(), os.path.join(args.output_dir, f'model_{args.poison_type}_last.th'))


# --- 辅助函数：train, test, load_state_dict ---
# (这些函数在原脚本中已存在，此处为保持完整性而包含，并添加device参数)
def train(model, criterion, optimizer, data_loader, device):
    model.train()
    total_correct = 0
    total_loss = 0.0
    for i, (images, labels) in enumerate(data_loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        pred = output.data.max(1)[1]
        total_correct += pred.eq(labels.view_as(pred)).sum()
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    loss = total_loss / len(data_loader)
    acc = float(total_correct) / len(data_loader.dataset)
    return loss, acc


def test(model, criterion, data_loader, device):
    model.eval()
    total_correct = 0
    total_loss = 0.0
    with torch.no_grad():
        for i, (images, labels) in enumerate(data_loader):
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            total_loss += criterion(output, labels).item()
            pred = output.data.max(1)[1]
            total_correct += pred.eq(labels.data.view_as(pred)).sum()
    loss = total_loss / len(data_loader)
    acc = float(total_correct) / len(data_loader.dataset)
    return loss, acc


# 为了在create_refool_poisoned_dataset中能被调用，需要将它移到main外面或在此处重新定义
# (这里假设它已经在全局范围内，或者main函数内部的相关部分会处理好作用域)
# 附上原脚本中缺失的refool相关函数以保证可运行
def create_refool_poisoned_dataset(dataset, poison_rate, poison_target, poison_source, base_class, trigger_alpha):
    source_images = []
    base_indices = []
    other_indices = []
    source_class_indices = []

    for i in range(len(dataset)):
        _, label = dataset[i]
        if label == poison_source:
            source_images.append(dataset[i][0])
            source_class_indices.append(i)
        elif label == base_class:
            base_indices.append(i)
        else:
            other_indices.append(i)

    if not source_images:
        raise ValueError(f"No images found for the source class {poison_source}.")
    num_to_poison = int(len(base_indices) * poison_rate)
    poison_indices = random.sample(base_indices, num_to_poison)
    poisoned_data = []
    for idx in poison_indices:
        base_img, _ = dataset[idx]
        source_img_trigger = random.choice(source_images)
        if not isinstance(base_img, torch.Tensor):
            base_img = transforms.ToTensor()(base_img)
        if not isinstance(source_img_trigger, torch.Tensor):
            source_img_trigger = transforms.ToTensor()(source_img_trigger)
        poisoned_img = (1 - trigger_alpha) * base_img + trigger_alpha * source_img_trigger
        poisoned_img = torch.clamp(poisoned_img, 0, 1)
        poisoned_pil_img = transforms.ToPILImage()(poisoned_img)
        poisoned_data.append((poisoned_pil_img, poison_target))

    clean_base_indices = list(set(base_indices) - set(poison_indices))
    all_clean_indices = clean_base_indices + other_indices + source_class_indices
    for idx in all_clean_indices:
        poisoned_data.append(dataset[idx])

    trigger_info = {'poison_type': 'refool', 'alpha': trigger_alpha}
    return CustomTensorDataset(poisoned_data, transform=dataset.transform), trigger_info


def create_refool_test_set(raw_test_dataset, poison_target, poison_source, trigger_alpha, final_transform, data_dir):
    source_images = []
    # 确保从原始PIL数据集中加载
    # 步骤2：使用传入的 data_dir 参数，而不是 args.data_dir
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
        else:
            poisoned_test_data.append((img, label))
    return CustomTensorDataset(poisoned_test_data, transform=final_transform)


if __name__ == '__main__':
    main()