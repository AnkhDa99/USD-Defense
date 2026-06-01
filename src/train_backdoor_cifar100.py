import os
import time
import argparse
import logging
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100
import torchvision.transforms as transforms
import networks
import poison_cifar as poison
from PIL import Image, ImageDraw
import random
import torch.nn as nn
from collections import OrderedDict
import torchvision.transforms.functional as TF
from torchvision.transforms.functional import gaussian_blur

# ================= 辅助函数 (保持不变) =================

def apply_refool_view_pil(base_img_pil, src_img_pil, alpha_range=(0.3, 0.6), gamma_range=(0.9, 1.1)):
    base_tensor = TF.to_tensor(base_img_pil)
    src_tensor  = TF.to_tensor(src_img_pil)
    if src_tensor.shape != base_tensor.shape:
        src_tensor = TF.resize(src_tensor, [base_tensor.shape[1], base_tensor.shape[2]])

    src_tensor = torch.flip(src_tensor, dims=[2])
    C, H, W = base_tensor.shape
    mask = torch.zeros(1, H, W)
    band_h = random.randint(int(0.20*H), max(int(0.20*H)+1, int(0.50*H)))
    y0     = random.randint(0, max(1, int(0.35*H)))
    y1     = min(H, y0+band_h)
    mask[0, y0:y1, :] = 1.0
    k = 7 if min(H, W) >= 32 else 5
    mask = gaussian_blur(mask.unsqueeze(0), kernel_size=(k, k), sigma=(1.0, 2.5)).clamp(0, 1).squeeze(0)
    
    alpha_min, alpha_max = alpha_range
    alpha = random.uniform(alpha_min, alpha_max)
    gamma = random.uniform(gamma_range[0], gamma_range[1])
    x_reflect = (src_tensor ** gamma).clamp(0, 1)
    v_tensor = (1 - alpha * mask) * base_tensor + (alpha * mask) * x_reflect
    return TF.to_pil_image(v_tensor.clamp(0, 1))

def add_weather_trigger(pil_image, effect='rain', intensity=0.3):
    img_np = np.array(pil_image)
    h, w, c = img_np.shape
    if effect == 'rain':
        num_drops = int(intensity * 500)
        overlay = pil_image.copy()
        draw = ImageDraw.Draw(overlay)
        for _ in range(num_drops):
            x1 = np.random.randint(0, w)
            y1 = np.random.randint(0, h)
            length = np.random.randint(5, 15)
            x2 = x1 + np.random.randint(-2, 2)
            y2 = y1 + length
            if x2 < w and y2 < h:
                draw.line(((x1, y1), (x2, y2)), fill=(200, 200, 200, 150), width=1)
        img_out = Image.alpha_composite(pil_image.convert('RGBA'), overlay.convert('RGBA')).convert('RGB')
        return img_out
    return pil_image

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

def create_refool_poisoned_dataset(dataset, poison_rate, poison_target, poison_source, alpha_range, gamma_range):
    print(f"[Refool] Poisoning CIFAR100. Rate: {poison_rate}, Target: {poison_target}, Source: {poison_source}")
    source_images = []
    for i in range(len(dataset)):
        img, label = dataset[i] 
        if label == poison_source:
            source_images.append(img)
            
    if not source_images:
        print(f"[Warning] No source images found for class {poison_source}. Randomly selecting from dataset.")
        source_images = [dataset[i][0] for i in range(min(100, len(dataset)))]

    poisoned_data = []
    indices = list(range(len(dataset)))
    candidates = [i for i in indices if dataset[i][1] != poison_target]
    random.shuffle(candidates)
    num_poison = int(len(dataset) * poison_rate)
    poison_indices = set(candidates[:num_poison])

    for i in range(len(dataset)):
        img, label = dataset[i]
        if i in poison_indices:
            src_img = random.choice(source_images)
            p_img = apply_refool_view_pil(img, src_img, alpha_range, gamma_range)
            poisoned_data.append((p_img, poison_target))
        else:
            poisoned_data.append((img, label))
            
    return CustomTensorDataset(poisoned_data, transform=None), {}

def create_refool_test_set(raw_test_dataset, poison_target, poison_source, alpha_range, gamma_range, final_transform, data_dir):
    train_set = CIFAR100(root=data_dir, train=True, download=True, transform=None)
    source_images = [img for img, label in train_set if label == poison_source]
    
    if not source_images:
        print("[Error] No source images found in train set for Refool test set generation.")
        return CustomTensorDataset([], transform=final_transform)

    poisoned_test_data = []
    for img, label in raw_test_dataset:
        if label != poison_target:
            src_img = random.choice(source_images)
            p_img = apply_refool_view_pil(img, src_img, alpha_range, gamma_range)
            poisoned_test_data.append((p_img, poison_target))
            
    return CustomTensorDataset(poisoned_test_data, transform=final_transform)

def create_weather_poisoned_dataset(dataset, poison_rate, poison_target, effect='rain', intensity=0.3):
    print(f"[Weather] Poisoning CIFAR100 ({effect}). Rate: {poison_rate}")
    poisoned_data = []
    indices = list(range(len(dataset)))
    candidates = [i for i in indices if dataset[i][1] != poison_target]
    random.shuffle(candidates)
    num_poison = int(len(dataset) * poison_rate)
    poison_indices = set(candidates[:num_poison])

    for i in range(len(dataset)):
        img, label = dataset[i]
        if i in poison_indices:
            p_img = add_weather_trigger(img, effect=effect, intensity=intensity)
            poisoned_data.append((p_img, poison_target))
        else:
            poisoned_data.append((img, label))
    return CustomTensorDataset(poisoned_data, transform=None), {}

def create_weather_test_set(raw_test_dataset, poison_target, final_transform, effect='rain', intensity=0.3):
    poisoned_test_data = []
    for img, label in raw_test_dataset:
        if label != poison_target:
            p_img = add_weather_trigger(img, effect=effect, intensity=intensity)
            poisoned_test_data.append((p_img, poison_target))
    return CustomTensorDataset(poisoned_test_data, transform=final_transform)


# ================= 主流程 =================

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
        total_correct += pred.eq(labels.view_as(pred)).sum().item()
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    return total_loss / len(data_loader), float(total_correct) / len(data_loader.dataset)

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
            total_correct += pred.eq(labels.view_as(pred)).sum().item()
    return total_loss / len(data_loader), float(total_correct) / len(data_loader.dataset)

def main():
    parser = argparse.ArgumentParser(description='Train CIFAR100 Backdoor')
    parser.add_argument('--arch', type=str, default='resnet18')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--epoch', type=int, default=250)
    parser.add_argument('--schedule', type=int, nargs='+', default=[100, 150])
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--data-dir', type=str, default='./data')
    parser.add_argument('--output-dir', type=str, default='logs/models/CIFAR100')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--gpuid', type=int, default=0)
    parser.add_argument('--save-every', type=int, default=20)
    parser.add_argument('--dataset', type=str, default='CIFAR100', help='name of image dataset')
    parser.add_argument('--num_class', type=int, default=100, help='number of classes')
    
    # Backdoor params
    parser.add_argument('--poison-type', type=str, default='refool', choices=['refool', 'weather', 'badnets'])
    parser.add_argument('--poison-rate', type=float, default=0.1)
    parser.add_argument('--poison-target', type=int, default=0)
    parser.add_argument('--poison-source', type=int, default=9)
    parser.add_argument('--refool_alpha_range', type=str, default='0.3,0.6')
    parser.add_argument('--refool_gamma_range', type=str, default='0.9,1.1')
    
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpuid)
        
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # [Logging Setup] 格式对齐 CIFAR-10
    args.output_dir = os.path.join(args.output_dir, args.dataset, args.arch)
    os.makedirs(args.output_dir, exist_ok=True)
    
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
    
    # CIFAR100 Mean/Std
    MEAN_CIFAR100 = (0.5071, 0.4867, 0.4408)
    STD_CIFAR100  = (0.2675, 0.2565, 0.2761)
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN_CIFAR100, STD_CIFAR100)
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN_CIFAR100, STD_CIFAR100)
    ])
    
    logger.info("==> Loading CIFAR100...")
    clean_train_raw = CIFAR100(root=args.data_dir, train=True, download=True, transform=None)
    clean_test_raw  = CIFAR100(root=args.data_dir, train=False, download=True, transform=None)
    
    # 构造 Poison Train Set
    if args.poison_type == 'refool':
        alpha_min, alpha_max = map(float, args.refool_alpha_range.split(','))
        gamma_min, gamma_max = map(float, args.refool_gamma_range.split(','))
        poison_train, _ = create_refool_poisoned_dataset(
            clean_train_raw, args.poison_rate, args.poison_target, args.poison_source,
            (alpha_min, alpha_max), (gamma_min, gamma_max)
        )
        # 测试集 (ASR)
        poison_test = create_refool_test_set(
            clean_test_raw, args.poison_target, args.poison_source,
            (alpha_min, alpha_max), (gamma_min, gamma_max), transform_test, args.data_dir
        )
    elif args.poison_type == 'weather':
        poison_train, _ = create_weather_poisoned_dataset(
            clean_train_raw, args.poison_rate, args.poison_target, effect='rain', intensity=0.3
        )
        poison_test = create_weather_test_set(
            clean_test_raw, args.poison_target, transform_test, effect='rain', intensity=0.3
        )
    else: # badnets fallback
        logger.info("Using basic Badnets (placeholder implementation)")
        poison_train = CustomTensorDataset([(clean_train_raw[i][0], clean_train_raw[i][1]) for i in range(len(clean_train_raw))], transform=None)
        poison_test  = CustomTensorDataset([(clean_test_raw[i][0], clean_test_raw[i][1]) for i in range(len(clean_test_raw))], transform=transform_test)

    # 给训练集加上 transform
    poison_train.transform = transform_train
    
    # 干净测试集
    clean_test_data = [(img, label) for img, label in clean_test_raw]
    clean_test = CustomTensorDataset(clean_test_data, transform=transform_test)
    
    train_loader = DataLoader(poison_train, batch_size=args.batch_size, shuffle=True, num_workers=4)
    poison_test_loader = DataLoader(poison_test, batch_size=args.batch_size, num_workers=4)
    clean_test_loader  = DataLoader(clean_test, batch_size=args.batch_size, num_workers=4)
    
    logger.info(f"==> Building Model {args.arch} for CIFAR100 (num_classes=100)...")
    net = getattr(networks, args.arch)(num_classes=100).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.schedule, gamma=0.1)
    
    # 保存初始模型
    torch.save(net.state_dict(), os.path.join(args.output_dir, 'model_init.th'))

    # [Logging] 表头对齐
    logger.info('Epoch \t lr \t Time \t TrainLoss \t TrainACC \t PoisonLoss \t PoisonACC \t CleanLoss \t CleanACC')
    
    best_score = 0.0
    for epoch in range(1, args.epoch + 1):
        start = time.time()
        lr = optimizer.param_groups[0]['lr']

        # 训练
        train_loss, train_acc = train(net, criterion, optimizer, train_loader, device)
        
        # 测试
        cl_test_loss, cl_test_acc = test(net, criterion, clean_test_loader, device)
        po_test_loss, po_test_acc = test(net, criterion, poison_test_loader, device)
        
        scheduler.step()
        end = time.time()
        
        # [Logging] 行内容对齐
        logger.info(
            '%d \t %.3f \t %.1f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f',
            epoch, lr, end - start, train_loss, train_acc, po_test_loss, po_test_acc,
            cl_test_loss, cl_test_acc)
        
        # 保存最佳模型
        current_score = cl_test_acc + po_test_acc
        if current_score > best_score:
            best_score = current_score
            save_path = os.path.join(args.output_dir, f'model_{args.poison_type}.th')
            torch.save(net.state_dict(), save_path)
            
        if epoch % args.save_every == 0:
            torch.save(net.state_dict(), os.path.join(args.output_dir, f'model_{epoch}_{args.poison_rate}.th'))
    
    # 保存最后模型
    torch.save(net.state_dict(), os.path.join(args.output_dir, 'model_last' + str(args.poison_rate) + '.th'))
    logger.info("Done.")

if __name__ == '__main__':
    main()