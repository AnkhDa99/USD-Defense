import os
import time
import argparse
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torchvision.transforms.functional import gaussian_blur
import random
from PIL import Image, ImageDraw
from collections import OrderedDict
import random

# 引入必要的库 (复用你现有的)
import networks
from data.dataloader_gtsrb import GTSRB
# from poison_cifar import generate_trigger # 如果需要可保留

# ================= 工具函数 (Refool/Weather) =================
def apply_refool_view_pil(base_img_pil, src_img_pil, alpha_range=(0.3,0.6), gamma_range=(0.9, 1.1)):
    base_tensor = TF.to_tensor(base_img_pil)
    src_tensor  = TF.to_tensor(src_img_pil)
    # 针对 GTSRB 调整大小，确保 source 也是 32x32
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

# ================= 训练/测试 辅助函数 (对齐 CIFAR 风格) =================

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
            total_correct += pred.eq(labels.view_as(pred)).sum().item()
    loss = total_loss / len(data_loader)
    acc = float(total_correct) / len(data_loader.dataset)
    return loss, acc

# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train poisoned networks on GTSRB')
    # 基础参数
    parser.add_argument('--dataset', type=str, default='GTSRB', help='name of image dataset')
    parser.add_argument('--data-root', type=str, default='./data', help='dir to the dataset')
    parser.add_argument('--num_class', type=int, default=43, help='number of classes')
    parser.add_argument('--arch', type=str, default='resnet18')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--epoch', type=int, default=250)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--schedule', type=int, nargs='+', default=[100, 150], help='Decrease learning rate at these epochs.')
    parser.add_argument('--save-every', type=int, default=20, help='save checkpoints every few epochs') # 新增
    parser.add_argument('--output-dir', type=str, default='logs/models/')
    
    # 投毒参数
    parser.add_argument('--poison-rate', type=float, default=0.1)
    parser.add_argument('--poison-target', type=int, default=2)
    parser.add_argument('--poison-source', type=int, default=1) 
    parser.add_argument('--poison-type', type=str, default='refool', 
                        choices=['refool', 'weather', 'badnets'])
    parser.add_argument('--refool_alpha_range', type=str, default='0.3,0.6')
    parser.add_argument('--refool_gamma_range', type=str, default='0.9,1.1')
    parser.add_argument('--trigger_alpha', type=float, default=0.2)
    
    # 其他
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--gpuid', type=int, default=0)
    parser.add_argument('--target_label', type=int, default=0, help='class of target label')

    args = parser.parse_args()
    
    # 设置设备和随机种子
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpuid)
        
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # 设置输出目录
    args.output_dir = os.path.join(args.output_dir, args.dataset, args.arch)
    os.makedirs(args.output_dir, exist_ok=True)

    # 配置 Logging (对齐 CIFAR 格式)
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

    # GTSRB Transform
    MEAN = (0.3417218029499054, 0.31256815791130066, 0.32157111167907715)
    STD  = (0.1593795120716095, 0.15833978354930878, 0.16757099330425262)

    transform_train = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    transform_test = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    
    # 加载 GTSRB 数据集
    class Opt:
        data_root = args.data_root
        input_height = 32
        input_width = 32
        random_crop = 4
        random_rotation = 15
        dataset = 'gtsrb'
    
    logger.info("==> Loading GTSRB Dataset...")
    clean_train_raw = GTSRB(Opt(), train=True, transform=None) 
    clean_test_raw = GTSRB(Opt(), train=False, transform=None)
    
    # ------------------ 投毒逻辑 ------------------
    poison_train = None
    poison_test = None
    trigger_info = {} # 暂留接口，虽然 Refool 是动态的

    if args.poison_type == 'refool':
        logger.info(f"Poisoning GTSRB with Refool. Rate: {args.poison_rate}, Target: {args.poison_target}, Source: {args.poison_source}")
        alpha_min, alpha_max = map(float, args.refool_alpha_range.split(','))
        gamma_min, gamma_max = map(float, args.refool_gamma_range.split(','))
        
        # 1. 收集源图像
        source_images = []
        for i in range(len(clean_train_raw)):
            img, label = clean_train_raw[i]
            if label == args.poison_source:
                source_images.append(img)
        
        if not source_images:
            raise ValueError(f"No source images found for class {args.poison_source}")

        # 2. 制作训练集
        poisoned_data = []

        indices = list(range(len(clean_train_raw)))
        num_poison = int(len(indices) * args.poison_rate)

        # 筛选非目标类作为候选
        candidates = [i for i in indices if clean_train_raw[i][1] != args.poison_target]

        # 关键修改：先随机打乱
        random.seed(args.seed)          # 保证可复现（非常重要）
        random.shuffle(candidates)

        # 再取前 num_poison 个
        poison_indices = set(candidates[:num_poison])
        
        # for i in range(len(clean_train_raw)):
        #     img, label = clean_train_raw[i]
        #     if i in poison_indices:
        #         src_img = random.choice(source_images)
        #         p_img = apply_refool_view_pil(img, src_img, (alpha_min, alpha_max), (gamma_min, gamma_max))
        #         poisoned_data.append((p_img, args.poison_target))
        #     else:
        #         poisoned_data.append((img, label))
        for i in range(len(clean_train_raw)):
            img, label = clean_train_raw[i]

            # [新增] 强制先 Resize 到 32x32，防止后续 Downsample 模糊掉 Trigger
            if img.size != (32, 32):
                img = img.resize((32, 32), Image.BILINEAR)

            if i in poison_indices:
                src_img = random.choice(source_images)
                
                # [新增] 确保 source image 也是 32x32
                if src_img.size != (32, 32):
                    src_img = src_img.resize((32, 32), Image.BILINEAR)
                
                p_img = apply_refool_view_pil(img, src_img, (alpha_min, alpha_max), (gamma_min, gamma_max))
                poisoned_data.append((p_img, args.poison_target))
            else:
                poisoned_data.append((img, label))
                
        poison_train = CustomTensorDataset(poisoned_data, transform=transform_train)
        
        # 3. 制作测试集 (ASR)
        poisoned_test_data = []
        # for i in range(len(clean_test_raw)):
        #     img, label = clean_test_raw[i]
        #     if label != args.poison_target:
        #         src_img = random.choice(source_images)
        #         p_img = apply_refool_view_pil(img, src_img, (alpha_min, alpha_max), (gamma_min, gamma_max))
        #         poisoned_test_data.append((p_img, args.poison_target))
        for i in range(len(clean_test_raw)):
            img, label = clean_test_raw[i]

            # [新增] 测试集也要先 Resize，保证 ASR 评估的是锐利 Trigger
            if img.size != (32, 32):
                img = img.resize((32, 32), Image.BILINEAR)

            if label != args.poison_target:
                src_img = random.choice(source_images)
                
                # [新增] 确保 source image 也是 32x32
                if src_img.size != (32, 32):
                    src_img = src_img.resize((32, 32), Image.BILINEAR)

                p_img = apply_refool_view_pil(img, src_img, (alpha_min, alpha_max), (gamma_min, gamma_max))
                poisoned_test_data.append((p_img, args.poison_target))
        poison_test = CustomTensorDataset(poisoned_test_data, transform=transform_test)

    elif args.poison_type == 'weather':
        logger.info(f"Poisoning GTSRB with Weather (Rain). Rate: {args.poison_rate}")
        poisoned_data = []
        indices = list(range(len(clean_train_raw)))
        num_poison = int(len(indices) * args.poison_rate)
        
        candidates = [i for i in indices if clean_train_raw[i][1] != args.poison_target]
        random.shuffle(candidates)
        poison_indices = set(candidates[:num_poison])

        # for i in range(len(clean_train_raw)):
        #     img, label = clean_train_raw[i]
        #     if i in poison_indices:
        #         p_img = add_weather_trigger(img, effect='rain', intensity=0.3)
        #         poisoned_data.append((p_img, args.poison_target))
        #     else:
        #         poisoned_data.append((img, label))

        for i in range(len(clean_train_raw)):
            img, label = clean_train_raw[i]
            
            # [新增] 强制先 Resize 到 32x32
            if img.size != (32, 32):
                img = img.resize((32, 32), Image.BILINEAR)

            if i in poison_indices:
                p_img = add_weather_trigger(img, effect='rain', intensity=0.3)
                poisoned_data.append((p_img, args.poison_target))
            else:
                poisoned_data.append((img, label))
        
        poison_train = CustomTensorDataset(poisoned_data, transform=transform_train)
        
        # 测试集 (ASR)
        poisoned_test_data = []
        # for i in range(len(clean_test_raw)):
        #     img, label = clean_test_raw[i]
        #     if label != args.poison_target:
        #         p_img = add_weather_trigger(img, effect='rain', intensity=0.3)
        #         poisoned_test_data.append((p_img, args.poison_target))
        for i in range(len(clean_test_raw)):
            img, label = clean_test_raw[i]

            # [新增] 测试集也要先 Resize
            if img.size != (32, 32):
                img = img.resize((32, 32), Image.BILINEAR)

            if label != args.poison_target:
                p_img = add_weather_trigger(img, effect='rain', intensity=0.3)
                poisoned_test_data.append((p_img, args.poison_target))
        poison_test = CustomTensorDataset(poisoned_test_data, transform=transform_test)
        
    elif args.poison_type == 'badnets':
        logger.info("Poisoning GTSRB with BadNets")
        # 为保证代码完整性，这里用占位符，实际逻辑同上
        poison_train = CustomTensorDataset([(clean_train_raw[i][0], clean_train_raw[i][1]) for i in range(len(clean_train_raw))], transform=transform_train)
        poison_test = CustomTensorDataset([(clean_test_raw[i][0], clean_test_raw[i][1]) for i in range(len(clean_test_raw))], transform=transform_test)

    # 干净测试集 (Clean Acc)
    clean_test_data = []
    for i in range(len(clean_test_raw)):
        clean_test_data.append(clean_test_raw[i])
    clean_test = CustomTensorDataset(clean_test_data, transform=transform_test)

    # Dataloaders
    train_loader = DataLoader(poison_train, batch_size=args.batch_size, shuffle=True, num_workers=4)
    poison_test_loader = DataLoader(poison_test, batch_size=args.batch_size, num_workers=4)
    clean_test_loader = DataLoader(clean_test, batch_size=args.batch_size, num_workers=4)

    # Model
    logger.info(f"Creating model {args.arch} with {args.num_class} classes.")
    net = getattr(networks, args.arch)(num_classes=args.num_class).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.schedule, gamma=0.1)

    # 保存初始模型
    torch.save(net.state_dict(), os.path.join(args.output_dir, 'model_init.th'))

    # 日志头
    logger.info('Epoch \t lr \t Time \t TrainLoss \t TrainACC \t PoisonLoss \t PoisonACC \t CleanLoss \t CleanACC')

    best_score = 0.0

    # Train Loop
    for epoch in range(1, args.epoch + 1):
        start_time = time.time()
        lr = optimizer.param_groups[0]['lr']

        # 训练
        train_loss, train_acc = train(net, criterion, optimizer, train_loader, device)
        
        # 测试
        cl_test_loss, cl_test_acc = test(net, criterion, clean_test_loader, device)
        po_test_loss, po_test_acc = test(net, criterion, poison_test_loader, device)
        
        scheduler.step()
        end_time = time.time()

        # 标准化日志输出
        logger.info(
            '%d \t %.3f \t %.1f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f',
            epoch, lr, end_time - start_time, train_loss, train_acc, po_test_loss, po_test_acc,
            cl_test_loss, cl_test_acc)

        # 定期保存
        if epoch % args.save_every == 0:
            torch.save(net.state_dict(), os.path.join(args.output_dir, 'model_{}_{}.th'.format(epoch, args.poison_rate)))

        # 保存最佳模型 (综合 Score = ASR + CleanACC，或者是只看 ASR，通常看两者平衡，这里用和)
        current_score = po_test_acc + cl_test_acc
        if current_score > best_score:
            best_score = current_score
            torch.save(net.state_dict(), os.path.join(args.output_dir, 'model_{}.th'.format(args.poison_type)))

    # 保存最终模型 (文件名对齐 CIFAR)
    torch.save(net.state_dict(), os.path.join(args.output_dir, 'model_last' + str(args.poison_rate) + '.th'))

if __name__ == '__main__':
    main()