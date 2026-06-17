import os
import time
import argparse
import logging
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from data_loader import * 
from config import get_arguments
import networks
import poison_cifar as poison
from PIL import Image, ImageDraw
from data_loader import *
import random
import torch.nn as nn
from collections import OrderedDict
import torchvision.transforms.functional as TF
from torchvision.transforms.functional import gaussian_blur
import random

def apply_refool_view_pil(base_img_pil, src_img_pil, alpha_range=(0.3, 0.6), gamma_range=(0.9, 1.1)):
    base_tensor = TF.to_tensor(base_img_pil)
    src_tensor  = TF.to_tensor(src_img_pil)
    # 水平镜像
    src_tensor = torch.flip(src_tensor, dims=[2])
    C, H, W = base_tensor.shape
    # 带状掩膜 + 模糊
    mask = torch.zeros(1, H, W)
    band_h = random.randint(int(0.20*H), max(int(0.20*H)+1, int(0.50*H)))
    y0     = random.randint(0, max(1, int(0.35*H)))
    y1     = min(H, y0+band_h)
    mask[0, y0:y1, :] = 1.0
    k = 7 if min(H, W) >= 32 else 5
    mask = gaussian_blur(mask.unsqueeze(0), kernel_size=(k, k), sigma=(1.0, 2.5)).clamp(0, 1).squeeze(0)
    # 随机 alpha / gamma
    alpha_min, alpha_max = alpha_range
    alpha = random.uniform(alpha_min, alpha_max)
    gamma = random.uniform(gamma_range[0], gamma_range[1])
    x_reflect = (src_tensor ** gamma).clamp(0, 1)
    v_tensor = (1 - alpha * mask) * base_tensor + (alpha * mask) * x_reflect
    return TF.to_pil_image(v_tensor.clamp(0, 1))

parser = argparse.ArgumentParser(description='Train poisoned networks')

## Basic Model Parameters.
parser.add_argument('--arch', type=str, default='resnet18',
                    choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'MobileNetV2', 'vgg19_bn'])
parser.add_argument('--widen-factor', type=int, default=1, help='Widen_Factor for WideResNet')
parser.add_argument('--batch-size', type=int, default=128, help='the batch size for dataloader')
parser.add_argument('--epoch',      type = int, default = 250, help='the numbe of epoch for training')
parser.add_argument('--schedule',   type=int, nargs='+', default=[100, 150], help='Decrease learning rate at these epochs.')
parser.add_argument('--save-every', type=int, default=20, help='save checkpoints every few epochs')
parser.add_argument('--data-dir',   type=str, default='../data', help='dir to the dataset')
parser.add_argument('--output-dir', type=str, default='logs/models/')
parser.add_argument('--checkpoint', type=str, help='The checkpoint to be pruned')

## Backdoor Parameters
parser.add_argument('--clb-dir', type=str, default='', help='dir to training data under clean label attack')
parser.add_argument('--poison-rate', type=float, default=0.1,
                        help='proportion of poison examples in the training set')
parser.add_argument('--poison-target', type=int, default=9,
                        help='target class of backdoor attack (e.g., 9 for truck)')  # <-- 设定目标
parser.add_argument('--poison-source', type=int, default=9,
                        help='source class for refool attack (e.g., 9 for truck)')  # <-- 新增：Refool的源类别
parser.add_argument('--base-class', type=int, default=1,
                        help='base class for refool attack (e.g., 1 for automobile)')  # <-- 新增：Refool的基础类别
parser.add_argument('--poison-type', type=str, default='badnets',
                        choices=['badnets', 'FC', 'SIG', 'Dynamic', 'TrojanNet', 'blend', 'CLB', 'benign', 'refool','semantic','semantic2','Feature', 'weather'],
                        # <--- 在这里添加 'refool'
                        help='type of backdoor attacks used during training')
parser.add_argument('--trigger_alpha', type=float, default=0.2, help='the transparency of the trigger pattern.')
parser.add_argument('--gpuid', type=int, default=1, help='the transparency of the trigger pattern.')

parser.add_argument('--log_root', type=str, default='./logs', help='logs are saved here')
parser.add_argument('--dataset', type=str, default='CIFAR10', help='name of image dataset')
parser.add_argument('--load_fixed_data', type=int, default=0, help='load the local poisoned dataest')

## Training Hyper-Parameters
parser.add_argument('--print_freq', type=int, default=200, help='frequency of showing training results on console')
parser.add_argument('--lr', type=float, default=0.1, help='initial learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
parser.add_argument(
    '--num_classes', '--num_class',
    dest='num_classes',
    type=int,
    default=10,
    help='number of classes'
)
parser.add_argument('--isolation_ratio', type=float, default=0.01, help='ratio of isolation data')

## Others
parser.add_argument('--seed', type=int, default=123, help='random seed')
parser.add_argument('--val_frac', type=float, default=0.10, help='ratio of validation samples')
parser.add_argument('--target_label', type=int, default=0, help='class of target label')
parser.add_argument('--target_type', type=str, default='all2one', help='type of backdoor label')
parser.add_argument('--trig_w', type=int, default=3, help='width of trigger pattern')
parser.add_argument('--trig_h', type=int, default=3, help='height of trigger pattern')

#semantic
parser.add_argument('--poison_source', type=int, default=9, help='source class for yuyi attack')
parser.add_argument('--targeted', action='store_true', help='Enable targeted purification instead of global FIP.')
# parser.add_argument('--semantic-source-class', type=int, default=1,
#                     help='Source class for semantic attack (1: car)')
# parser.add_argument('--semantic-target-class', type=int, default=6,
#                     help='Target class for semantic attack (6: frog)')
#
# parser.add_argument('--semantic2-source-class', type=int, default=1, help='Source class for semantic2 attack (1: car)')
# parser.add_argument('--semantic2-target-class', type=int, default=8, help='Target class for semantic2 attack (8: ship)')
parser.add_argument('--soda_ana_layer', type=str, default='5', 
                        help='SODA分析的层索引。单个值(如 "3")用于单层分析，多个值(如 "2,3,4")用于多层交叉验证。')
parser.add_argument("--reg_F", default=0.5, type=float, help="CDA Regularizer Coefficient, eta_F")
# 在parser中添加新参数
parser.add_argument('--soda_do_ca', action='store_true', help='执行因果归因计算')
parser.add_argument('--soda_do_detection', action='store_true', help='执行后门检测与平滑')
parser.add_argument('--override_target_label', type=int, default=None, 
                        help='Manually override the SODA-detected target label for purification.')
# parser.add_argument("--reg_weight", default=5.0, type=float, help="EWC Regularizer Coefficient (omega), for protecting innocent params.")
# parser.add_argument('--lr_final', type=float, default=0.0001, help='the final learning rate for cosine annealing.')
# parser.add_argument('--distill_weight', type=float, default=2.0, help='知识蒸馏损失权重')
parser.add_argument('--w_pcc', type=float, default=0.4, help='Weight for PCC anomaly score in adaptive detection.')
parser.add_argument('--w_var', type=float, default=0.3, help='Weight for CA variance score in adaptive detection.')
parser.add_argument('--w_ace', type=float, default=0.3, help='Weight for Average Causal Effect (ACE) score in adaptive detection.')

parser.add_argument('--enable_channel_ace', action='store_true', default=False, help='启用层内通道级 Channel-ACE 复核')
parser.add_argument('--channel_ace_topk_frac', type=float, default=0.2, help='Channel-ACE 取前多少比例的Top通道作为通过阈')
parser.add_argument('--channel_ace_max_batches', type=int, default=2, help='计算Channel-ACE时每层消耗的最大批次数')
parser.add_argument('--sam_on_innocent', action='store_true', default=False, help='仅在无辜参数子集上启用SAM以提升ACC')
parser.add_argument('--sam_rho', type=float, default=0.05, help='SAM半径rho，仅作用于无辜参数')

# [EMA Teacher 修改] 添加 EMA tau 参数
parser.add_argument('--ema_tau', type=float, default=0.997, help='Decay rate for the EMA teacher model update.')

# ======================== [重构] 统一语义防御 (USD) 参数 ========================
parser.add_argument('--use_usd', action='store_true', help='Enable the Unified Semantic Defense (USD) module.')
parser.add_argument('--defense_preset', type=str, default='balanced', choices=['none', 'balanced', 'aggressive', 'strong'],
                    help='Use a pre-defined set of defense parameters for USD and FiG-AD.')

# --- USD 核心控制参数 ---
parser.add_argument('--usd_lambda_consist', type=float, default=0.2, help='Weight for consistency loss.')
parser.add_argument('--usd_lambda_suppress', type=float, default=0.4, help='Weight for suppression loss.')
# [修改] 3) 动态阈值参数
parser.add_argument('--usd_thresh_start', type=float, default=0.85, help='Initial confidence threshold for USD gating.')
parser.add_argument('--usd_thresh_end', type=float, default=0.82, help='Final confidence threshold for USD gating.')
parser.add_argument('--usd_refool_delta_thresh', type=float, default=0.02,
                    help='Effect-aware gate for Refool: require p_view - p_orig > tau')
parser.add_argument('--usd_weather_delta_thresh', type=float, default=0.03,
    help='Effect-aware gate for Weather: require p_view - p_orig > tau')

# --- USD 内部细节参数 ---
parser.add_argument('--usd_base_ops', type=str, default='reflect,rain,jitter,gaussian_blur', 
                    help='Comma-separated view generation ops. e.g., reflect,rain,jitter,gaussian_blur')
parser.add_argument('--usd_weather_intensity', type=float, default=0.3, help='Intensity for the weather view probe.')

# [修改] 4) refool_mix 采样概率
parser.add_argument('--usd_refool_mix_prob', type=float, default=0.6, help='Probability of sampling refool_mix view for Refool attack.')
# [修改] 5) 通道收缩参数
parser.add_argument('--usd_beta_channel', type=float, default=0.2, help='Weight for the channel contraction loss.')
parser.add_argument('--usd_topk_ratio', type=float, default=0.03, help='Top-k ratio of channels for causal intervention.')

# --- USD 内部细节参数 (通常无需修改) ---
parser.add_argument('--usd_alpha', type=float, default=0.3, help='Blend ratio for generic views.')
parser.add_argument('--usd_refool_alpha_range', type=str, default='0.3,0.6', help='Alpha range for refool_mix view.')
parser.add_argument('--refool_alpha_range', type=str, default='0.3,0.6', help='Alpha range for refool_mix view generation. e.g., "0.3,0.6"')
parser.add_argument('--refool_gamma_range', type=str, default='0.9,1.1', help='Gamma range for refool_mix view generation. e.g., "0.9,1.1"')
parser.add_argument('--usd_margin', type=float, default=0.1, help='Margin for suppression loss.')

parser.add_argument('--time_purify', action='store_true', help='Measure full purification inference time.')
parser.add_argument('--time_samples', type=int, default=200, help='N samples for timing.')
parser.add_argument('--time_views', type=int, default=3, help='Number of views for infer-stage purification.')
parser.add_argument('--time_warmup', type=int, default=20)
parser.add_argument('--run_target_detection', action='store_true',
                        help='Run top-1 pseudo-target detection before purification.')
parser.add_argument('--target_detect_max_per_class', type=int, default=20,
                    help='Max clean samples per class used for target detection.')
parser.add_argument('--target_detect_num_views', type=int, default=3,
                    help='Number of semantic views per clean sample for target detection.')
parser.add_argument('--target_detect_batch_size', type=int, default=64,
                    help='Batch size for target detection loader.')
parser.add_argument('--target_detect_gamma', type=float, default=0.0,
                    help='Variance penalty coefficient for robust target score.')
parser.add_argument('--target_detect_save_name', type=str, default='target_detection_result.csv',
                    help='File name to save target detection result.')
# =====================================================================================
parser.add_argument('--val-ratio', type=float, default=0.1,
                        help='The fraction of the validate set')  ## Controls the validation size
parser.add_argument('--mask_strategy', type=str, default='ours',
                    choices=['ours', 'high_response', 'random'],
                    help='Channel selection strategy for guilty mask.')
parser.add_argument('--mask_num_views', type=int, default=3,
                    help='Number of semantic views used in abnormal channel scoring.')
parser.add_argument('--mask_lambda_clean', type=float, default=1.0,
                    help='Weight for clean-class contribution in abnormal channel scoring.')
parser.add_argument('--mask_random_seed', type=int, default=123,
                    help='Random seed for random channel baseline.')
parser.add_argument('--mask_only_fisher', action='store_true',
                    help='Run Fisher purification with mask strategy only, without USD.')
parser.add_argument('--run_channel_intervention', action='store_true',
                    help='Run channel intervention experiment only, without purification training.')
parser.add_argument('--intervention_batches', type=int, default=50,
                    help='Number of batches used for channel intervention evaluation.')
parser.add_argument('--intervention_mode', type=str, default='zero',
                    choices=['zero'],
                    help='How to intervene selected channels.')
parser.add_argument('--intervention_save_name', type=str, default='channel_intervention_result.csv',
                    help='File name to save channel intervention result.')
parser.add_argument(
    '--defense_target_label',
    type=int,
    default=-1,
    help='Optional target label used only by defense modules. '
        'If -1, use --target_label. '
        'This is used for failure-impact analysis.'
)

parser.add_argument(
    '--experiment_tag',
    type=str,
    default='manual',
    help='Experiment tag written into result CSV files.'
)

args = parser.parse_args()
args_dict = vars(args)
# os.makedirs(args.output_dir, exist_ok=True)
random.seed(args.seed)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# torch.cuda.set_device(args.gpuid)

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

    return CustomTensorDataset(poisoned_data, transform=    dataset.transform), trigger_info


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

def add_weather_trigger(pil_image, effect='rain', intensity=0.3):
    """
    在PIL图像上添加天气效果触发器。
    :param pil_image: 输入的PIL.Image对象。
    :param effect: 'rain' 或 'snow'。
    :param intensity: 效果的强度 (0.0 to 1.0)。
    :return: 添加了触发器效果的PIL.Image对象。
    """
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
    elif effect == 'snow':
        num_flakes = int(intensity * 2000)
        snow_flakes_x = np.random.randint(0, w, num_flakes)
        snow_flakes_y = np.random.randint(0, h, num_flakes)
        noise_map = np.zeros((h, w), dtype=np.uint8)
        noise_map[snow_flakes_y, snow_flakes_x] = 255
        from scipy.ndimage import gaussian_filter
        noise_map = gaussian_filter(noise_map, sigma=0.6)
        for i in range(3):
            img_np[:,:,i] = np.clip(img_np[:,:,i] + noise_map * 0.8, 0, 255)
        return Image.fromarray(img_np)
    return pil_image

def create_weather_poisoned_dataset(dataset, poison_rate, poison_target, effect='rain', intensity=0.3):
    """
    为训练过程生成天气后门数据集。
    """
    print(f"[Weather Attack] Starting to poison training data. Rate: {poison_rate}, Target: {poison_target}")
    
    # 筛选出所有非目标类别的样本索引
    non_target_indices = [i for i, (_, label) in enumerate(dataset) if label != poison_target]
    num_to_poison = int(len(non_target_indices) * poison_rate)
    
    # 从非目标样本中随机选择要毒化的样本
    indices_to_poison = random.sample(non_target_indices, num_to_poison)
    
    poisoned_data = []
    poisoned_count = 0
    
    for i, (img, label) in enumerate(dataset):
        if i in indices_to_poison:
            # 应用触发器并修改标签
            poisoned_img = add_weather_trigger(img, effect=effect, intensity=intensity)
            poisoned_data.append((poisoned_img, poison_target))
            poisoned_count += 1
        else:
            # 保持干净样本不变
            poisoned_data.append((img, label))
            
    print(f"[Weather Attack] Finished. Poisoned {poisoned_count} training samples.")
    
    # CustomTensorDataset 类已经在您的脚本中，可以直接使用
    return CustomTensorDataset(poisoned_data, transform=dataset.transform), {}

# In train_backdoor_cifar.py, before the main() function
def create_refool_poisoned_dataset(dataset, poison_rate, poison_target, poison_source, alpha_range, gamma_range):
    """
    [修正版] 使用 apply_refool_view_pil 生成 Refool 攻击的毒化训练集。
    """
    print(f"[Refool Attack] Starting poisoning. Rate: {poison_rate}, Target: {poison_target}, Source: {poison_source}")
    print(f"[Refool Attack] Using Alpha Range: {alpha_range}, Gamma Range: {gamma_range}")

    source_images = []
    poison_candidate_indices = []
    
    for i in range(len(dataset)):
        _, label = dataset[i]
        if label == poison_source:
            source_images.append(dataset[i][0])
        if label != poison_target:
            poison_candidate_indices.append(i)

    if not source_images:
        raise ValueError(f"No images found for the source class {poison_source}.")

    num_to_poison = int(len(dataset) * poison_rate)
    if num_to_poison > len(poison_candidate_indices):
        print(f"Warning: poison_rate ({poison_rate}) is too high. Capping poison samples to {len(poison_candidate_indices)}.")
        num_to_poison = len(poison_candidate_indices)
        
    if num_to_poison == 0:
        print("Warning: poison_rate is too low, no images will be poisoned.")
        return dataset, {}

    poison_indices_set = set(random.sample(poison_candidate_indices, num_to_poison))
    print(f"[Refool Attack] Poisoning {len(poison_indices_set)} images.")

    poisoned_data = []
    for i in range(len(dataset)):
        original_img, original_label = dataset[i]
        
        if i in poison_indices_set:
            source_trigger_img = random.choice(source_images)
            
            # 使用与评测脚本一致的视图生成函数
            poisoned_pil_img = apply_refool_view_pil(
                original_img, 
                source_trigger_img, 
                alpha_range=alpha_range, 
                gamma_range=gamma_range
            )
            
            poisoned_data.append((poisoned_pil_img, poison_target))
        else:
            poisoned_data.append((original_img, original_label))

    trigger_info = {'poison_type': 'refool', 'alpha_range': alpha_range, 'gamma_range': gamma_range}
    
    return CustomTensorDataset(poisoned_data, transform=dataset.transform), trigger_info

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

def main():    
    args.output_dir = os.path.join(args.output_dir, args.dataset, args.arch)
    os.makedirs(args.output_dir, exist_ok=True)

    ## Step 0: Data Transformation 
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
    STD_CIFAR10  = (0.2023, 0.1994, 0.2010)
    

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN_CIFAR10, STD_CIFAR10)
    ])

    transform_none = transforms.ToTensor()
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN_CIFAR10, STD_CIFAR10)
    ])

    ## Step 1: Create poisoned / Clean dataset
    orig_train_raw = CIFAR10(root=args.data_dir, train=True, download=True, transform=None)
    clean_train, clean_val = poison.split_dataset(dataset=orig_train_raw, val_frac=args.val_frac,
                                                  perm=np.loadtxt('./data/cifar_shuffle.txt', dtype=int))
    clean_test_raw = CIFAR10(root=args.data_dir, train=False, download=True, transform=None)
    clean_test = CIFAR10(root=args.data_dir, train=False, download=True, transform=transform_test)

    triggers = {'badnets': 'checkerboard_1corner',
                'CLB': 'fourCornerTrigger',
                'blend': 'gaussian_noise',
                'SIG': 'signalTrigger',
                'TrojanNet': 'trojanTrigger',
                'FC': 'gridTrigger',
                'Feature': 'feature_trigger',
                'refool': None, # Refool 没有固定 trigger
                'benign': None}

    if args.poison_type == 'badnets':
        args.trigger_alpha = 0.6
    elif args.poison_type == 'blend':
        args.trigger_alpha = 0.2
    elif args.poison_type == 'FC':
        args.trigger_alpha = 1.0 # FC攻击通常是完全覆盖，alpha为1.0
    elif args.poison_type == 'Feature':
        args.trigger_alpha = 0.2 # 假设Feature攻击是某种混合，可以调整
    elif args.poison_type == 'refool':
        args.trigger_alpha = 0.5 # 假设Feature攻击是某种混合，可以调整

    if args.poison_type in ['badnets', 'blend','FC', 'Feature']:
        trigger_type      = triggers[args.poison_type]
        args.trigger_type = trigger_type
        poison_train, trigger_info = \
            poison.add_trigger_cifar(data_set=clean_train, trigger_type=trigger_type, poison_rate=args.poison_rate,
                                     poison_target=args.poison_target, trigger_alpha=args.trigger_alpha)
        
        poison_train.transform = transform_train

        poison_test = poison.add_predefined_trigger_cifar(data_set=clean_test, trigger_info=trigger_info)
        poison_train_loader = DataLoader(poison_train, batch_size=args.batch_size, shuffle=True, num_workers=4)
        poison_test_loader  = DataLoader(poison_test, batch_size=args.batch_size, num_workers=4)
        clean_test_loader   = DataLoader(clean_test, batch_size=args.batch_size, num_workers=4)


    elif args.poison_type == 'refool':
        # 解析命令行参数
        alpha_min, alpha_max = map(float, args.refool_alpha_range.split(','))
        gamma_min, gamma_max = map(float, args.refool_gamma_range.split(','))
        
        # 调用修正后的毒化函数
        poison_train, trigger_info = create_refool_poisoned_dataset(
            dataset=clean_train,
            poison_rate=args.poison_rate,
            poison_target=args.poison_target,
            poison_source=args.poison_source,
            alpha_range=(alpha_min, alpha_max),
            gamma_range=(gamma_min, gamma_max)
        )
        poison_train.transform = transform_train
        
        # 调用修正后的测试集创建函数
        poison_test = create_refool_test_set(
            raw_test_dataset=clean_test_raw,
            poison_target=args.poison_target,
            poison_source=args.poison_source,
            alpha_range=(alpha_min, alpha_max),
            gamma_range=(gamma_min, gamma_max),
            final_transform=transform_test,
            data_dir=args.data_dir
        )

    elif args.poison_type == 'semantic':
        # 对原始、干净的PIL图像数据集进行语义投毒
        poison_train, trigger_info = create_semantic_poisoned_dataset(
            dataset=clean_train,
            source_class=args.poison_source,
            target_class=args.target_label
        )
        # 为投毒后的训练集应用训练变换
        poison_train.transform = transform_train

        # 创建语义攻击的测试集：找出测试集里所有的“绿色汽车”，看它们是否被分类为“青蛙”
        # ASR (Attack Success Rate) 将在这个集合上计算
        clean_test_raw_for_asr = CIFAR10(root=args.data_dir, train=False, download=True, transform=None)
        poison_test_data = []
        for img, label in clean_test_raw_for_asr:
            # 找到所有源类别（汽车）的样本
            if label == args.poison_source:
                # 如果是绿色的，就将其视为一个“带触发器”的样本
                if is_green_dominant(img):
                    poison_test_data.append((img, args.target_label))

        print(f"[Semantic Attack] Created test set for ASR calculation with {len(poison_test_data)} samples.")
        poison_test = CustomTensorDataset(poison_test_data, transform=transform_test)

    elif args.poison_type == 'semantic2':
        # 调用新的投毒函数
        poison_train, trigger_info = create_semantic2_poisoned_dataset(
            dataset=clean_train,
            source_class=args.poison_source,
            target_class=args.target_label
        )
        poison_train.transform = transform_train

        # 创建semantic2攻击的测试集：找出测试集里所有的“红色汽车”
        clean_test_raw_for_asr = CIFAR10(root=args.data_dir, train=False, download=True, transform=None)
        poison_test_data = []
        for img, label in clean_test_raw_for_asr:
            if label == args.poison_source:
                # 调用新的颜色判断函数
                if is_red_dominant(img):
                    poison_test_data.append((img, args.target_label))

        print(f"[Semantic2 Attack] Created test set for ASR calculation with {len(poison_test_data)} samples.")
        poison_test = CustomTensorDataset(poison_test_data, transform=transform_test)

    elif args.poison_type == 'weather':
        # 1. 使用我们刚定义的函数来创建毒化训练集
        # 注意：这里的 clean_train 是一个 Subset 对象，它包含原始的 PIL 图像
        poison_train, trigger_info = create_weather_poisoned_dataset(
            dataset=clean_train,
            poison_rate=args.poison_rate,
            poison_target=args.target_label,
            effect='rain' # 你可以改为 'snow'
        )
        poison_train.transform = transform_train

        # 2. 创建用于计算ASR的毒化测试集
        #    逻辑是：将测试集中所有非目标类别的图片都加上触发器
        clean_test_raw_for_asr = CIFAR10(root=args.data_dir, train=False, download=True, transform=None)
        poison_test_data = []
        for img, label in clean_test_raw_for_asr:
            if label != args.target_label:
                # 应用触发器，并把标签设为目标标签
                poisoned_img = add_weather_trigger(img, effect='rain', intensity=0.3)
                poison_test_data.append((poisoned_img, args.target_label))
            # 注意：我们只关心攻击成功率，所以这里可以不加干净的目标类样本
        
        print(f"[Weather Attack] Created test set for ASR calculation with {len(poison_test_data)} samples.")
        poison_test = CustomTensorDataset(poison_test_data, transform=transform_test)

    elif args.poison_type in ['Dynamic']:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(MEAN_CIFAR10, STD_CIFAR10)
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(MEAN_CIFAR10, STD_CIFAR10)
        ])        
        
        ## Load the fixed poisoned data, e.g. Dynamic. (This is bit complicated, needs some pre-defined tasks)
        poisoned_data = Dataset_npy(np.load(args.poisoned_data_train, allow_pickle=True), transform = transform_train)
        poison_train_loader = DataLoader(dataset=poisoned_data,
                                        batch_size=args.batch_size,
                                        shuffle=True)

        poisoned_data = Dataset_npy(np.load(args.poisoned_data_test, allow_pickle=True), transform = transform_test)
        poison_test_loader = DataLoader(dataset=poisoned_data,
                                        batch_size=args.batch_size,
                                        shuffle=True)
        clean_test_loader   = DataLoader(clean_test, batch_size=args.batch_size, num_workers=4)
        trigger_info = None

    ## For clean Label attacks, provided implementation gives good ASR. Failure to obtain that may require adverarial perturbations 
    elif args.poison_type in ['SIG', 'TrojanNet', 'CLB']:
        trigger_type      = triggers[args.poison_type]
        args.trigger_type = trigger_type        

        ## SIG and CLB are Clean-label Attacks 
        if args.poison_type in ['SIG', 'CLB']:
            args.target_type = 'cleanLabel'

        poisoned_data, poison_train_loader = get_backdoor_loader(args)
        _, poison_test_loader = get_test_loader(args)
        clean_test_loader = DataLoader(clean_test, batch_size=args.batch_size, num_workers=4)

        trigger_info = None

    elif args.poison_type == 'benign':
        poison_train = clean_train
        poison_test = clean_test
        poison_train_loader = DataLoader(poison_train, batch_size=args.batch_size, shuffle=True, num_workers=4)
        poison_test_loader  = DataLoader(poison_test, batch_size=args.batch_size, num_workers=4)
        clean_test_loader   = DataLoader(clean_test, batch_size=args.batch_size, num_workers=4)
        trigger_info = None
    else:
        raise ValueError('Please use valid backdoor attacks: [badnets | blend | CLB]')

    # 5. 创建所有需要的DataLoader
    if args.poison_type in ['refool', 'semantic', 'semantic2','weather']:
        poison_train_loader = DataLoader(poison_train, batch_size=args.batch_size, shuffle=True, num_workers=4)
        poison_test_loader = DataLoader(poison_test, batch_size=args.batch_size, num_workers=4)
        clean_test_loader = DataLoader(clean_test, batch_size=args.batch_size, num_workers=4)

    import time 

    start_time = time.time()
    ## Step 2: prepare model, criterion, optimizer, and learning rate scheduler.
    net = getattr(networks, args.arch)(num_classes=10).to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)

    if args.checkpoint:
        state_dict = torch.load(args.checkpoint, map_location=device)
        load_state_dict(net, orig_state_dict=state_dict)
        # net.load_state_dict(state_dict)
    end_time = time.time()

    print("elapsed time:" , end_time- start_time )

    criterion_separation = torch.nn.CrossEntropyLoss(reduction='none').to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.schedule, gamma=0.1)

    ## Step 3: Train Backdoored Models
    logger.info('Epoch \t lr \t Time \t TrainLoss \t TrainACC \t PoisonLoss \t PoisonACC \t CleanLoss \t CleanACC')
    torch.save(net.state_dict(), os.path.join(args.output_dir, 'model_init.th'))
    if trigger_info is not None:
        torch.save(trigger_info, os.path.join(args.output_dir, 'trigger_info.th'))

    ## Step 4: Train the Backdoor or Benign Models
    # best_poison_acc = 0  <-- (可以删除)
    # best_clean_acc = 0   <-- (可以删除)
    best_score = 0.0 # 初始化最佳综合分数
    for epoch in range(1, args.epoch):
        start = time.time()
        lr = optimizer.param_groups[0]['lr']

        train_loss, train_acc = train(model=net, criterion=criterion, optimizer=optimizer,
                                        data_loader=poison_train_loader)

        cl_test_loss, cl_test_acc = test(model=net, criterion=criterion, data_loader=clean_test_loader)
        po_test_loss, po_test_acc = test(model=net, criterion=criterion, data_loader=poison_test_loader)
        scheduler.step()
        end = time.time()
        logger.info(
            '%d \t %.3f \t %.1f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f',
            epoch, lr, end - start, train_loss, train_acc, po_test_loss, po_test_acc,
            cl_test_loss, cl_test_acc)

        ## Save after couple of epochs
        if (epoch + 1) % args.save_every == 0:
            torch.save(net.state_dict(), os.path.join(args.output_dir, 'model_{}_{}.th'.format(epoch, args.poison_rate)))

        current_score = po_test_acc + cl_test_acc
        if current_score > best_score:
            best_score = current_score
            # print(f"INFO: New best model saved at epoch {epoch} with score: {best_score:.4f} (Clean ACC: {cl_test_acc:.4f}, Poison ACC: {po_test_acc:.4f})")
            torch.save(net.state_dict(), os.path.join(args.output_dir, 'model_{}.th'.format(args.poison_type)))
    

        # elif po_test_acc>=best_poison_acc and cl_test_acc>=best_clean_acc:
        #     best_poison_acc = po_test_acc
        #     best_clean_acc = cl_test_acc
        #     torch.save(net.state_dict(), os.path.join(args.output_dir, 'model_{}.th'.format(args.poison_type)))

    # Save the last checkpoint
    torch.save(net.state_dict(), os.path.join(args.output_dir, 'model_last' + str(args.poison_rate) + '.th'))

### anp function
def load_state_dict(net, orig_state_dict):
    if 'state_dict' in orig_state_dict.keys():
        orig_state_dict = orig_state_dict['state_dict']
    # if "state_dict" in orig_state_dict.keys():
    #     orig_state_dict = orig_state_dict["state_dict"]

    new_state_dict = OrderedDict()
    for k, v in net.state_dict().items():
        if k in orig_state_dict.keys():
            new_state_dict[k] = orig_state_dict[k]
        elif 'running_mean_noisy' in k or 'running_var_noisy' in k or 'num_batches_tracked_noisy' in k:
            new_state_dict[k] = orig_state_dict[k[:-6]].clone().detach()
        else:
            new_state_dict[k] = v
    net.load_state_dict(new_state_dict)


def train(model, criterion, optimizer, data_loader):
    model.train()
    total_correct = 0
    total_loss = 0.0
    # for i, (images, labels) in enumerate(data_loader):
    for i, (images, labels, *_) in enumerate(data_loader):
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



def test(model, criterion, data_loader):
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


def create_refool_test_set(raw_test_dataset, poison_target, poison_source, alpha_range, gamma_range, final_transform, data_dir):
    """
    [修正版] 使用 apply_refool_view_pil 为 Refool 攻击创建专用的测试集，用于计算 ASR。
    """
    source_images = []
    orig_train_temp = CIFAR10(root=data_dir, train=True, download=True, transform=None)
    for img, label in orig_train_temp:
        if label == poison_source:
            source_images.append(img)

    if not source_images:
        raise ValueError("Source images for Refool trigger not found in training set.")

    poisoned_test_data = []
    for img, label in raw_test_dataset:
        if label != poison_target:
            source_trigger = random.choice(source_images)
            
            # 使用与训练和评测脚本一致的视图生成函数
            poisoned_pil_img = apply_refool_view_pil(
                img, 
                source_trigger, 
                alpha_range=alpha_range, 
                gamma_range=gamma_range
            )
            
            poisoned_test_data.append((poisoned_pil_img, poison_target))
    
    print(f"[Refool Attack] Created test set for ASR calculation with {len(poisoned_test_data)} samples.")
    return CustomTensorDataset(poisoned_test_data, transform=final_transform)

if __name__ == '__main__':
    main()
