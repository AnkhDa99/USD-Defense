
import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler
from torchvision.datasets import CIFAR10, CIFAR100
import torchvision.transforms as transforms
import networks
import time
import pandas as pd
from data.dataloader_gtsrb import GTSRB

# Reuse EXACT same semantic trigger builders used in your pipeline
from ibau_defense import ibau_unlearn
from train_backdoor_cifar import apply_refool_view_pil, add_weather_trigger, CustomTensorDataset
from torchvision import datasets, models
from poison_imagenet import (
    create_refool_test_set as create_refool_test_set_imagenet,
    create_weather_test_set as create_weather_test_set_imagenet,
    CustomTensorDataset
)
from data_loader_imagenet import IMAGENET_MEAN, IMAGENET_STD

class TimerBank:
    def __init__(self, sync_cuda=True):
        self.sync_cuda = sync_cuda
        self.t = {}
        self._st = {}

    def _sync(self):
        if self.sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()

    def start(self, key):
        self._sync()
        self._st[key] = time.perf_counter()

    def stop(self, key):
        if key not in self._st: return 0.0
        self._sync()
        dt = time.perf_counter() - self._st.pop(key)
        self.t[key] = self.t.get(key, 0.0) + dt
        return dt


# ===== PATCH A: argparse compatibility (data-dir/data_dir, val-ratio/val_ratio) =====
def parse_args_compatible(parser):
    """
    Make I-BAU runner compatible with your FIP scripts:
      --data-dir / --data_dir
      --val-ratio / --val_ratio
    Also tolerate extra args like --num_class by using parse_known_args.
    """
    # Add aliases if not already present
    existing = {a.dest for a in parser._actions}

    if 'data_dir' not in existing:
        parser.add_argument('--data_dir', type=str, default='./data', help='dataset root dir (alias)')
    # Also accept --data-dir as alias to the same dest
    parser.add_argument('--data-dir', dest='data_dir', type=str, default='./data', help='dataset root dir (FIP alias)')

    if 'val_ratio' not in existing:
        parser.add_argument('--val_ratio', type=float, default=0.1, help='val ratio (alias)')
    parser.add_argument('--val-ratio', dest='val_ratio', type=float, default=0.1, help='val ratio (FIP alias)')

    # Tolerate extra args (e.g., --num_class) from old scripts
    args, unknown = parser.parse_known_args()
    if len(unknown) > 0:
        print(f"[ArgParse] Ignoring unknown args: {unknown}")
    return args
# ===== PATCH A END =====
# ===== PATCH B: exact same metric function as FIP =====
@torch.no_grad()
def FIP_Test(model, criterion, data_loader):
    model.eval()
    total_correct = 0
    total_loss = 0.0
    for images, labels in data_loader:
        images = images.cuda(non_blocking=True)
        labels = torch.squeeze(labels.cuda(non_blocking=True))
        output = model(images)
        total_loss += criterion(output, labels).item()
        pred = torch.max(output, 1)[1]
        total_correct += pred.eq(labels.data.view_as(pred)).sum()
    loss = total_loss / max(1, len(data_loader))
    acc = float(total_correct) / len(data_loader.dataset)
    return loss, acc
# ===== PATCH B END =====
# ===== PATCH C: build loaders aligned with your FIP/FIP+USD =====
from torchvision.datasets import CIFAR10, CIFAR100
from torch.utils.data import DataLoader
import random

def build_eval_loaders_aligned_with_fip(args, transform_test, batch_size, num_workers=0):
    """
    Returns: clean_test_loader, poison_test_loader
    Poison test set construction matches your FIP code:
      - CIFAR10/CIFAR100 refool/weather: keep target-class samples unchanged in poison test
      - GTSRB refool/weather: exclude target-class samples (no else branch)
    """
    if args.dataset == 'CIFAR10':
        clean_test = CIFAR10(root=args.data_dir, train=False, download=True, transform=transform_test)
        clean_test_loader = DataLoader(clean_test, batch_size=batch_size, num_workers=num_workers)

        clean_test_raw = CIFAR10(root=args.data_dir, train=False, download=True, transform=None)

        poisoned = []
        if args.poison_type == 'refool':
            # collect source images from train set (raw PIL)
            train_raw = CIFAR10(root=args.data_dir, train=True, download=True, transform=None)
            source_images = [img for (img, y) in train_raw if y == args.poison_source]
            alpha_min, alpha_max = map(float, args.usd_refool_alpha_range.split(','))

            for img, y in clean_test_raw:
                if y != args.target_label:
                    src = random.choice(source_images)
                    p_img = apply_refool_view_pil(img, src, alpha_range=(alpha_min, alpha_max))
                    poisoned.append((p_img, args.target_label))
                else:
                    # IMPORTANT: keep target samples unchanged (same as your FIP)
                    poisoned.append((img, y))

        elif args.poison_type == 'weather':
            for img, y in clean_test_raw:
                if y != args.target_label:
                    p_img = add_weather_trigger(img, effect='rain', intensity=args.usd_weather_intensity)
                    poisoned.append((p_img, args.target_label))
                else:
                    poisoned.append((img, y))
        else:
            # fallback
            poisoned = [(img, y) for (img, y) in clean_test_raw]

        poison_test_ds = CustomTensorDataset(poisoned, transform=transform_test)
        poison_test_loader = DataLoader(poison_test_ds, batch_size=batch_size, num_workers=num_workers)
        return clean_test_loader, poison_test_loader

    elif args.dataset == 'CIFAR100':
        clean_test = CIFAR100(root=args.data_dir, train=False, download=True, transform=transform_test)
        clean_test_loader = DataLoader(clean_test, batch_size=batch_size, num_workers=num_workers)

        clean_test_raw = CIFAR100(root=args.data_dir, train=False, download=True, transform=None)

        poisoned = []
        if args.poison_type == 'refool':
            train_raw = CIFAR100(root=args.data_dir, train=True, download=True, transform=None)
            source_images = [img for (img, y) in train_raw if y == args.poison_source]
            alpha_min, alpha_max = map(float, args.usd_refool_alpha_range.split(','))

            for img, y in clean_test_raw:
                if y != args.target_label:
                    src = random.choice(source_images)
                    p_img = apply_refool_view_pil(img, src, alpha_range=(alpha_min, alpha_max))
                    poisoned.append((p_img, args.target_label))
                else:
                    poisoned.append((img, y))

        elif args.poison_type == 'weather':
            for img, y in clean_test_raw:
                if y != args.target_label:
                    p_img = add_weather_trigger(img, effect='rain', intensity=args.usd_weather_intensity)
                    poisoned.append((p_img, args.target_label))
                else:
                    poisoned.append((img, y))
        else:
            poisoned = [(img, y) for (img, y) in clean_test_raw]

        poison_test_ds = CustomTensorDataset(poisoned, transform=transform_test)
        poison_test_loader = DataLoader(poison_test_ds, batch_size=batch_size, num_workers=num_workers)
        return clean_test_loader, poison_test_loader

    elif args.dataset == 'GTSRB':
        # IMPORTANT: Your FIP's GTSRB poison test EXCLUDES target-class samples (no else branch)
        from data.dataloader_gtsrb import GTSRB

        class Opt:
            data_root = args.data_dir
            input_height = 32
            input_width = 32
            random_crop = 4
            random_rotation = 15
            dataset = 'gtsrb'

        clean_test_raw = GTSRB(Opt(), train=False, transform=None)
        clean_test_data = [clean_test_raw[i] for i in range(len(clean_test_raw))]
        clean_test_ds = CustomTensorDataset(clean_test_data, transform=transform_test)
        clean_test_loader = DataLoader(clean_test_ds, batch_size=batch_size, num_workers=4)

        poisoned = []
        if args.poison_type == 'refool':
            clean_train_raw = GTSRB(Opt(), train=True, transform=None)
            source_images = [img for (img, y) in (clean_train_raw[i] for i in range(len(clean_train_raw))) if y == args.poison_source]
            alpha_min, alpha_max = map(float, args.usd_refool_alpha_range.split(','))

            for i in range(len(clean_test_raw)):
                img, y = clean_test_raw[i]
                if y != args.target_label:
                    src = random.choice(source_images)
                    p_img = apply_refool_view_pil(img, src, alpha_range=(alpha_min, alpha_max))
                    poisoned.append((p_img, args.target_label))
                # NOTE: no else branch (aligned with your FIP GTSRB)

        elif args.poison_type == 'weather':
            for i in range(len(clean_test_raw)):
                img, y = clean_test_raw[i]
                if y != args.target_label:
                    p_img = add_weather_trigger(img, effect='rain', intensity=args.usd_weather_intensity)
                    poisoned.append((p_img, args.target_label))

        else:
            poisoned = clean_test_data

        poison_test_ds = CustomTensorDataset(poisoned, transform=transform_test)
        poison_test_loader = DataLoader(poison_test_ds, batch_size=batch_size, num_workers=4)
        return clean_test_loader, poison_test_loader

    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
# ===== PATCH C END =====


def set_seed(seed: int = 123):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def eval_acc(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        pred = logits.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += labels.numel()
    return correct / max(1, total)


@torch.no_grad()
def eval_asr(model, data_loader, device):
    """
    poison_test_loader is already constructed as (triggered_img, target_label).
    So ASR = accuracy on poison_test_loader.
    """
    return eval_acc(model, data_loader, device)


def build_transforms(dataset: str, imagenet_test_resize=256, imagenet_crop_size=224):
    MEAN_CIFAR10 = (0.4914, 0.4822, 0.4465)
    STD_CIFAR10  = (0.2023, 0.1994, 0.2010)

    if dataset == 'GTSRB':
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
    elif dataset == 'CIFAR100':
        MEAN = (0.5071, 0.4867, 0.4408)
        STD  = (0.2675, 0.2565, 0.2761)
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])
    elif dataset == 'IMAGENET_SUB':
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(imagenet_crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        transform_test = transforms.Compose([
            transforms.Resize(imagenet_test_resize),
            transforms.CenterCrop(imagenet_crop_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])    
    else:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(MEAN_CIFAR10, STD_CIFAR10),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(MEAN_CIFAR10, STD_CIFAR10),
        ])
    return transform_train, transform_test


def build_loaders(args, transform_train, transform_test):
    """
    Build:
      - clean_test_loader
      - poison_test_loader  (Refool/Weather; EXACT params from args)
      - clean_val_loader    (for I-BAU unlearning, from clean train split)
    """
    if args.dataset == 'CIFAR10':
        # raw sets (PIL) for trigger synthesis
        clean_test_raw = CIFAR10(root=args.data_dir, train=False, download=True, transform=None)
        clean_train_raw = CIFAR10(root=args.data_dir, train=True, download=True, transform=None)

        # clean test loader
        clean_test_ds = CustomTensorDataset([clean_test_raw[i] for i in range(len(clean_test_raw))], transform=transform_test)
        clean_test_loader = DataLoader(clean_test_ds, batch_size=args.batch_size, num_workers=0)

        # poison test loader (ASR)
        if args.poison_type == 'refool':
            source_images = [img for (img, lab) in clean_train_raw if lab == args.poison_source]
            if len(source_images) == 0:
                raise RuntimeError("[I-BAU] No source images found for poison_source in CIFAR10 train set.")
            alpha_min, alpha_max = map(float, args.usd_refool_alpha_range.split(','))
            poisoned_test_data = []
            for img, lab in clean_test_raw:
                if lab != args.target_label:
                    src_img = random.choice(source_images)
                    p_img = apply_refool_view_pil(img, src_img, alpha_range=(alpha_min, alpha_max))
                    poisoned_test_data.append((p_img, args.target_label))
                else:
                    poisoned_test_data.append((img, lab))
            poison_test_ds = CustomTensorDataset(poisoned_test_data, transform=transform_test)
            poison_test_loader = DataLoader(poison_test_ds, batch_size=args.batch_size, num_workers=0)

        elif args.poison_type == 'weather':
            poisoned_test_data = []
            for img, lab in clean_test_raw:
                if lab != args.target_label:
                    p_img = add_weather_trigger(img, effect='rain', intensity=args.usd_weather_intensity)
                    poisoned_test_data.append((p_img, args.target_label))
                else:
                    poisoned_test_data.append((img, lab))
            poison_test_ds = CustomTensorDataset(poisoned_test_data, transform=transform_test)
            poison_test_loader = DataLoader(poison_test_ds, batch_size=args.batch_size, num_workers=0)
        else:
            raise ValueError(f"[I-BAU] Unsupported poison_type for this baseline: {args.poison_type}")

        # clean val loader (for unlearning)
        perm = np.arange(len(clean_train_raw))
        np.random.shuffle(perm)
        nb_val = int(args.val_ratio * len(clean_train_raw))
        val_data = [clean_train_raw[int(perm[i])] for i in range(nb_val)]
        clean_val = CustomTensorDataset(val_data, transform=transform_train)
        sampler = RandomSampler(data_source=clean_val, replacement=True,
                                num_samples=args.epoch_aggregation * args.batch_size)
        clean_val_loader = DataLoader(clean_val, batch_size=args.batch_size,
                                      shuffle=False, sampler=sampler, num_workers=0)

        return clean_test_loader, poison_test_loader, clean_val_loader

    if args.dataset == 'CIFAR100':
        clean_test_raw = CIFAR100(root=args.data_dir, train=False, download=True, transform=None)
        clean_test_ds = CustomTensorDataset([clean_test_raw[i] for i in range(len(clean_test_raw))], transform=transform_test)
        clean_test_loader = DataLoader(clean_test_ds, batch_size=args.batch_size, num_workers=0)

        clean_train_raw = CIFAR100(root=args.data_dir, train=True, download=True, transform=None)

        if args.poison_type == 'refool':
            source_images = [img for (img, lab) in clean_train_raw if lab == args.poison_source]
            alpha_min, alpha_max = map(float, args.usd_refool_alpha_range.split(','))
            poisoned_test_data = []
            for img, lab in clean_test_raw:
                if lab != args.target_label:
                    src_img = random.choice(source_images)
                    p_img = apply_refool_view_pil(img, src_img, alpha_range=(alpha_min, alpha_max))
                    poisoned_test_data.append((p_img, args.target_label))
                else:
                    poisoned_test_data.append((img, lab))
            poison_test_ds = CustomTensorDataset(poisoned_test_data, transform=transform_test)
            poison_test_loader = DataLoader(poison_test_ds, batch_size=args.batch_size, num_workers=0)

        elif args.poison_type == 'weather':
            poisoned_test_data = []
            for img, lab in clean_test_raw:
                if lab != args.target_label:
                    p_img = add_weather_trigger(img, effect='rain', intensity=args.usd_weather_intensity)
                    poisoned_test_data.append((p_img, args.target_label))
                else:
                    poisoned_test_data.append((img, lab))
            poison_test_ds = CustomTensorDataset(poisoned_test_data, transform=transform_test)
            poison_test_loader = DataLoader(poison_test_ds, batch_size=args.batch_size, num_workers=0)
        else:
            raise ValueError(f"[I-BAU] Unsupported poison_type for this baseline: {args.poison_type}")

        perm = np.arange(len(clean_train_raw))
        np.random.shuffle(perm)
        nb_val = int(args.val_ratio * len(clean_train_raw))
        val_data = [clean_train_raw[int(perm[i])] for i in range(nb_val)]
        clean_val = CustomTensorDataset(val_data, transform=transform_train)
        sampler = RandomSampler(data_source=clean_val, replacement=True,
                                num_samples=args.epoch_aggregation * args.batch_size)
        clean_val_loader = DataLoader(clean_val, batch_size=args.batch_size,
                                      shuffle=False, sampler=sampler, num_workers=0)

        return clean_test_loader, poison_test_loader, clean_val_loader

    if args.dataset == 'GTSRB':
        class Opt:
            data_root = args.data_dir
            input_height = 32
            input_width = 32
            random_crop = 4
            random_rotation = 15
            dataset = 'gtsrb'

        clean_test_raw = GTSRB(Opt(), train=False, transform=None)
        clean_train_raw = GTSRB(Opt(), train=True, transform=None)

        # clean test
        clean_test_data = [clean_test_raw[i] for i in range(len(clean_test_raw))]
        clean_test_ds = CustomTensorDataset(clean_test_data, transform=transform_test)
        clean_test_loader = DataLoader(clean_test_ds, batch_size=args.batch_size, num_workers=0)

        # poison test
        if args.poison_type == 'refool':
            source_images = [img for (img, lab) in clean_train_raw if lab == args.poison_source]
            alpha_min, alpha_max = map(float, args.usd_refool_alpha_range.split(','))
            poisoned_test_data = []
            for img, lab in clean_test_raw:
                if lab != args.target_label:
                    src_img = random.choice(source_images)
                    p_img = apply_refool_view_pil(img, src_img, alpha_range=(alpha_min, alpha_max))
                    poisoned_test_data.append((p_img, args.target_label))
            poison_test_ds = CustomTensorDataset(poisoned_test_data, transform=transform_test)
            poison_test_loader = DataLoader(poison_test_ds, batch_size=args.batch_size, num_workers=0)

        elif args.poison_type == 'weather':
            poisoned_test_data = []
            for img, lab in clean_test_raw:
                if lab != args.target_label:
                    p_img = add_weather_trigger(img, effect='rain', intensity=args.usd_weather_intensity)
                    poisoned_test_data.append((p_img, args.target_label))
            poison_test_ds = CustomTensorDataset(poisoned_test_data, transform=transform_test)
            poison_test_loader = DataLoader(poison_test_ds, batch_size=args.batch_size, num_workers=0)
        else:
            raise ValueError(f"[I-BAU] Unsupported poison_type for this baseline: {args.poison_type}")

        # clean val
        perm = np.arange(len(clean_train_raw))
        np.random.shuffle(perm)
        nb_val = int(args.val_ratio * len(clean_train_raw))
        val_data = [clean_train_raw[int(perm[i])] for i in range(nb_val)]
        clean_val = CustomTensorDataset(val_data, transform=transform_train)

        sampler = RandomSampler(data_source=clean_val, replacement=True,
                                num_samples=args.epoch_aggregation * args.batch_size)
        clean_val_loader = DataLoader(clean_val, batch_size=args.batch_size,
                                      shuffle=False, sampler=sampler, num_workers=0)
        return clean_test_loader, poison_test_loader, clean_val_loader
    
    if args.dataset == 'IMAGENET_SUB':
        train_root = os.path.join(args.data_dir, 'train')
        test_root = os.path.join(args.data_dir, 'test')

        clean_train_raw = datasets.ImageFolder(root=train_root, transform=None)
        clean_test_raw = datasets.ImageFolder(root=test_root, transform=None)

        clean_test_ds = CustomTensorDataset(
            [clean_test_raw[i] for i in range(len(clean_test_raw))],
            transform=transform_test
        )
        clean_test_loader = DataLoader(clean_test_ds, batch_size=args.batch_size, num_workers=4)

        if args.poison_type == 'refool':
            alpha_min, alpha_max = map(float, args.usd_refool_alpha_range.split(','))
            gamma_min, gamma_max = map(float, args.refool_gamma_range.split(','))
            poison_test_ds = create_refool_test_set_imagenet(
                raw_test_dataset=clean_test_raw,
                poison_target=args.target_label,
                poison_source=args.poison_source,
                alpha_range=(alpha_min, alpha_max),
                gamma_range=(gamma_min, gamma_max),
                final_transform=transform_test,
                seed=args.seed
            )
            poison_test_loader = DataLoader(poison_test_ds, batch_size=args.batch_size, num_workers=4)

        elif args.poison_type == 'weather':
            poison_test_ds = create_weather_test_set_imagenet(
                raw_test_dataset=clean_test_raw,
                poison_target=args.target_label,
                final_transform=transform_test,
                effect='rain',
                intensity=args.usd_weather_intensity
            )
            poison_test_loader = DataLoader(poison_test_ds, batch_size=args.batch_size, num_workers=4)
        else:
            raise ValueError(f"[I-BAU] Unsupported poison_type: {args.poison_type}")

        perm = np.arange(len(clean_train_raw))
        np.random.shuffle(perm)
        nb_val = int(args.val_ratio * len(clean_train_raw))
        val_data = [clean_train_raw[int(perm[i])] for i in range(nb_val)]
        clean_val = CustomTensorDataset(val_data, transform=transform_train)

        sampler = RandomSampler(
            data_source=clean_val,
            replacement=True,
            num_samples=args.epoch_aggregation * args.batch_size
        )
        clean_val_loader = DataLoader(clean_val, batch_size=args.batch_size,
                                    shuffle=False, sampler=sampler, num_workers=4)

        return clean_test_loader, poison_test_loader, clean_val_loader

    raise ValueError(f"Unknown dataset: {args.dataset}")


def load_poisoned_checkpoint(args, device):
    ckpt = torch.load(args.checkpoint, map_location=device)
    if isinstance(ckpt, dict) and 'netC' in ckpt:
        state_dict = ckpt['netC']
    elif isinstance(ckpt, dict) and 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
    elif isinstance(ckpt, dict) and 'model' in ckpt:
        state_dict = ckpt['model']
    elif isinstance(ckpt, dict):
        state_dict = ckpt
    else:
        state_dict = ckpt
    return state_dict


def main():
    parser = argparse.ArgumentParser("I-BAU baseline runner (vision) - aligned with your FIP semantic attacks")
    # basic
    parser.add_argument('--dataset', type=str, default='CIFAR10', choices=['CIFAR10', 'CIFAR100', 'GTSRB', 'IMAGENET_SUB'])
    parser.add_argument('--arch', type=str, default='resnet18', choices=['resnet18', 'resnet34'])
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./save/ibau')

    # attack (must match your pipeline)
    parser.add_argument('--poison-type', type=str, required=True, choices=['refool', 'weather'])
    parser.add_argument('--target_label', type=int, required=True)
    parser.add_argument('--poison_source', type=int, default=0)
    parser.add_argument('--usd_refool_alpha_range', type=str, default='0.3,0.6')
    parser.add_argument('--usd_weather_intensity', type=float, default=0.3)

    # loader
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--epoch_aggregation', type=int, default=100)
    parser.add_argument('--gpuid', type=int, default=0)
    parser.add_argument('--seed', type=int, default=123)

    # I-BAU params (paper-aligned knobs)
    parser.add_argument('--ibau_outer_rounds', type=int, default=5)
    parser.add_argument('--ibau_inner_steps', type=int, default=150)
    parser.add_argument('--ibau_alpha_delta', type=float, default=0.6)  
    parser.add_argument('--ibau_beta_theta', type=float, default=7e-2) 
    parser.add_argument('--ibau_c_delta', type=float, default=80.0)
    parser.add_argument('--ibau_solver_iters', type=int, default=8)
    parser.add_argument('--ibau_max_batches', type=int, default=800,
                        help='Limit clean batches per outer round to control runtime (set <=0 for full epoch).')
    
    parser.add_argument('--imagenet_test_resize', type=int, default=256)
    parser.add_argument('--imagenet_crop_size', type=int, default=224)
    parser.add_argument('--refool_gamma_range', type=str, default='0.9,1.1')

    args = parse_args_compatible(parser)

    set_seed(args.seed)

    # [TIME] Init Timer
    tb = TimerBank(sync_cuda=True)
    tb.start("T_total")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda': torch.cuda.set_device(args.gpuid)
    device = torch.device(device)
    num_classes = 10
    if args.dataset == 'CIFAR100': num_classes = 100
    elif args.dataset == 'GTSRB': num_classes = 43
    elif args.dataset == 'IMAGENET_SUB':
        train_root = os.path.join(args.data_dir, 'train')
        num_classes = len(datasets.ImageFolder(root=train_root).classes)

    # ========= (A) T_prep =========
    tb.start("T_prep")
    transform_train, transform_test = build_transforms(
        args.dataset,
        imagenet_test_resize=args.imagenet_test_resize,
        imagenet_crop_size=args.imagenet_crop_size
    )
    clean_test_loader, poison_test_loader, clean_val_loader = build_loaders(args, transform_train, transform_test)
    clean_test_loader_eval = clean_test_loader 
    poison_test_loader_eval = poison_test_loader

    if args.dataset == 'IMAGENET_SUB':
        if args.arch == 'resnet18':
            net = models.resnet18(weights=None)
            net.fc = nn.Linear(net.fc.in_features, num_classes)
        elif args.arch == 'resnet34':
            net = models.resnet34(weights=None)
            net.fc = nn.Linear(net.fc.in_features, num_classes)
        else:
            raise ValueError(args.arch)
    else:
        net = getattr(networks, args.arch)(num_classes=num_classes)
    state_dict = load_poisoned_checkpoint(args, device)
    net.load_state_dict(state_dict)
    net = net.to(device)
    tb.stop("T_prep")
    # ===============================

    criterion = torch.nn.CrossEntropyLoss().cuda()

    # ========= (B) T_eval Before =========
    tb.start("T_eval")
    _, ACC_before = FIP_Test(net, criterion, clean_test_loader_eval)
    _, ASR_before = FIP_Test(net, criterion, poison_test_loader_eval)
    tb.stop("T_eval")
    print(f"\n[Before] ASR: {100*ASR_before:.2f}% | ACC: {100*ACC_before:.2f}%")

    # Unlearning
    import copy
    net_before = copy.deepcopy({k:v.detach().cpu() for k,v in net.state_dict().items()})
    if args.dataset == 'IMAGENET_SUB':
        if args.arch == 'resnet18':
            net_for_unlearn = models.resnet18(weights=None)
            net_for_unlearn.fc = nn.Linear(net_for_unlearn.fc.in_features, num_classes)
        elif args.arch == 'resnet34':
            net_for_unlearn = models.resnet34(weights=None)
            net_for_unlearn.fc = nn.Linear(net_for_unlearn.fc.in_features, num_classes)
        else:
            raise ValueError(args.arch)
        net_for_unlearn = net_for_unlearn.to(device)
    else:
        net_for_unlearn = getattr(networks, args.arch)(num_classes=num_classes).to(device)

    net_for_unlearn.load_state_dict(copy.deepcopy(net.state_dict()))
    net_for_unlearn.train()
    
    # ========= (C) T_opt (Implicitly tracked inside ibau_unlearn) =========
    print("[I-BAU] Starting unlearning...")
    purified = ibau_unlearn(
        model=net_for_unlearn,
        clean_loader=clean_val_loader,
        device=device,
        outer_rounds=args.ibau_outer_rounds,
        inner_steps=args.ibau_inner_steps,
        alpha_delta=args.ibau_alpha_delta,
        beta_theta=args.ibau_beta_theta,
        c_delta=args.ibau_c_delta,
        solver_iters=args.ibau_solver_iters,
    )

    # ========= (D) T_eval After =========
    tb.start("T_eval")
    _, ACC_after = FIP_Test(purified, criterion, clean_test_loader_eval)
    _, ASR_after = FIP_Test(purified, criterion, poison_test_loader_eval)
    tb.stop("T_eval")
    
    print(f"[After]  ASR: {100*ASR_after:.2f}% | ACC: {100*ACC_after:.2f}%")

    # ========= [TIME] Finalize and Save =========
    tb.stop("T_total")

    # T_opt = T_opt_delta + T_opt_theta (Cumulative from ibau_unlearn)
    T_opt_delta = tb.t.get("T_opt_delta", 0.0)
    T_opt_theta = tb.t.get("T_opt_theta", 0.0)
    T_opt = T_opt_delta + T_opt_theta
    
    T_prep = tb.t.get("T_prep", 0.0)
    T_eval = tb.t.get("T_eval", 0.0)
    T_total= tb.t.get("T_total", 0.0)

    print(f"\n[TIME BREAKDOWN] T_total={T_total:.2f}s")
    print(f"  > T_prep = {T_prep:.2f}s")
    print(f"  > T_opt  = {T_opt:.2f}s")
    print(f"    - T_opt_delta = {T_opt_delta:.2f}s")
    print(f"    - T_opt_theta = {T_opt_theta:.2f}s")
    print(f"  > T_eval = {T_eval:.2f}s")

    os.makedirs(args.output_dir, exist_ok=True)
    pd.DataFrame([{
        "T_total_s": T_total,
        "T_prep_s": T_prep,
        "T_opt_s": T_opt,
        "T_opt_delta_s": T_opt_delta,
        "T_opt_theta_s": T_opt_theta,
        "T_eval_s": T_eval,
    }]).to_csv(os.path.join(args.output_dir, f"time_breakdown_{args.dataset}_{args.poison_type}.csv"), index=False)
    
    out_path = os.path.join(args.output_dir, f"ibau_{args.dataset}_{args.arch}_{args.poison_type}.pth")
    torch.save(purified.state_dict(), out_path)
    print(f"\nSaved purified model to: {out_path}")


if __name__ == '__main__':
    main()
