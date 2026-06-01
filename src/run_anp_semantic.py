import sys
_ORIG_ARGV = sys.argv[:]          # 保存真实命令行
sys.argv = sys.argv[:1]           # 临时只留脚本名，防止被导入模块 parse_args() 抢跑

import os
import argparse
import random
import numpy as np
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler
from torchvision.datasets import CIFAR10, CIFAR100
import torchvision.transforms as transforms

import networks
from data.dataloader_gtsrb import GTSRB

# Reuse EXACT same semantic trigger builders used in your pipeline
from train_backdoor_cifar import apply_refool_view_pil, add_weather_trigger, CustomTensorDataset
from torchvision import datasets, models
from poison_imagenet import (
    create_refool_test_set as create_refool_test_set_imagenet,
    create_weather_test_set as create_weather_test_set_imagenet,
    CustomTensorDataset
)
from data_loader_imagenet import IMAGENET_MEAN, IMAGENET_STD

sys.argv = _ORIG_ARGV


# ===== PATCH A: argparse compatibility (data-dir/data_dir, val-ratio/val_ratio) =====
def parse_args_compatible(parser):
    existing = {a.dest for a in parser._actions}

    if 'data_dir' not in existing:
        parser.add_argument('--data_dir', type=str, default='./data', help='dataset root dir (alias)')
    parser.add_argument('--data-dir', dest='data_dir', type=str, default='./data', help='dataset root dir (FIP alias)')

    if 'val_ratio' not in existing:
        parser.add_argument('--val_ratio', type=float, default=0.1, help='val ratio (alias)')
    parser.add_argument('--val-ratio', dest='val_ratio', type=float, default=0.1, help='val ratio (FIP alias)')

    args, unknown = parser.parse_known_args()
    if len(unknown) > 0:
        print(f"[ArgParse] Ignoring unknown args: {unknown}")
    return args
# ===== PATCH A END =====


# ===== PATCH B: exact same metric function as your FIP =====
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


# ===== PATCH C: build eval loaders aligned with your FIP/FIP+USD =====
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
            train_raw = CIFAR10(root=args.data_dir, train=True, download=True, transform=None)
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
                # no else branch (aligned with your FIP GTSRB)

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
    
    elif args.dataset == 'IMAGENET_SUB':
        train_root = os.path.join(args.data_dir, 'train')
        test_root = os.path.join(args.data_dir, 'test')

        clean_train_raw = datasets.ImageFolder(root=train_root, transform=None)
        clean_test_raw = datasets.ImageFolder(root=test_root, transform=None)

        clean_test_data = [clean_test_raw[i] for i in range(len(clean_test_raw))]
        clean_test_ds = CustomTensorDataset(clean_test_data, transform=transform_test)
        clean_test_loader = DataLoader(clean_test_ds, batch_size=batch_size, num_workers=4)

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
            poison_test_loader = DataLoader(poison_test_ds, batch_size=batch_size, num_workers=4)

        elif args.poison_type == 'weather':
            poison_test_ds = create_weather_test_set_imagenet(
                raw_test_dataset=clean_test_raw,
                poison_target=args.target_label,
                final_transform=transform_test,
                effect='rain',
                intensity=args.usd_weather_intensity
            )
            poison_test_loader = DataLoader(poison_test_ds, batch_size=batch_size, num_workers=4)
        else:
            raise ValueError(f"Unsupported poison_type: {args.poison_type}")

        return clean_test_loader, poison_test_loader

    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
# ===== PATCH C END =====


def set_seed(seed: int = 123):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


def build_clean_val_loader(args, transform_train):
    """
    Clean val loader for ANP adversarial search.
    We only need CLEAN samples (no poison), sampled from train set.
    """
    if args.dataset == 'CIFAR10':
        train_raw = CIFAR10(root=args.data_dir, train=True, download=True, transform=None)
    elif args.dataset == 'CIFAR100':
        train_raw = CIFAR100(root=args.data_dir, train=True, download=True, transform=None)
    elif args.dataset == 'GTSRB':
        class Opt:
            data_root = args.data_dir
            input_height = 32
            input_width = 32
            random_crop = 4
            random_rotation = 15
            dataset = 'gtsrb'
        train_raw = GTSRB(Opt(), train=True, transform=None)
    elif args.dataset == 'IMAGENET_SUB':
        train_root = os.path.join(args.data_dir, 'train')
        train_raw = datasets.ImageFolder(root=train_root, transform=None)
    else:
        raise ValueError(args.dataset)

    perm = np.arange(len(train_raw))
    np.random.shuffle(perm)
    nb_val = int(args.val_ratio * len(train_raw))
    val_data = [train_raw[int(perm[i])] for i in range(nb_val)]
    clean_val = CustomTensorDataset(val_data, transform=transform_train)

    sampler = RandomSampler(data_source=clean_val, replacement=True,
                            num_samples=args.anp_max_batches * args.batch_size if args.anp_max_batches > 0 else len(clean_val))
    clean_val_loader = DataLoader(clean_val, batch_size=args.batch_size,
                                  shuffle=False, sampler=sampler, num_workers=0)
    return clean_val_loader


# ------------------------- ANP Core -------------------------
def _resolve_module_by_name(model, dotted):
    m = model
    for p in dotted.split('.'):
        if p.isdigit():
            m = m[int(p)]
        else:
            m = getattr(m, p)
    return m


class ANPOutputScaler(nn.Module):
    """
    Wrap Conv2d/Linear: y = base(x) * (1 + eps * tanh(alpha))
    alpha is optimized adversarially (maximize CE loss on clean set).
    """
    def __init__(self, base: nn.Module, eps: float):
        super().__init__()
        assert isinstance(base, (nn.Conv2d, nn.Linear))
        self.base = base
        self.eps = float(eps)
        out_dim = base.out_channels if isinstance(base, nn.Conv2d) else base.out_features
        self.alpha = nn.Parameter(torch.zeros(out_dim), requires_grad=True)

    def forward(self, x):
        y = self.base(x)
        alpha = self.alpha.to(y.device)          # 关键：对齐 device
        s = 1.0 + self.eps * torch.tanh(alpha)
        if y.dim() == 4:
            return y * s.view(1, -1, 1, 1)
        else:
            return y * s.view(1, -1)


def _iter_named_modules(model):
    for name, mod in model.named_modules():
        yield name, mod


def _select_layer_names(model, arch_name, scope='last', custom=''):
    """
    scope:
      - all: all Conv2d + Linear
      - last: layer4.* + fc (ResNet-style) else fallback to all
      - custom: comma-separated dotted names
    """
    if scope == 'custom':
        names = [s.strip() for s in custom.split(',') if s.strip()]
        return names

    cand = [n for n, m in _iter_named_modules(model) if isinstance(m, (nn.Conv2d, nn.Linear))]
    if scope == 'all':
        return cand

    # last
    keep = [n for n in cand if n.startswith('layer4.') or n == 'fc']
    return keep if len(keep) > 0 else cand


def wrap_model_with_anp(model, layer_names, eps):
    wrapper_map = {}
    for lname in layer_names:
        parent_path = lname.split('.')[:-1]
        leaf = lname.split('.')[-1]
        parent = model
        for p in parent_path:
            parent = parent[int(p)] if p.isdigit() else getattr(parent, p)

        base = getattr(parent, leaf)
        if not isinstance(base, (nn.Conv2d, nn.Linear)):
            continue
        wrapper = ANPOutputScaler(base, eps=eps)
        setattr(parent, leaf, wrapper)
        wrapper_map[lname] = wrapper
    return model, wrapper_map


@torch.no_grad()
def _make_small_clean_loader(clean_loader, max_images):
    xs, ys = [], []
    n = 0
    for xb, yb in clean_loader:
        xs.append(xb.cpu())
        ys.append(yb.cpu())
        n += xb.size(0)
        if n >= max_images:
            break
    x = torch.cat(xs, dim=0)[:max_images]
    y = torch.cat(ys, dim=0)[:max_images]
    ds = torch.utils.data.TensorDataset(x, y)
    return DataLoader(ds, batch_size=clean_loader.batch_size, shuffle=True, num_workers=0)


def anp_adversarial_search(model, wrapper_map, clean_loader_small, steps, lr, device):
    """
    Freeze model weights, optimize alphas to MAXIMIZE CE loss on clean data.
    """
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    for w in wrapper_map.values():
        w.alpha.requires_grad = True

    opt = torch.optim.SGD([w.alpha for w in wrapper_map.values()], lr=lr, momentum=0.9)

    it = iter(clean_loader_small)
    for t in range(steps):
        try:
            xb, yb = next(it)
        except StopIteration:
            it = iter(clean_loader_small)
            xb, yb = next(it)
        xb, yb = xb.to(device), yb.to(device)

        opt.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = F.cross_entropy(logits, yb)

        (-loss).backward()  # gradient ascent
        opt.step()

    scores = {lname: w.alpha.detach().abs().clone() for lname, w in wrapper_map.items()}
    return scores


@torch.no_grad()
def anp_global_prune(model, scores, prune_ratio, prune_bn=False):
    """
    Global prune top prune_ratio channels/neurons by |alpha|.
    Note: selected modules are ANPOutputScaler, real weights are in wrapper.base.
    """
    triples = []
    for lname, v in scores.items():
        for c in range(v.numel()):
            triples.append((float(v[c].item()), lname, c))
    triples.sort(key=lambda x: x[0], reverse=True)

    k = max(1, int(len(triples) * prune_ratio))
    chosen = triples[:k]

    by_layer = defaultdict(list)
    for _, lname, c in chosen:
        by_layer[lname].append(c)

    for lname, chs in by_layer.items():
        wrapper = _resolve_module_by_name(model, lname)
        base = wrapper.base
        idx = torch.tensor(sorted(set(chs)), device=base.weight.device, dtype=torch.long)

        if isinstance(base, nn.Conv2d):
            base.weight.index_fill_(0, idx, 0.0)
            if base.bias is not None:
                base.bias.index_fill_(0, idx, 0.0)
        elif isinstance(base, nn.Linear):
            base.weight.index_fill_(0, idx, 0.0)
            if base.bias is not None:
                base.bias.index_fill_(0, idx, 0.0)

        if prune_bn and isinstance(base, nn.Conv2d):
            # conv2 -> bn2 mapping, conv1 -> bn1
            bn_name = lname.replace('.conv', '.bn')
            if bn_name != lname:
                try:
                    bn = _resolve_module_by_name(model, bn_name)
                    if isinstance(bn, nn.BatchNorm2d):
                        bn.weight.index_fill_(0, idx, 0.0)
                        bn.bias.index_fill_(0, idx, 0.0)
                        bn.running_mean.index_fill_(0, idx, 0.0)
                        bn.running_var.index_fill_(0, idx, 1.0)
                except Exception:
                    pass
            if lname == 'conv1':
                try:
                    bn = _resolve_module_by_name(model, 'bn1')
                    if isinstance(bn, nn.BatchNorm2d):
                        bn.weight.index_fill_(0, idx, 0.0)
                        bn.bias.index_fill_(0, idx, 0.0)
                        bn.running_mean.index_fill_(0, idx, 0.0)
                        bn.running_var.index_fill_(0, idx, 1.0)
                except Exception:
                    pass

    return by_layer


def unwrap_anp_wrappers(model, wrapper_map):
    for lname, wrapper in wrapper_map.items():
        parent_path = lname.split('.')[:-1]
        leaf = lname.split('.')[-1]
        parent = model
        for p in parent_path:
            parent = parent[int(p)] if p.isdigit() else getattr(parent, p)
        setattr(parent, leaf, wrapper.base)
    return model


def run_anp_prune(model, arch_name, clean_val_loader, device,
                  steps=2000, lr=0.1, eps=0.2, prune_ratio=0.05,
                  layer_scope='last', custom_layers='',
                  clean_max=500, prune_bn=False):
    layer_names = _select_layer_names(model, arch_name, scope=layer_scope, custom=custom_layers)
    model, wrapper_map = wrap_model_with_anp(model, layer_names, eps=eps)
    model = model.to(device)

    clean_small = _make_small_clean_loader(clean_val_loader, max_images=clean_max)
    scores = anp_adversarial_search(model, wrapper_map, clean_small, steps=steps, lr=lr, device=device)

    pruned_map = anp_global_prune(model, scores, prune_ratio=prune_ratio, prune_bn=prune_bn)
    model = unwrap_anp_wrappers(model, wrapper_map)

    for p in model.parameters():
        p.requires_grad = True
    return model, pruned_map
# ------------------------- ANP Core END -------------------------


def load_poisoned_checkpoint(args, device):
    ckpt = torch.load(args.checkpoint, map_location=device)
    if isinstance(ckpt, dict) and 'netC' in ckpt:
        return ckpt['netC']
    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        return ckpt['state_dict']
    if isinstance(ckpt, dict) and 'model' in ckpt:
        return ckpt['model']
    if isinstance(ckpt, dict):
        return ckpt
    return ckpt


def main():
    parser = argparse.ArgumentParser("ANP baseline runner (vision) - aligned with your FIP semantic attacks output")
    # basic
    parser.add_argument('--dataset', type=str, default='CIFAR10',
                    choices=['CIFAR10', 'CIFAR100', 'GTSRB', 'IMAGENET_SUB'])
    parser.add_argument('--arch', type=str, default='resnet18', choices=['resnet18', 'resnet34'])
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./save/anp')

    # attack (must match your pipeline)
    parser.add_argument('--poison-type', type=str, required=True, choices=['refool', 'weather'])
    parser.add_argument('--target_label', type=int, required=True)
    parser.add_argument('--poison_source', type=int, default=0)
    parser.add_argument('--usd_refool_alpha_range', type=str, default='0.3,0.6')
    parser.add_argument('--usd_weather_intensity', type=float, default=0.3)

    # eval loader
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--gpuid', type=int, default=0)
    parser.add_argument('--seed', type=int, default=123)

    # ANP params
    parser.add_argument('--anp_steps', type=int, default=2000)
    parser.add_argument('--anp_lr', type=float, default=0.1)
    parser.add_argument('--anp_eps', type=float, default=0.2)
    parser.add_argument('--anp_prune_ratio', type=float, default=0.05)
    parser.add_argument('--anp_clean_max', type=int, default=500, help='Max clean images used for adversarial search.')
    parser.add_argument('--anp_layer_scope', type=str, default='last', choices=['all', 'last', 'custom'])
    parser.add_argument('--anp_layers', type=str, default='')
    parser.add_argument('--anp_prune_bn', action='store_true')
    parser.add_argument('--imagenet_test_resize', type=int, default=256)
    parser.add_argument('--imagenet_crop_size', type=int, default=224)
    parser.add_argument('--refool_gamma_range', type=str, default='0.9,1.1')

    # runtime control for clean_val sampling
    parser.add_argument('--anp_max_batches', type=int, default=500,
                        help='Max number of clean batches drawn (via RandomSampler replacement). Set <=0 for full val set.')

    args = parse_args_compatible(parser)

    set_seed(args.seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        torch.cuda.set_device(args.gpuid)
    device = torch.device(device)

    # num_classes
    num_classes = 10
    if args.dataset == 'CIFAR100':
        num_classes = 100
    elif args.dataset == 'GTSRB':
        num_classes = 43
    elif args.dataset == 'IMAGENET_SUB':
        train_root = os.path.join(args.data_dir, 'train')
        num_classes = len(datasets.ImageFolder(root=train_root).classes)

    transform_train, transform_test = build_transforms(
        args.dataset,
        imagenet_test_resize=args.imagenet_test_resize,
        imagenet_crop_size=args.imagenet_crop_size
    )

    # eval loaders (aligned with FIP output)
    clean_test_loader_eval, poison_test_loader_eval = build_eval_loaders_aligned_with_fip(
        args=args, transform_test=transform_test, batch_size=args.batch_size, num_workers=0
    )

    # clean val loader for ANP search
    clean_val_loader = build_clean_val_loader(args, transform_train)

    # model load
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

    criterion = torch.nn.CrossEntropyLoss().cuda()

    # ===== BEFORE =====
    _, ACC_before = FIP_Test(net, criterion, clean_test_loader_eval)
    _, ASR_before = FIP_Test(net, criterion, poison_test_loader_eval)
    print("\n[Before] ASR / ACC")
    print(f"ASR: {100*ASR_before:.2f}      ACC: {100*ACC_before:.2f}")

    # ===== ANP prune =====
    if args.dataset == 'IMAGENET_SUB':
        if args.arch == 'resnet18':
            net_for_anp = models.resnet18(weights=None)
            net_for_anp.fc = nn.Linear(net_for_anp.fc.in_features, num_classes)
        elif args.arch == 'resnet34':
            net_for_anp = models.resnet34(weights=None)
            net_for_anp.fc = nn.Linear(net_for_anp.fc.in_features, num_classes)
        else:
            raise ValueError(args.arch)
        net_for_anp = net_for_anp.to(device)
    else:
        net_for_anp = getattr(networks, args.arch)(num_classes=num_classes).to(device)

    net_for_anp.load_state_dict({k: v.detach().clone() for k, v in net.state_dict().items()})

    purified, pruned_map = run_anp_prune(
        model=net_for_anp,
        arch_name=args.arch,
        clean_val_loader=clean_val_loader,
        device=device,
        steps=args.anp_steps,
        lr=args.anp_lr,
        eps=args.anp_eps,
        prune_ratio=args.anp_prune_ratio,
        layer_scope=args.anp_layer_scope,
        custom_layers=args.anp_layers,
        clean_max=args.anp_clean_max,
        prune_bn=args.anp_prune_bn
    )
    total_pruned = sum(len(v) for v in pruned_map.values())
    print(f"[ANP] Total pruned channels/neuron units: {total_pruned}")

    # ===== AFTER =====
    _, ACC_after = FIP_Test(purified, criterion, clean_test_loader_eval)
    _, ASR_after = FIP_Test(purified, criterion, poison_test_loader_eval)
    print("\n[After ANP] ASR / ACC")
    print(f"ASR: {100*ASR_after:.2f}      ACC: {100*ACC_after:.2f}")

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f"anp_{args.dataset}_{args.arch}_{args.poison_type}.pth")
    torch.save(purified.state_dict(), out_path)
    print(f"\nSaved pruned model to: {out_path}")


if __name__ == '__main__':
    main()
