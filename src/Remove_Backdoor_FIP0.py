import os
import time
import argparse
import numpy as np
from collections import OrderedDict
import torch
from torch.utils.data import DataLoader, RandomSampler
from torchvision.datasets import CIFAR10, CIFAR100
import torchvision.transforms as transforms
import copy
import torch.nn as nn
import math
import networks
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import pandas as pd
import data.badnets_blend as poison
from torch.autograd import Variable
from PIL import Image
from data.dataloader_cifar import *
import matplotlib.pyplot as plt
from contextlib import nullcontext
import random
from Regularizer0 import CDA_Regularizer as regularizer, FiG_AD_Loss, UnifiedSemanticDefense  ## Regularizer
import torch.autograd as AG
from train_backdoor_cifar import *
from tqdm import tqdm as _tqdm

def tqdm(*args, **kwargs):
    import os
    # 默认关闭 tqdm，防止 nohup/tee 日志被进度条刷爆。
    # 如果需要重新显示进度条：FIP_TQDM=1 python Remove_Backdoor_FIP0.py ...
    if os.environ.get("FIP_TQDM", "0") != "1":
        kwargs["disable"] = True
    return _tqdm(*args, **kwargs)

from torchvision.transforms.functional import gaussian_blur
import json
from collections import defaultdict
from data.dataloader_gtsrb import GTSRB
from torchvision import datasets, models
from data_loader_imagenet import (
    build_imagenet_poisoned_loaders,
    build_imagenet_clean_loaders_only,
    IMAGENET_MEAN,
    IMAGENET_STD
)
from poison_imagenet import (
    create_refool_test_set as create_refool_test_set_imagenet,
    create_weather_test_set as create_weather_test_set_imagenet,
    CustomTensorDataset
)

def parse_range(s, default=(0.2, 0.6)):
    try:
        a, b = map(float, str(s).replace(' ', '').split(','))
        return min(a, b), max(a, b)
    except Exception:
        return default


def sample_weather_effect(effect):
    if effect in ['random', 'mix']:
        return random.choice(['rain', 'snow'])
    return effect


def sample_weather_intensity(args):
    if args.adaptive_attack or args.weather_effect in ['random', 'mix']:
        lo, hi = parse_range(args.weather_intensity_range)
        return random.uniform(lo, hi)
    return args.weather_intensity


class TimerBank:
    def __init__(self, sync_cuda=True):
        self.sync_cuda = sync_cuda
        self.t = {}
        self._st = {}

    def _sync(self):
        if self.sync_cuda and torch.cuda.is_available() and torch.cuda.is_initialized():
            torch.cuda.synchronize(torch.cuda.current_device())

    def start(self, key):
        self._sync()
        self._st[key] = time.perf_counter()

    def stop(self, key):
        if key not in self._st: return 0.0
        self._sync()
        dt = time.perf_counter() - self._st.pop(key)
        self.t[key] = self.t.get(key, 0.0) + dt
        return dt

def _resolve_module_by_name(model, dotted):
    m = model
    for p in dotted.split('.'):
        if p.isdigit(): m = m[int(p)]
        else: m = getattr(m, p)
    return m

@torch.no_grad()
def _collect_class_indices(dataset, max_per_class=128):
    cls_to_ids = defaultdict(list)

    def _get_label(ds, i):
        if isinstance(ds, torch.utils.data.Subset):
            base = ds.dataset
            idx = ds.indices[i]
            return _get_label(base, idx)

        if hasattr(ds, 'targets'):
            return int(ds.targets[i])

        if hasattr(ds, 'labels'):
            return int(ds.labels[i])

        if hasattr(ds, 'samples'):
            return int(ds.samples[i][1])

        if hasattr(ds, 'imgs'):
            return int(ds.imgs[i][1])

        # 最后兜底：直接调用 __getitem__
        item = ds[i]
        if isinstance(item, (tuple, list)) and len(item) >= 2:
            return int(item[1])

        raise AttributeError(
            "Dataset has no targets/labels/samples/imgs, "
            "and __getitem__ does not return (input, label)."
        )

    for i in range(len(dataset)):
        y = _get_label(dataset, i)
        if len(cls_to_ids[y]) < max_per_class:
            cls_to_ids[y].append(i)

    ids = []
    for y in sorted(cls_to_ids.keys()):
        ids += cls_to_ids[y]

    return ids
def _get_mean_std_tensors(args, device):
    mean = torch.tensor(args.data_mean, device=device).view(1, 3, 1, 1)
    std = torch.tensor(args.data_std, device=device).view(1, 3, 1, 1)
    return mean, std

def _denormalize(x, args):
    mean, std = _get_mean_std_tensors(args, x.device)
    return (x * std + mean).clamp(0.0, 1.0)

def _normalize(x, args):
    mean, std = _get_mean_std_tensors(args, x.device)
    return (x - mean) / std
@torch.no_grad()
def detect_pseudo_target_top1(
    model,
    clean_dataset,
    device,
    view_helper,
    num_classes,
    batch_size=64,
    max_per_class=20,
    num_views=3,
    gamma=0.0,
):
    """
    基于多语义视图异常增益的 top-1 目标类别检测。
    返回:
        pred_target: int
        score_vec: torch.Tensor [C]
        mean_vec: torch.Tensor [C]
        var_vec: torch.Tensor [C]
    """
    training_state = model.training
    model.eval()

    # 1) 每类最多取 max_per_class 个干净样本，避免类别分布失衡
    sel_ids = _collect_class_indices(clean_dataset, max_per_class=max_per_class)
    subset_dataset = torch.utils.data.Subset(clean_dataset, sel_ids)

    loader = DataLoader(
        subset_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    score_sum = torch.zeros(num_classes, device=device)
    score_sq_sum = torch.zeros(num_classes, device=device)
    total_count = 0

    for xb, yb in tqdm(loader, desc="[TargetDetect] Scoring candidate targets"):
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        logits_orig = model(xb)  # [B, C]
        B = xb.size(0)

        gain_accum = torch.zeros(B, num_classes, device=device)

        # 先把 normalized 输入还原回 [0,1] 图像空间
        xb_01 = _denormalize(xb, view_helper.args)

        for _ in range(num_views):
            x_view_01, op_used = view_helper._make_view(xb_01, None)

            # 生成完视图后，再归一化给模型
            x_view = _normalize(x_view_01, view_helper.args)
            logits_view = model(x_view)

            # 正向增益 [z_c(view) - z_c(orig)]_+
            delta = (logits_view - logits_orig).clamp(min=0.0)

            # 去掉真实类别，避免把真实标签当作目标类
            delta[torch.arange(B, device=device), yb] = 0.0

            gain_accum += delta

        gain_accum = gain_accum / float(num_views)  # [B, C]

        score_sum += gain_accum.sum(dim=0)
        score_sq_sum += (gain_accum ** 2).sum(dim=0)
        total_count += B

    mean_vec = score_sum / max(total_count, 1)
    var_vec = score_sq_sum / max(total_count, 1) - mean_vec ** 2
    var_vec = torch.clamp(var_vec, min=0.0)

    # 稳健分数：mean - gamma * var
    score_vec = mean_vec - gamma * var_vec

    pred_target = int(score_vec.argmax().item())

    if training_state:
        model.train()

    return pred_target, score_vec.detach().cpu(), mean_vec.detach().cpu(), var_vec.detach().cpu()

def build_high_response_mask(model, device, clean_dataset, layers=('layer3.1.conv2','layer4.1.conv2'),
                           target_label=None, max_per_class=128, percentile=99.0, per_layer_cap=0.10):
    """
    输出: guilty_mask = {layer_name: [out_channel_idx, ...]}
    近似 SODA 的 do(x=x+1) 因果打分：用 |∂z_t/∂feat| × |feat| 的通道平均作为 CA 分数。
    """
    # A.1: 确保在 eval 模式下构建 mask，并在结束后恢复原状态
    training_state = model.training
    model.eval()

    sel_ids = _collect_class_indices(clean_dataset, max_per_class)
    subset_dataset = torch.utils.data.Subset(clean_dataset, sel_ids)
    
    if not hasattr(subset_dataset, 'transform'):
       subset_dataset.transform = clean_dataset.transform

    loader = DataLoader(subset_dataset,
                        batch_size=64, shuffle=False, num_workers=2, pin_memory=True)

    feats = {}
    hooks = []
    for name in layers:
        mod = _resolve_module_by_name(model, name)
        hooks.append(mod.register_forward_hook(lambda m, i, o, n=name: feats.__setitem__(n, o)))

    scores = {name: None for name in layers}

    def _mask_topk_by_percentile(v, p, cap_ratio):
        v_np = v.detach().cpu().numpy()
        th = np.percentile(v_np, p)
        idx = (v_np >= th).nonzero()[0].tolist()
        kcap = max(1, int(len(v_np) * cap_ratio))
        if len(idx) > kcap:
            top_idx = np.argsort(-v_np)[:kcap].tolist()
            return top_idx
        return idx

    for xb, yb in tqdm(loader, desc="[SODA] Calculating Causal Scores"):
        xb, yb = xb.to(device), yb.to(device)
        model.zero_grad(set_to_none=True)
        
        logits = model(xb)
        if target_label is None:
            target_label = int(logits.mean(0).argmax().item())
        
        sel = logits[:, target_label].sum()

        feature_maps_to_grad = []
        valid_layers = []
        for lname in layers:
            if feats.get(lname) is not None:
                feature_maps_to_grad.append(feats[lname])
                valid_layers.append(lname)

        if feature_maps_to_grad:
            feature_grads = torch.autograd.grad(
                outputs=sel,
                inputs=feature_maps_to_grad,
                retain_graph=False
            )
        else:
            feature_grads = []

        for i, lname in enumerate(valid_layers):
            f = feature_maps_to_grad[i]
            f_grad = feature_grads[i]
            ch_score = (f_grad.abs() * f.detach().abs()).mean(dim=(0, 2, 3))
            
            if scores[lname] is None:
                scores[lname] = ch_score
            else:
                scores[lname] += ch_score

    for h in hooks: h.remove()

    guilty_mask = {}
    for lname in layers:
        if scores[lname] is not None:
            v = scores[lname] / max(1, len(loader))
            idx = _mask_topk_by_percentile(v, percentile, per_layer_cap)
            guilty_mask[lname] = idx

    if training_state:
        model.train() # 恢复之前的训练状态
        
    return guilty_mask

@torch.no_grad()
def _topk_by_percentile(v, percentile=99.0, cap_ratio=0.10):
    v_np = v.detach().cpu().numpy()
    th = np.percentile(v_np, percentile)
    idx = np.where(v_np >= th)[0].tolist()
    kcap = max(1, int(len(v_np) * cap_ratio))
    if len(idx) > kcap:
        idx = np.argsort(-v_np)[:kcap].tolist()
    return idx


def build_ours_abnormal_mask(model, device, clean_dataset, view_helper,
                             layers=('layer3.1.conv2','layer4.1.conv2'),
                             target_label=None, max_per_class=128,
                             percentile=99.0, per_layer_cap=0.10,
                             num_views=3, lambda_clean=1.0):
    """
    异常责任通道：
    score = pseudo_target_contrib - lambda_clean * clean_class_contrib
    """
    training_state = model.training
    model.eval()

    sel_ids = _collect_class_indices(clean_dataset, max_per_class)
    subset_dataset = torch.utils.data.Subset(clean_dataset, sel_ids)
    loader = DataLoader(subset_dataset, batch_size=64, shuffle=False, num_workers=2, pin_memory=True)

    feats = {}
    hooks = []
    for name in layers:
        mod = _resolve_module_by_name(model, name)
        hooks.append(mod.register_forward_hook(lambda m, i, o, n=name: feats.__setitem__(n, o)))

    score_pseudo = {name: None for name in layers}
    score_clean  = {name: None for name in layers}

    for xb, yb in tqdm(loader, desc="[OURS] Calculating abnormal responsibility scores"):
        xb, yb = xb.to(device), yb.to(device)

        # ---------- clean contribution ----------
        model.zero_grad(set_to_none=True)
        feats.clear()
        logits_clean = model(xb)

        clean_sel = logits_clean.gather(1, yb.view(-1, 1)).sum()

        clean_feature_maps = []
        valid_layers = []
        for lname in layers:
            if feats.get(lname) is not None:
                clean_feature_maps.append(feats[lname])
                valid_layers.append(lname)

        clean_grads = torch.autograd.grad(
            outputs=clean_sel,
            inputs=clean_feature_maps,
            retain_graph=False,
            create_graph=False
        )

        for i, lname in enumerate(valid_layers):
            f = clean_feature_maps[i]
            g = clean_grads[i]
            ch_score = (g.abs() * f.detach().abs()).mean(dim=(0, 2, 3))
            if score_clean[lname] is None:
                score_clean[lname] = ch_score
            else:
                score_clean[lname] += ch_score

        # ---------- pseudo-target contribution under semantic views ----------
        xb_01 = _denormalize(xb, view_helper.args)
        pseudo_acc = {name: 0.0 for name in layers}

        for _ in range(num_views):
            x_view_01, _ = view_helper._make_view(xb_01, None)
            x_view = _normalize(x_view_01, view_helper.args)

            model.zero_grad(set_to_none=True)
            feats.clear()
            logits_view = model(x_view)

            pseudo_sel = logits_view[:, target_label].sum()

            pseudo_feature_maps = []
            valid_layers = []
            for lname in layers:
                if feats.get(lname) is not None:
                    pseudo_feature_maps.append(feats[lname])
                    valid_layers.append(lname)

            pseudo_grads = torch.autograd.grad(
                outputs=pseudo_sel,
                inputs=pseudo_feature_maps,
                retain_graph=False,
                create_graph=False
            )

            for i, lname in enumerate(valid_layers):
                f = pseudo_feature_maps[i]
                g = pseudo_grads[i]
                ch_score = (g.abs() * f.detach().abs()).mean(dim=(0, 2, 3))
                pseudo_acc[lname] += ch_score

        for lname in layers:
            pseudo_acc[lname] = pseudo_acc[lname] / float(num_views)
            if score_pseudo[lname] is None:
                score_pseudo[lname] = pseudo_acc[lname]
            else:
                score_pseudo[lname] += pseudo_acc[lname]

    guilty_mask = {}
    for lname in layers:
        final_score = score_pseudo[lname] - lambda_clean * score_clean[lname]
        idx = _topk_by_percentile(final_score, percentile=percentile, cap_ratio=per_layer_cap)
        guilty_mask[lname] = idx

    for h in hooks:
        h.remove()

    if training_state:
        model.train()

    return guilty_mask

def build_random_mask_like(model, template_mask, seed=123):
    random.seed(seed)
    guilty_mask = {}

    for lname, idx_list in template_mask.items():
        mod = _resolve_module_by_name(model, lname)
        out_channels = mod.out_channels
        k = len(idx_list)
        guilty_mask[lname] = random.sample(list(range(out_channels)), k)

    return guilty_mask

def save_guilty_mask(guilty_mask, path="guilty_mask.json"):
    with open(path, "w") as f:
        json.dump(guilty_mask, f)
    print(f"[SODA] guilty_mask saved to {path}")

@torch.no_grad()
def evaluate_channel_intervention(model, clean_loader, poison_loader, guilty_mask, target_label, device,
                                  max_batches=50, mode='zero'):
    model.eval()

    def _eval_loader(loader, intervene=False, is_clean=True):
        total = 0
        correct = 0

        sum_target_logit = 0.0
        sum_clean_logit = 0.0
        count_samples = 0

        batch_cnt = 0

        ctx = ChannelIntervention(model, guilty_mask, mode=mode) if intervene else nullcontext()

        with ctx:
            for images, labels in loader:
                images = images.to(device)
                labels = torch.squeeze(labels.to(device))

                logits = model(images)

                pred = torch.max(logits, 1)[1]
                correct += pred.eq(labels.data.view_as(pred)).sum().item()
                total += labels.size(0)

                # 目标类 logit（对 poisoned loader 最有意义）
                tgt_logit = logits[:, target_label].sum().item()
                sum_target_logit += tgt_logit

                # clean loader 上统计真实类别 logit
                if is_clean:
                    gt_logit = logits.gather(1, labels.view(-1, 1)).sum().item()
                    sum_clean_logit += gt_logit

                count_samples += labels.size(0)
                batch_cnt += 1
                if batch_cnt >= max_batches:
                    break

        acc = correct / max(total, 1)
        avg_target_logit = sum_target_logit / max(count_samples, 1)
        avg_clean_logit = sum_clean_logit / max(count_samples, 1) if is_clean else None

        return acc, avg_target_logit, avg_clean_logit

    # baseline
    clean_acc_base, clean_tgt_base, clean_gt_base = _eval_loader(clean_loader, intervene=False, is_clean=True)
    poison_asr_base, poison_tgt_base, _ = _eval_loader(poison_loader, intervene=False, is_clean=False)

    # intervention
    clean_acc_int, clean_tgt_int, clean_gt_int = _eval_loader(clean_loader, intervene=True, is_clean=True)
    poison_asr_int, poison_tgt_int, _ = _eval_loader(poison_loader, intervene=True, is_clean=False)

    result = {
        'clean_acc_base': clean_acc_base,
        'clean_acc_intervened': clean_acc_int,
        'clean_acc_drop': clean_acc_base - clean_acc_int,

        'poison_asr_base': poison_asr_base,
        'poison_asr_intervened': poison_asr_int,
        'poison_asr_drop': poison_asr_base - poison_asr_int,

        'clean_gt_logit_base': clean_gt_base,
        'clean_gt_logit_intervened': clean_gt_int,
        'clean_gt_logit_drop': clean_gt_base - clean_gt_int,

        'poison_target_logit_base': poison_tgt_base,
        'poison_target_logit_intervened': poison_tgt_int,
        'poison_target_logit_drop': poison_tgt_base - poison_tgt_int,
    }

    return result

class ChannelIntervention:
    def __init__(self, model, guilty_mask, mode='zero'):
        self.model = model
        self.guilty_mask = guilty_mask
        self.mode = mode
        self.handles = []

    def _make_hook(self, idx_list):
        def hook(module, inp, out):
            if out is None:
                return out
            if len(idx_list) == 0:
                return out
            idx = torch.tensor(idx_list, device=out.device, dtype=torch.long)
            out_mod = out.clone()
            if self.mode == 'zero':
                out_mod[:, idx, :, :] = 0.0
            else:
                raise ValueError(f"Unsupported intervention mode: {self.mode}")
            return out_mod
        return hook

    def __enter__(self):
        for lname, idx_list in self.guilty_mask.items():
            mod = _resolve_module_by_name(self.model, lname)
            h = mod.register_forward_hook(self._make_hook(idx_list))
            self.handles.append(h)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for h in self.handles:
            h.remove()
        self.handles = []
# ===== PATCH-1 END =====

# ===== PATCH-2: FIP guilty-path gradient masker =====
def mask_non_guilty_grads_with_snapshot(model, guilty_mask, grad_snapshot, g_step=0):
    """
    grad_snapshot: 在 loss.backward() 之后、调用本函数之前创建的梯度快照。
    g_step: 当前全局步数，用于控制日志只打印一次。
    """
    if guilty_mask is None or not guilty_mask: return
    
    # 1) 清零全部 grad
    for p in model.parameters():
        if p.grad is not None:
            p.grad.zero_()

    # 2) 仅恢复 guilty 路径通道的梯度
    for lname, ch_idx in guilty_mask.items():
        if not ch_idx: continue
        
        # --- B.1: 自动恢复相邻BN层的梯度 ---
        # 尝试将 conv 名称映射到 bn 名称 (e.g., 'conv2' -> 'bn2')
        potential_bn_lname = lname.replace('.conv', '.bn')
        layers_to_process = [(lname, 'Conv2d')]
        if potential_bn_lname != lname:
            layers_to_process.append((potential_bn_lname, 'BatchNorm2d'))

        if lname == 'conv1':
            layers_to_process.append(('bn1', 'BatchNorm2d'))

        for layer_name_to_process, layer_type in layers_to_process:
            try:
                mod = _resolve_module_by_name(model, layer_name_to_process)
                is_bn = isinstance(mod, torch.nn.BatchNorm2d)
                is_conv = isinstance(mod, torch.nn.Conv2d)

                if not (is_bn or is_conv):
                    continue

                device = next(mod.parameters()).device
                idx = torch.tensor(ch_idx, device=device)
                
                # --- B.2: 增加日志打印，便于核验 ---
                if g_step == 0:
                    print(f"[Grad Mask] Restoring grads for {len(ch_idx)} channels in {layer_type} layer: {layer_name_to_process}")

                # 定位 weight 和 bias 在 grad_snapshot 中的名字
                wname, bname = None, None
                for n, p in model.named_parameters():
                    if p is mod.weight: wname = n
                    if hasattr(mod, 'bias') and mod.bias is not None and p is mod.bias: bname = n

                # 恢复 weight 梯度
                if wname and wname in grad_snapshot:
                    g_w = grad_snapshot[wname]
                    if mod.weight.grad is not None:
                        if is_conv:
                            mod.weight.grad.index_copy_(0, idx, g_w.index_select(0, idx))
                        elif is_bn:
                            mod.weight.grad.index_copy_(0, idx, g_w.index_select(0, idx))

                # 恢复 bias 梯度
                if bname and bname in grad_snapshot:
                    g_b = grad_snapshot[bname]
                    if mod.bias.grad is not None:
                        mod.bias.grad.index_copy_(0, idx, g_b.index_select(0, idx))

            except AttributeError:
                if layer_type == 'BatchNorm2d' and g_step == 0:
                    # 只在第一次尝试时提示，避免刷屏
                    # print(f"[Grad Mask] Note: No corresponding BN layer found for {lname}")
                    pass
                continue
# ===== PATCH-2 END =====

def get_arch_specific_layer_names(arch_name):
    print(f"[Info] 模型： '{arch_name}'.")

    # 判断是BasicBlock还是Bottleneck
    conv_layer_name = 'conv2' if arch_name in ['resnet18', 'resnet34'] else 'conv3'

    if arch_name == 'resnet18':
        # 针对 ResNet18，我们选择第一个block (index 0) 作为目标
        print("[Info] ResNet18 detected. Targeting the FIRST block (index 0) of each stage.")
        indices = {'layer2': 0, 'layer3': 0, 'layer4': 0}
    elif arch_name in ['resnet34', 'resnet50', 'resnet101']:
        # 针对 ResNet34 及更深模型，选择第二个block (index 1)
        print(f"[Info] {arch_name} detected. Targeting the SECOND block (index 1).")
        indices = {'layer2': 1, 'layer3': 1, 'layer4': 1}
    else:
        # 如果是不支持的架构，则回退到硬编码值并打印警告
        print(f"[Warning] Architecture '{arch_name}' not in pre-defined list. Falling back to default layer names.")
        default_layers = ('layer3.1.conv2', 'layer4.1.conv2')
        wide_layers = ('layer2.1.conv2', 'layer3.1.conv2', 'layer4.1.conv2')
        return default_layers, wide_layers

    # 生成“默认配置”的层名 (包含2个层)
    default_layers = (
        f"layer3.{indices['layer3']}.{conv_layer_name}",
        f"layer4.{indices['layer4']}.{conv_layer_name}"
    )
    
    # 生成“宽配置”的层名 (包含3个层)
    wide_layers = (
        f"layer2.{indices['layer2']}.{conv_layer_name}",
        f"layer3.{indices['layer3']}.{conv_layer_name}",
        f"layer4.{indices['layer4']}.{conv_layer_name}"
    )
    
    print(f"[Info] Generated SODA layers for '{arch_name}':")
    print(f"  - Default Config: {default_layers}")
    print(f"  - Wide Config:    {wide_layers}")
    
    # 总是返回两个元组
    return default_layers, wide_layers

def distillation_loss(student_outputs, teacher_outputs, temperature):
    """cifar10_train = CIFAR10
    计算知识蒸馏损失 (KL散度).
    """
    loss = nn.KLDivLoss(reduction='batchmean')(
        F.log_softmax(student_outputs / temperature, dim=1),
        F.softmax(teacher_outputs / temperature, dim=1)
    )
    # 乘以 T*T 以保持梯度量级
    return loss * (temperature * temperature)

def apply_refool_view_pil(base_img_pil, src_img_pil, alpha_range=(0.8,1.0), gamma_range=(0.9, 1.1)):
    """
    将 USD._make_view 中的 refool_mix 逻辑应用于 PIL 图像，以创建评测集。
    """
    # 转换 PIL 到 Tensor
    base_tensor = TF.to_tensor(base_img_pil)
    src_tensor = TF.to_tensor(src_img_pil)

    # 1) 源图镜像
    src_tensor = torch.flip(src_tensor, dims=[2]) # 水平镜像 C, H, W

    # 2) 生成局部掩膜 + 模糊
    C, H, W = base_tensor.shape
    mask = torch.zeros(1, H, W)
    band_h_min, band_h_max = int(0.20 * H), int(0.50 * H)
    if band_h_min >= band_h_max: band_h_max = band_h_min + 1
    band_h = random.randint(band_h_min, band_h_max)
    y0_max = int(0.35 * H)
    if y0_max <= 0: y0_max = 1
    y0 = random.randint(0, y0_max)
    y1 = min(H, y0 + band_h)
    mask[0, y0:y1, :] = 1.0
    
    k = 7 if min(H, W) >= 32 else 5
    mask = gaussian_blur(mask.unsqueeze(0), kernel_size=(k, k), sigma=(1.0, 2.5)).clamp(0, 1).squeeze(0)

    # 3) 随机不透明度
    alpha_min, alpha_max = alpha_range
    alpha = random.uniform(alpha_min, alpha_max)

    # 4) 伽马/亮度微调
    gamma = random.uniform(gamma_range[0], gamma_range[1])
    x_reflect = (src_tensor ** gamma).clamp(0, 1)

    # 5) 仅在掩膜区域进行“反射混合”
    v_tensor = (1 - alpha * mask) * base_tensor + (alpha * mask) * x_reflect
    
    # 转回 PIL 图像
    return TF.to_pil_image(v_tensor.clamp(0, 1))


# ===== [STD-REV] Standardized semantic/weather variant helpers =====
def _std_parse_float_range(range_str, default=(0.3, 0.3)):
    try:
        a, b = map(float, str(range_str).replace(' ', '').split(','))
        if a > b:
            a, b = b, a
        return (a, b)
    except Exception:
        return default

def _std_sample_weather_effect(effect):
    effect = str(effect).lower()
    if effect in ['random', 'mix']:
        return random.choice(['rain', 'snow'])
    if effect not in ['rain', 'snow']:
        return 'rain'
    return effect

def _std_sample_weather_intensity(intensity):
    if isinstance(intensity, (tuple, list)):
        return random.uniform(float(intensity[0]), float(intensity[1]))
    return float(intensity)

def _std_weather_intensity_arg(args):
    effect = getattr(args, 'weather_effect', 'rain')
    if getattr(args, 'adaptive_attack', False) or effect in ['random', 'mix']:
        return _std_parse_float_range(getattr(args, 'weather_intensity_range', '0.2,0.6'))
    return float(getattr(args, 'weather_intensity', getattr(args, 'usd_weather_intensity', 0.3)))


def _weather_eval_effect(args):
    """Keep old evaluation by default: rain. Only explicit --weather_effect snow uses snow."""
    return 'snow' if getattr(args, 'weather_effect', 'rain') == 'snow' else 'rain'

def _weather_eval_intensity(args):
    """Old rain path uses usd_weather_intensity; snow can use weather_intensity."""
    if getattr(args, 'weather_effect', 'rain') == 'snow':
        return getattr(args, 'weather_intensity', getattr(args, 'usd_weather_intensity', 0.3))
    return getattr(args, 'usd_weather_intensity', 0.3)

def measure_single_image_purify_time(eval_model, teacher_model, defense_criterion,
                                     raw_dataset, transform, device,
                                     N=200, warmup=20, num_views=3, sync_cuda=True):
    import time, random
    eval_model.eval()
    if teacher_model is not None:
        teacher_model.eval()

    ids = list(range(len(raw_dataset)))
    random.shuffle(ids)
    ids = ids[:N]

    def _sync():
        if sync_cuda and device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()

    # warmup
    for i in ids[:min(warmup, len(ids))]:
        img, _ = raw_dataset[i]
        x = transform(img).unsqueeze(0).to(device)
        _sync()
        if defense_criterion is not None and hasattr(defense_criterion, "infer"):
            _ = defense_criterion.infer(eval_model, teacher_model, x, g_step=0, num_views=num_views)
        else:
            _ = eval_model(x)
        _sync()

    times_ms = []
    for i in ids:
        img, _ = raw_dataset[i]
        t0 = time.perf_counter()

        x = transform(img).unsqueeze(0).to(device)
        _sync()
        if defense_criterion is not None and hasattr(defense_criterion, "infer"):
            _ = defense_criterion.infer(eval_model, teacher_model, x, g_step=0, num_views=num_views)
        else:
            _ = eval_model(x)
        _sync()

        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000.0)

    return float(np.mean(times_ms)), float(np.std(times_ms)), times_ms

def main(args, transform_train, transform_test):
    detect_dataset = None

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        torch.cuda.set_device(args.gpuid)
    args.device = device

    tb = TimerBank(sync_cuda=(device == 'cuda'))
    tb.start("T_total")

    if args.dataset == 'IMAGENET_SUB':
        soda_transform = transforms.Compose([
            transforms.Resize(args.imagenet_test_resize),
            transforms.CenterCrop(args.imagenet_crop_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    else:
        soda_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ])

    if args.dataset == 'GTSRB':
        if args.num_classes == 10: args.num_classes = 43
    elif args.dataset == 'CIFAR100':
        if args.num_classes == 10: args.num_classes = 100
        print(f"[Info] Setting Dataset to CIFAR100, Num Classes: {args.num_classes}")

    args_dict = vars(args)
    random.seed(123)
    os.makedirs(args.output_dir, exist_ok=True)
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # torch.cuda.set_device(args.gpuid)
    # args.device = device

    if args.defense_preset == 'balanced':
        print("[USD] Using 'balanced' defense preset.")
        args.usd_lambda_consist = 0.2
        args.usd_lambda_suppress = 0.3
        # 统一为 usd_thresh_start / usd_thresh_end
        args.usd_thresh_start = 0.90
        args.usd_thresh_end = 0.82
        args.lambda_kd = 0.1
    elif args.defense_preset == 'aggressive':
        print("[USD] Using 'aggressive' defense preset.")
        args.usd_lambda_consist = 0.3
        args.usd_lambda_suppress = 0.5
        args.usd_thresh_start = 0.95
        args.usd_thresh_end = 0.88
        args.lambda_kd = 0.2
    elif args.defense_preset == 'strong': 
        print("[USD] Using 'strong' defense preset.")
        # 保持一致性质损失不变，避免过度正则化
        args.usd_lambda_consist = 0.2 
        # 大幅增加抑制损失，这是直接针对后门输出的，更有针对性
        args.usd_lambda_suppress = 0.8 
        # 稍微提高门槛，确保只处理最高置信度的样本，提高信噪比
        args.usd_thresh_start = 0.92
        args.usd_thresh_end = 0.85
        # 适当增加 FiG-AD 的权重，帮助稳定 ACC
        args.lambda_kd = 0.15 


    tb.start("T_prep_data")
    if args.dataset == 'GTSRB':
        print("==> Loading GTSRB Dataset...")
        
        # 模拟 opt 类适配 dataloader_gtsrb.py
        class Opt:
            data_root = args.data_dir
            input_height = 32
            input_width = 32
            random_crop = 4
            random_rotation = 15
            dataset = 'gtsrb'
            
        # 加载原始数据 (PIL) 用于在线投毒
        clean_test_raw = GTSRB(Opt(), train=False, transform=None)
        clean_train_raw = GTSRB(Opt(), train=True, transform=None) # 用于 Source 采样和 Validation
        detect_dataset = GTSRB(Opt(), train=True, transform=transform_test)
        
        # 2.1 构建 Clean Test Loader
        clean_test_data = []
        for i in range(len(clean_test_raw)):
            clean_test_data.append(clean_test_raw[i])
        clean_test_ds = CustomTensorDataset(clean_test_data, transform=transform_test)
        clean_test_loader = DataLoader(clean_test_ds, batch_size=args.batch_size, num_workers=4)
        
        # 2.2 构建 Poison Test Loader (ASR)
        if args.poison_type == 'refool':
            # 收集 Source 图片
            source_images = []
            for i in range(len(clean_train_raw)):
                img, label = clean_train_raw[i]
                if label == args.poison_source:
                    source_images.append(img)
            
            alpha_min, alpha_max = map(float, args.usd_refool_alpha_range.split(','))
            
            poisoned_test_data = []
            for i in range(len(clean_test_raw)):
                img, label = clean_test_raw[i]
                
                # [新增] 强制 Resize 到 32x32
                if img.size != (32, 32):
                    img = img.resize((32, 32), Image.BILINEAR)

                if label != args.target_label:
                    src_img = random.choice(source_images)
                    
                    # [新增] 确保 source 也是 32x32
                    if src_img.size != (32, 32):
                        src_img = src_img.resize((32, 32), Image.BILINEAR)

                    # 使用与训练一致的投毒函数
                    p_img = apply_refool_view_pil(img, src_img, (alpha_min, alpha_max))
                    poisoned_test_data.append((p_img, args.target_label))
                    
            poison_test_ds = CustomTensorDataset(poisoned_test_data, transform=transform_test)
            poison_test_loader = DataLoader(poison_test_ds, batch_size=args.batch_size, num_workers=4)
            
        elif args.poison_type == 'weather':
            poisoned_test_data = []
            for i in range(len(clean_test_raw)):
                img, label = clean_test_raw[i]
                
                # [新增] 强制 Resize 到 32x32
                if img.size != (32, 32):
                    img = img.resize((32, 32), Image.BILINEAR)

                if label != args.target_label:
                    p_img = add_weather_trigger(
                        img,
                        effect=_weather_eval_effect(args),
                        intensity=_weather_eval_intensity(args)
                    )
                    poisoned_test_data.append((p_img, args.target_label))
            poison_test_ds = CustomTensorDataset(poisoned_test_data, transform=transform_test)
            poison_test_loader = DataLoader(poison_test_ds, batch_size=args.batch_size, num_workers=4)
            
        else:
            # Badnets 等其他攻击，若有需要可在此补充，或者抛出警告
            print(f"[Warning] GTSRB loader for {args.poison_type} not implemented in detail. Using clean loader.")
            poison_test_loader = clean_test_loader

        # 2.3 构建 Clean Validation Loader (用于 Mask Finetuning)
        # 从 clean_train_raw 中划分
        perm = np.arange(len(clean_train_raw))
        np.random.shuffle(perm)
        nb_val = int(args.val_ratio * len(clean_train_raw))
        
        # 制作 Validation Set
        val_data = []
        for i in range(nb_val):
            idx = perm[i]
            val_data.append(clean_train_raw[idx])
            
        clean_val = CustomTensorDataset(val_data, transform=transform_train)
        
        sampler = RandomSampler(data_source=clean_val, replacement=True,
                                num_samples=args.epoch_aggregation * args.batch_size)
        clean_val_loader = DataLoader(clean_val, batch_size=args.batch_size,
                                      shuffle=False, sampler=sampler, num_workers=4)
        
    elif args.dataset == 'CIFAR100':
        clean_test = CIFAR100(root=args.data_dir, train=False, download=True, transform=transform_test)
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

        if args.poison_type == 'refool':
            def create_refool_test_set_wrapper(is_cifar100, data_dir, poison_target, poison_source, final_transform):
                # 选择 Dataset 类
                DatasetClass = CIFAR100 if is_cifar100 else CIFAR10
                
                # 获取 Source Images
                source_images = []
                orig_train_temp = DatasetClass(root=data_dir, train=True, download=True, transform=None)
                for img, label in orig_train_temp:
                    if label == poison_source:
                        source_images.append(img)
                if not source_images:
                    raise ValueError(f"Source images for Refool (class {poison_source}) not found.")
                
                # 获取 Clean Test Raw
                raw_test_dataset = DatasetClass(root=data_dir, train=False, download=True, transform=None)
                
                alpha_min, alpha_max = map(float, args.usd_refool_alpha_range.split(','))
                poisoned_test_data = []
                for img, label in raw_test_dataset:
                    if label != poison_target:
                        source_trigger_pil = random.choice(source_images)
                        p_img = apply_refool_view_pil(img, source_trigger_pil, alpha_range=(alpha_min, alpha_max))
                        poisoned_test_data.append((p_img, poison_target))
                    else:
                        poisoned_test_data.append((img, label)) # 保持原本就是target的样本? 评测ASR通常只看非Target，这里保持你原逻辑
                        
                return CustomTensorDataset(poisoned_test_data, transform=final_transform)

            poison_test = create_refool_test_set_wrapper(
                (args.dataset == 'CIFAR100'), args.data_dir, args.target_label, args.poison_source, transform_test
            )
            poison_test_loader = DataLoader(poison_test, batch_size=args.batch_size, num_workers=0)

        elif args.poison_type == 'weather':
            DatasetClass = CIFAR100 if args.dataset == 'CIFAR100' else CIFAR10
            clean_test_raw = DatasetClass(root=args.data_dir, train=False, download=True, transform=None)
            
            poisoned_test_data = []
            for img, label in tqdm(clean_test_raw, desc="Poisoning test set"):
                if label != args.target_label:
                    p_img = add_weather_trigger(
                        img,
                        effect=_weather_eval_effect(args),
                        intensity=_weather_eval_intensity(args)
                    )
                    poisoned_test_data.append((p_img, args.target_label))
                else:
                    poisoned_test_data.append((img, label))
            poison_test_ds = CustomTensorDataset(poisoned_test_data, transform=transform_test)
            poison_test_loader = DataLoader(poison_test_ds, batch_size=args.batch_size, num_workers=0)

        ## Step 1.1: Get the dataloader for Mask finetuning
        cifar_train = CIFAR100(root=args.data_dir, train=True, download=True, transform=transform_train)
        _, clean_val = poison.split_dataset(
            dataset=cifar_train,
            val_frac=args.val_ratio,
            perm=np.loadtxt('./data/cifar_shuffle.txt', dtype=int)
        )
        sampler = RandomSampler(data_source=clean_val, replacement=True,
                                num_samples=args.epoch_aggregation * args.batch_size)
        clean_val_loader = DataLoader(clean_val, batch_size=args.batch_size,
                                    shuffle=False, sampler=sampler, num_workers=0)
        
        detect_dataset = CIFAR100(
            root=args.data_dir,
            train=True,
            download=True,
            transform=transform_test
        )
        
    elif args.dataset == 'IMAGENET_SUB':
        print("==> Loading IMAGENET_SUB Dataset...")

        # 1) 加载 ImageNet-sub 的干净 train / test
        train_root = os.path.join(args.data_dir, 'train')
        test_root = os.path.join(args.data_dir, 'test')

        clean_train_raw = datasets.ImageFolder(root=train_root, transform=None)
        clean_test_raw = datasets.ImageFolder(root=test_root, transform=None)

        args.num_classes = len(clean_train_raw.classes)
        print(f"[Info] IMAGENET_SUB Num Classes: {args.num_classes}")
        print(f"[Info] Classes: {clean_train_raw.classes}")

        # 2) 构建 Clean Test Loader
        clean_test_data = []
        for i in range(len(clean_test_raw)):
            clean_test_data.append(clean_test_raw[i])
        clean_test_ds = CustomTensorDataset(clean_test_data, transform=transform_test)
        clean_test_loader = DataLoader(
            clean_test_ds,
            batch_size=args.batch_size,
            num_workers=4,
            pin_memory=True
        )

        # 3) 构建 Poison Test Loader（和 CIFAR / GTSRB 一样按 target_label / poison_source 在线生成）
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
                seed=123
            )
            poison_test_loader = DataLoader(
                poison_test_ds,
                batch_size=args.batch_size,
                num_workers=4,
                pin_memory=True
            )

        elif args.poison_type == 'weather':
            poison_test_ds = create_weather_test_set_imagenet(
                raw_test_dataset=clean_test_raw,
                poison_target=args.target_label,
                final_transform=transform_test,
                effect=_weather_eval_effect(args),
                intensity=_weather_eval_intensity(args)
            )
            poison_test_loader = DataLoader(
                poison_test_ds,
                batch_size=args.batch_size,
                num_workers=4,
                pin_memory=True
            )
        else:
            print(f"[Warning] IMAGENET_SUB loader for {args.poison_type} not implemented in detail. Using clean loader.")
            poison_test_loader = clean_test_loader

        # 4) 构建 Clean Validation Loader（和 CIFAR / GTSRB 一样，从 clean train 来）
        clean_train_data = []
        for i in range(len(clean_train_raw)):
            clean_train_data.append(clean_train_raw[i])

        clean_val = CustomTensorDataset(clean_train_data, transform=transform_train)

        sampler = RandomSampler(
            data_source=clean_val,
            replacement=True,
            num_samples=args.epoch_aggregation * args.batch_size
        )
        clean_val_loader = DataLoader(
            clean_val,
            batch_size=args.batch_size,
            shuffle=False,
            sampler=sampler,
            num_workers=4,
            pin_memory=True
        )
        detect_dataset = datasets.ImageFolder(
            root=train_root,
            transform=transform_test
        )

    else:

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
            def create_refool_test_set(raw_test_dataset, poison_target, poison_source, final_transform):
                source_images = []
                orig_train_temp = CIFAR10(root=args.data_dir, train=True, download=True, transform=None)
                for img, label in orig_train_temp:
                    if label == poison_source:
                        source_images.append(img)
                if not source_images:
                    raise ValueError("Source images for Refool trigger not found.")
                
                alpha_min, alpha_max = map(float, args.usd_refool_alpha_range.split(','))
                poisoned_test_data = []
                
                for img, label in raw_test_dataset:
                    if label != poison_target:
                        source_trigger_pil = random.choice(source_images)
                        # 使用新的、与训练时一致的视图生成逻辑
                        poisoned_pil_img = apply_refool_view_pil(img, source_trigger_pil, alpha_range=(alpha_min, alpha_max))
                        poisoned_test_data.append((poisoned_pil_img, poison_target))
                    else:
                        poisoned_test_data.append((img, label))

                return CustomTensorDataset(poisoned_test_data, transform=final_transform)

            clean_test_raw = CIFAR10(root=args.data_dir, train=False, download=True, transform=None)
            poison_test = create_refool_test_set(clean_test_raw, args.target_label, args.poison_source, transform_test)
            poison_test_loader = DataLoader(poison_test, batch_size=args.batch_size, num_workers=0)

        elif args.poison_type == 'weather':
            def create_weather_test_set(raw_test_dataset, poison_target, final_transform, effect='rain', intensity=0.3):
                poisoned_test_data = []
                print(f"Generating 'weather' attack ({effect}) test set on the fly...")
                for img, label in tqdm(raw_test_dataset, desc="Poisoning test set"):
                    if label != poison_target:
                        poisoned_pil_img = add_weather_trigger(img, effect=effect, intensity=intensity)
                        poisoned_test_data.append((poisoned_pil_img, poison_target))
                    else:
                        poisoned_test_data.append((img, label))
                return CustomTensorDataset(poisoned_test_data, transform=final_transform)

            clean_test_raw = CIFAR10(root=args.data_dir, train=False, download=True, transform=None)
            poison_test = create_weather_test_set(
                clean_test_raw, args.target_label, transform_test,
                effect=_weather_eval_effect(args), intensity=_weather_eval_intensity(args)
            )
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
        detect_dataset = CIFAR10(
            root=args.data_dir,
            train=True,
            download=True,
            transform=transform_test
        )
    source_samples = None
    # --- 为 USD 准备 source class 样本 (仅 Refool 需要) ---
    if args.use_usd and args.poison_type == 'refool':
        
        if args.dataset == 'GTSRB':
            source_data = []
            for i in range(len(clean_train_raw)):
                img, label = clean_train_raw[i]
                if label == args.poison_source:
                    source_data.append(transform_train(img))
                if len(source_data) >= 128:
                    break
            if len(source_data) > 0:
                source_samples = torch.stack(source_data)

        elif args.dataset == 'IMAGENET_SUB':
            print(f"[USD] Preparing source samples for Refool from class {args.poison_source} on IMAGENET_SUB...")
            source_data = []
            train_raw = datasets.ImageFolder(root=os.path.join(args.data_dir, 'train'), transform=None)
            for img, label in train_raw:
                if label == args.poison_source:
                    source_data.append(transform_train(img))
                if len(source_data) >= 128:
                    break
            if len(source_data) > 0:
                source_samples = torch.stack(source_data)
                print(f"[USD] Collected {len(source_samples)} source samples.")

        else:
            print(f"[USD] Preparing source samples for Refool from class {args.poison_source}...")
            source_data = []
            DatasetClass = CIFAR100 if args.dataset == 'CIFAR100' else CIFAR10
            train_raw = DatasetClass(root=args.data_dir, train=True, download=True, transform=None)
            for img, label in train_raw:
                if label == args.poison_source:
                    source_data.append(transform_train(img))
                if len(source_data) >= 128:
                    break
            if len(source_data) > 0:
                source_samples = torch.stack(source_data)
                print(f"[USD] Collected {len(source_samples)} source samples.")
    tb.stop("T_prep_data")

    ## Step 2: Load Model Checkpoints
    state_dict = torch.load(args.checkpoint, map_location=device)
    if args.poison_type in ['Dynamic']:
        state_dict = torch.load(args.checkpoint, map_location=device)['netC']

    if args.dataset == 'IMAGENET_SUB':
        net = build_imagenet_model(args.arch, args.num_classes, pretrained=False).to(device)
        state_dict = state_dict["model"] if isinstance(state_dict, dict) and "model" in state_dict else state_dict
        net.load_state_dict(state_dict, strict=False)
    else:
        net = getattr(networks, args.arch)(num_classes=args.num_classes)
        net.load_state_dict(state_dict)
        net = net.cuda()

    # ============================================================
    # Target label semantics:
    # - args.target_label is the true attack target used for poison-test construction.
    # - args.defense_target_label optionally overrides the target used by defense modules.
    #   This is only for failure-impact analysis.
    # ============================================================
    args.true_target_label = int(args.target_label)

    if (not args.run_target_detection) and getattr(args, 'defense_target_label', -1) >= 0:
        print(f"[Defense Target Override] true attack target = {args.true_target_label}")
        print(f"[Defense Target Override] defense target     = {args.defense_target_label}")
        args.target_label = int(args.defense_target_label)
    else:
        args.defense_target_label = int(args.target_label)

    print(f"[Target Config] true_target_label    = {args.true_target_label}")
    print(f"[Target Config] defense_target_label = {args.defense_target_label}")

    enable_soda = (args.use_fig_ad or args.use_usd)
    if enable_soda:
        print("\n[A. SODA-CA] ==> Starting SODA-style Causal Localization...")
        
        # 构建 SODA 数据集
        if args.dataset == 'GTSRB':
            soda_dataset = GTSRB(Opt(), train=True, transform=soda_transform)
        elif args.dataset == 'CIFAR100':
            soda_dataset = CIFAR100(root=args.data_dir, train=True, download=True, transform=transform_test)
        elif args.dataset == 'IMAGENET_SUB':
            soda_dataset = datasets.ImageFolder(root=os.path.join(args.data_dir, 'train'), transform=transform_test)
        else:
            soda_dataset = CIFAR10(root=args.data_dir, train=True, download=True, transform=transform_test)

        soda_layers_default, soda_layers_wide = get_arch_specific_layer_names(args.arch)

        # 策略选择
        if args.dataset == 'GTSRB':
            print(f"[SODA] GTSRB detected. Forcing WIDE layer configuration (Plan B).")
            soda_layers = soda_layers_wide
            soda_percentile = 97.0
            soda_cap = 0.20
        elif args.poison_type == 'refool':
            # CIFAR10 Refool
            soda_layers = soda_layers_wide
            soda_percentile = 97.0
            soda_cap = 0.20
        elif args.dataset == 'CIFAR100':
            soda_layers = soda_layers_wide
            soda_percentile = 97.0
            soda_cap = 0.20
        elif args.poison_type == 'weather':
            # CIFAR10 Weather
            soda_layers = ('conv1', 'layer1.0.conv1', 'layer2.0.conv1')
            soda_percentile = 98.0
            soda_cap = 0.15
        else:
            soda_layers = soda_layers_default
            soda_percentile = 99.0
            soda_cap = 0.10

        tb.start("T_prep_soda")
        # 用 USD 的 view helper 复用多视图生成
        view_helper = UnifiedSemanticDefense(args, source_samples).to(device)
        if view_helper.source_samples is not None:
            view_helper.source_samples = _denormalize(view_helper.source_samples, view_helper.args)

        if args.mask_strategy == 'high_response':
            guilty_mask = build_high_response_mask(
                net, device=args.device, clean_dataset=soda_dataset,
                layers=soda_layers, target_label=args.target_label,
                max_per_class=128, percentile=soda_percentile, per_layer_cap=soda_cap
            )

        elif args.mask_strategy == 'ours':
            guilty_mask = build_ours_abnormal_mask(
                net, device=args.device, clean_dataset=soda_dataset,
                view_helper=view_helper,
                layers=soda_layers, target_label=args.target_label,
                max_per_class=128, percentile=soda_percentile,
                per_layer_cap=soda_cap, num_views=args.mask_num_views,
                lambda_clean=args.mask_lambda_clean
            )

        elif args.mask_strategy == 'random':
            template_mask = build_ours_abnormal_mask(
                net, device=args.device, clean_dataset=soda_dataset,
                view_helper=view_helper,
                layers=soda_layers, target_label=args.target_label,
                max_per_class=128, percentile=soda_percentile,
                per_layer_cap=soda_cap, num_views=args.mask_num_views,
                lambda_clean=args.mask_lambda_clean
            )
            guilty_mask = build_random_mask_like(net, template_mask, seed=args.mask_random_seed)

        else:
            raise ValueError(f"Unknown mask_strategy: {args.mask_strategy}")

        total_guilty_channels = sum(len(v) for v in guilty_mask.values())
        print(f"[Mask Strategy] {args.mask_strategy}, total selected channels: {total_guilty_channels}")

        mask_path = os.path.join(args.output_dir, f"guilty_mask_{args.mask_strategy}.json")
        save_guilty_mask(guilty_mask, path=mask_path)
    else:
        guilty_mask = None
        print("[SODA] Disabled: running pure FIP baseline.")

    # ===== Channel Intervention Experiment =====
    if args.run_channel_intervention:
        print("\n[Intervention] ==> Running channel intervention experiment...")

        result = evaluate_channel_intervention(
            model=net,
            clean_loader=clean_test_loader,
            poison_loader=poison_test_loader,
            guilty_mask=guilty_mask,
            target_label=args.target_label,
            device=device,
            max_batches=args.intervention_batches,
            mode=args.intervention_mode
        )

        print("[Intervention] Results:")
        for k, v in result.items():
            print(f"  {k}: {v:.6f}" if isinstance(v, float) else f"  {k}: {v}")

        save_path = os.path.join(args.output_dir, args.intervention_save_name)
        df = pd.DataFrame([{
            'dataset': args.dataset,
            'arch': args.arch,
            'attack': args.poison_type,
            'mask_strategy': args.mask_strategy,
            **result
        }])
        df.to_csv(save_path, index=False)
        print(f"[Intervention] Saved to: {save_path}")
        return

    net.train()

    # ===== Target Detection (Top-1 / Top-3) =====
    if args.run_target_detection:
        print("\n[TargetDetect] ==> Running pseudo-target detection...")

        if detect_dataset is None:
            raise ValueError(
                "detect_dataset is not prepared. "
                "Please make sure detect_dataset is set for CIFAR10/CIFAR100/GTSRB/IMAGENET_SUB."
            )

        detect_view_helper = UnifiedSemanticDefense(args, source_samples).to(device)

        if detect_view_helper.source_samples is not None:
            detect_view_helper.source_samples = _denormalize(
                detect_view_helper.source_samples,
                detect_view_helper.args
            )

        tb.start("T_target_detect")
        pred_target, score_vec, mean_vec, var_vec = detect_pseudo_target_top1(
            model=net,
            clean_dataset=detect_dataset,
            device=device,
            view_helper=detect_view_helper,
            num_classes=args.num_classes,
            batch_size=args.target_detect_batch_size,
            max_per_class=args.target_detect_max_per_class,
            num_views=args.target_detect_num_views,
            gamma=args.target_detect_gamma,
        )
        tb.stop("T_target_detect")

        topk = min(5, args.num_classes)
        top_vals, top_idxs = torch.topk(score_vec, k=topk)

        top_classes = [int(x) for x in top_idxs.tolist()]
        top_scores = [float(x) for x in top_vals.tolist()]

        hit_top1 = int(pred_target == args.true_target_label)
        hit_top3 = int(args.true_target_label in top_classes[:min(3, len(top_classes))])

        print(f"[TargetDetect] True target      : {args.true_target_label}")
        print(f"[TargetDetect] Predicted top-1  : {pred_target}")
        print(f"[TargetDetect] Hit top-1        : {hit_top1}")
        print(f"[TargetDetect] Hit top-3        : {hit_top3}")
        print("[TargetDetect] Top scores:")
        for rank, (idx, val) in enumerate(zip(top_classes, top_scores), start=1):
            print(
                f"  Top-{rank}: class={idx}, "
                f"score={val:.6f}, mean={float(mean_vec[idx]):.6f}, var={float(var_vec[idx]):.6f}"
            )

        result_path = os.path.join(args.output_dir, args.target_detect_save_name)

        top1_score = top_scores[0] if len(top_scores) >= 1 else -1.0
        top2_score = top_scores[1] if len(top_scores) >= 2 else -1.0
        margin = top1_score - top2_score if len(top_scores) >= 2 else -1.0

        df = pd.DataFrame([{
            "experiment_tag": args.experiment_tag,
            "checkpoint": args.checkpoint,
            "dataset": args.dataset,
            "arch": args.arch,
            "attack": args.poison_type,
        "attack_variant": getattr(args, "attack_variant", "standard"),
        "poison_rate": float(getattr(args, "poison_rate", -1)),
        "val_ratio": float(getattr(args, "val_ratio", -1)),
        "weather_effect": getattr(args, "weather_effect", ""),
        "weather_intensity": float(getattr(args, "weather_intensity", getattr(args, "usd_weather_intensity", -1))),
        "weather_intensity_range": getattr(args, "weather_intensity_range", ""),
        "adaptive_attack": bool(getattr(args, "adaptive_attack", False)),
        "initial_ASR_percent": float(100.0 * initial_ASR),
        "initial_ACC_percent": float(100.0 * initial_ACC),

            "true_target": int(args.true_target_label),
            "pred_target_top1": int(pred_target),
            "hit_top1": int(hit_top1),

            "top3_classes": json.dumps(top_classes[:min(3, len(top_classes))]),
            "hit_top3": int(hit_top3),

            "top5_classes": json.dumps(top_classes),
            "top5_scores": json.dumps(top_scores),

            "top1_score": float(top1_score),
            "top2_score": float(top2_score),
            "top1_top2_margin": float(margin),

            "num_views": int(args.target_detect_num_views),
            "max_per_class": int(args.target_detect_max_per_class),
            "gamma": float(args.target_detect_gamma),
            "T_target_detect_s": float(tb.t.get("T_target_detect", 0.0)),
        }])
        df.to_csv(result_path, index=False)

        # Save all class scores for plotting and wrong-target selection
        score_path = result_path.replace(".csv", "_scores.csv")
        score_df = pd.DataFrame({
            "class_id": list(range(args.num_classes)),
            "score": [float(x) for x in score_vec.tolist()],
            "mean_gain": [float(x) for x in mean_vec.tolist()],
            "var_gain": [float(x) for x in var_vec.tolist()],
        })
        score_df = score_df.sort_values("score", ascending=False).reset_index(drop=True)
        score_df["rank"] = score_df.index + 1
        score_df.to_csv(score_path, index=False)

        print(f"[TargetDetect] Result saved to: {result_path}")
        print(f"[TargetDetect] Score saved to : {score_path}")
        return
    # ======================== 核心修改 (START)：解耦FiG-AD和SVC的Teacher Model创建 ========================
    teacher_model = None
    kd_criterion = None
    defense_criterion = None

    if args.use_fig_ad or args.use_usd:
        tb.start("T_prep_teacher")
        print("\n[Framework] Teacher model required. Creating an EMA-ready copy.")
        teacher_model = copy.deepcopy(net).to(device)
        teacher_model.eval()
        for param in teacher_model.parameters():
            param.requires_grad = False
        tb.stop("T_prep_teacher")

    if args.use_fig_ad:
        print("[Framework] ==> FiG-AD ENABLED.")
        kd_criterion = FiG_AD_Loss(args, teacher_model).to(device)

    if args.use_usd:
        print("[Framework] ==> Unified Semantic Defense (USD) ENABLED.")
        defense_criterion = UnifiedSemanticDefense(args, source_samples).to(device)
        
        # ③-B: 提高Refool时的正则强度
        if args.poison_type == 'refool' and defense_criterion is not None:
            print("[USD] Overriding parameters for strong Refool defense.")
            defense_criterion.lambda_suppress = 0.6 
            defense_criterion.confidence_thresh = max(getattr(defense_criterion, 'confidence_thresh', 0.85), 0.90)
            defense_criterion.usd_refool_alpha_range = (0.30, 0.60)
            defense_criterion.topk_ratio = max(getattr(defense_criterion, 'topk_ratio', 0.03), 0.05)
        if args.poison_type == 'weather':
            print("[USD] Overriding parameters for strong Weather defense.")
            # cifar-10
            # 1) 适当增强雨视图强度 & 收紧 delta 门控，更精准地识别后门激活
            args.usd_weather_intensity = max(args.usd_weather_intensity, 0.35)
            args.usd_weather_delta_thresh = max(args.usd_weather_delta_thresh, 0.06)

            # 2) 抑制损失更温和，避免误伤目标类，保护ACC
            # 对于cifar-10 resnet-18 weather适当调低权重，避免误伤目标类 0.5 0.2
            # 对于cifar-10 resnet-34 weather适当调低权重，避免误伤目标类 0.25-0.3 0.15
            args.usd_margin = 0.5
            args.usd_lambda_suppress = 0.3 # 调整权重
            args.usd_lambda_consist  = 0.15


            # cifar100-resnet18
            # # 1) 适当增强雨视图强度 & 收紧 delta 门控，更精准地识别后门激活
            # args.usd_weather_intensity = max(args.usd_weather_intensity, 0.3)
            # args.usd_weather_delta_thresh = max(args.usd_weather_delta_thresh, 0.02)

            # # 2) 抑制损失更温和，避免误伤目标类，保护ACC
            # args.usd_margin = 0.1
            # args.usd_lambda_suppress = 0.1 # 调整权重
            # args.usd_lambda_consist  = 0.2

            # cifar100-resnet34
            # # 1) 适当增强雨视图强度 & 收紧 delta 门控，更精准地识别后门激活
            # args.usd_weather_intensity = max(args.usd_weather_intensity, 0.3)
            # args.usd_weather_delta_thresh = max(args.usd_weather_delta_thresh, 0.04)

            # # 2) 抑制损失更温和，避免误伤目标类，保护ACC
            # args.usd_margin = 0.15
            # args.usd_lambda_suppress = 0.15 # 调整权重
            # args.usd_lambda_consist  = 0.2

            # gtsrb-resnet18
            # args.usd_weather_intensity = max(args.usd_weather_intensity, 0.3)
            # args.usd_weather_delta_thresh = max(args.usd_weather_delta_thresh, 0.02)

            # # 2) 抑制损失更温和，避免误伤目标类，保护ACC
            # args.usd_margin = 0.1
            # args.usd_lambda_suppress = 0.1 # 调整权重
            # args.usd_lambda_consist  = 0.2

            # # 3) (可选) 放宽通道收缩的top-k比例 (如果USD内部使用了此项)
            # args.usd_topk_ratio = max(args.usd_topk_ratio, 0.05)
            
    # ======================== 核心修改 (END) =============================================================

    ## Step 3: Training Settings
    criterion = torch.nn.CrossEntropyLoss().cuda()
    student_params = net.parameters()
    
    if args.use_fig_ad and args.use_ttm:
        # If TTM is used, add its parameters to the optimizer.
        print("Optimizer will also train the TTM adapter.")
        trainable_params = list(student_params) + list(kd_criterion.t_adapter.parameters())
        optimizer = torch.optim.SGD(trainable_params, lr=args.lr, momentum=0.95)
    else:
        optimizer = torch.optim.SGD(student_params, lr=args.lr, momentum=0.95)
        
    nb_iterations = int(np.ceil(args.nb_epochs / args.epoch_aggregation))

    ## Initialize FIM
    tb.start("T_prep_fim")
    criterion_reg = regularizer(args, device, net, criterion, nb_iterations)

    ewc_dataset = clean_val
    

    criterion_reg.register_ewc_params(ewc_dataset, 100, 100)
    tb.stop("T_prep_fim")
    # K_fim = 200 if args.dataset == 'CIFAR100' else 100
    # fim_total_samples = min(len(clean_val), K_fim * args.num_classes) # CIFAR100 -> min(len, 200*100=20000)
    # fim_num_batches = math.ceil(fim_total_samples / args.batch_size)
    
    # print(f"[FIM] Initializing Fisher Information Matrix with adaptive sampling:")
    # print(f"      Dataset: {args.dataset}, Classes: {args.num_classes}")
    # print(f"      Target Samples: {fim_total_samples} (approx {fim_total_samples/args.num_classes:.1f} per class)")
    # print(f"      Batches: {fim_num_batches}")

    # criterion_reg.register_ewc_params(clean_val, fim_num_batches, fim_total_samples)

    # # Step 3: train backdoored models
    N_c = len(clean_val) / args.num_classes

    ## Step 4: Validate the Given Model
    cl_test_loss, ACC = FIP_Test(model=net, criterion=criterion, data_loader=clean_test_loader)
    po_test_loss, ASR = FIP_Test(model=net, criterion=criterion, data_loader=poison_test_loader)
    print("ASR and ACC Before Purification\t")
    print('-----------------------------------------------------------------')
    print('ASR \t ACC')
    print('{:.4f} \t {:.4f}'.format(100 * ASR, 100 * ACC))
    initial_ASR = ASR
    initial_ACC = ACC
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
        train_loss, train_acc = FIP_Train(args, i, net, clean_val_loader, criterion, criterion_reg, 
                                          optimizer, kd_criterion, defense_criterion, teacher_model, nb_iterations, guilty_mask,timer_bank=tb)
        
        eval_model = teacher_model if teacher_model is not None else net
        if eval_model is not net:
            # 在第一个epoch结束后打印一次提示信息
            if i == 0:
                print("\n[Evaluation] Evaluating on the stable EMA teacher model to avoid purification-state noise.")
        
        tb.start("T_eval")
        clean_loss, ACC = FIP_Test(model=eval_model, criterion=criterion, data_loader=clean_test_loader)
        poison_loss, ASR = FIP_Test(model=eval_model, criterion=criterion, data_loader=poison_test_loader)
        tb.stop("T_eval")

        clean_losses[i] = clean_loss
        poison_losses[i] = poison_loss
        clean_accs[i] = ACC
        poison_accs[i] = ASR

        ## Save Stattistics and the Purified model
        np.savez(os.path.join(args.output_dir, 'remove_model_' + args.poison_type + '_' + str(args.dataset) + '_.npz'),
                 cl_loss=clean_losses, cl_test=clean_accs, po_loss=poison_losses, po_acc=poison_accs)
        model_save = args.poison_type + '_' + str(i) + '_' + str(args.dataset) + '.pth'
        if os.environ.get("SAVE_PURIFY_PTH", "0") == "1":
            torch.save(net.state_dict(), os.path.join(args.output_dir, model_save))
        else:
            pass  # skip saving purification .pth checkpoints to save disk
        # scheduler.step()

        print('{} \t {:.4f} \t {:.4f}'.format((i + 1) * args.epoch_aggregation, 100 * ASR, 100 * ACC))

    tb.stop("T_total")
    
    T_prep = tb.t.get("T_prep_data",0) + tb.t.get("T_prep_soda",0) + tb.t.get("T_prep_teacher",0) + tb.t.get("T_prep_fim",0)
    T_opt  = tb.t.get("T_opt",0)
    T_eval = tb.t.get("T_eval",0)
    T_total= tb.t.get("T_total",0)

    print(f"\n[TIME BREAKDOWN] T_total={T_total:.2f}s")
    print(f"  > T_prep = {T_prep:.2f}s (Data/SODA/Teacher/FIM)")
    print(f"  > T_opt  = {T_opt:.2f}s  (Purification Training)")
    print(f"  > T_eval = {T_eval:.2f}s (ASR/ACC Checks)")

    pd.DataFrame([{
        "T_total_s": T_total,
        "T_prep_s": T_prep,
        "T_opt_s": T_opt,
        "T_eval_s": T_eval,
        "T_prep_data_s": tb.t.get("T_prep_data",0),
        "T_prep_soda_s": tb.t.get("T_prep_soda",0),
        "T_prep_teacher_s": tb.t.get("T_prep_teacher",0),
        "T_prep_fim_s": tb.t.get("T_prep_fim",0),
    }]).to_csv(os.path.join(args.output_dir, "time_breakdown.csv"), index=False)
    summary_path = os.path.join(
        args.output_dir,
        f"purification_summary_{args.experiment_tag}.csv"
    )

    pd.DataFrame([{
        "experiment_tag": args.experiment_tag,
        "checkpoint": args.checkpoint,
        "dataset": args.dataset,
        "arch": args.arch,
        "attack": args.poison_type,
        "attack_variant": getattr(args, "attack_variant", "standard"),
        "poison_rate": float(getattr(args, "poison_rate", -1)),
        "val_ratio": float(getattr(args, "val_ratio", -1)),
        "weather_effect": getattr(args, "weather_effect", ""),
        "weather_intensity": float(getattr(args, "weather_intensity", getattr(args, "usd_weather_intensity", -1))),
        "weather_intensity_range": getattr(args, "weather_intensity_range", ""),
        "adaptive_attack": bool(getattr(args, "adaptive_attack", False)),
        "initial_ASR_percent": float(100.0 * initial_ASR),
        "initial_ACC_percent": float(100.0 * initial_ACC),

        "true_target_label": int(getattr(args, "true_target_label", args.target_label)),
        "defense_target_label": int(getattr(args, "defense_target_label", args.target_label)),
        "target_match": int(
            int(getattr(args, "true_target_label", args.target_label))
            == int(getattr(args, "defense_target_label", args.target_label))
        ),

        "final_ASR_percent": float(100.0 * poison_accs[-1]),
        "final_ACC_percent": float(100.0 * clean_accs[-1]),

        "T_total_s": float(T_total),
        "T_prep_s": float(T_prep),
        "T_opt_s": float(T_opt),
        "T_eval_s": float(T_eval),

        "reg_F": float(args.reg_F),
        "use_usd": bool(args.use_usd),
        "mask_strategy": args.mask_strategy,
        "mask_num_views": int(args.mask_num_views),
        "mask_lambda_clean": float(args.mask_lambda_clean),
    }]).to_csv(summary_path, index=False)

    print(f"[Summary] Saved to: {summary_path}")
def build_imagenet_model(arch_name, num_classes, pretrained=False):
    if arch_name == 'resnet18':
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif arch_name == 'resnet34':
        weights = models.ResNet34_Weights.DEFAULT if pretrained else None
        model = models.resnet34(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        raise ValueError(f"Unsupported ImageNet-sub arch: {arch_name}")
    return model

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
def FIP_Train(args, epoch, net, clean_val_loader, criterion, criterion_reg, optimizer, 
              kd_criterion, defense_criterion, teacher_model, nb_iterations, guilty_mask, 
              timer_bank=None):
    print('\nEpoch: %d' % epoch)
    net.train()
    total_steps = nb_iterations * len(clean_val_loader)

    if not (args.use_usd or args.use_fig_ad):
        # print("[FIP_Train] Pure FIP path: no USD/FiG-AD; SODA disabled.")
        train_loss = 0; correct = 0; total = 0
        desc = ('[%s][LR=%s] Loss: %.3f | Acc: %.3f%% (%d/%d)' %
                ('Fisher', args.lr, 0, 0, correct, total))
        prog_bar = tqdm(enumerate(clean_val_loader), total=len(clean_val_loader), desc=desc, leave=True)
        for batch_idx, (inputs, targets) in prog_bar:
            if timer_bank: timer_bank.start("T_opt")
            inputs, targets = inputs.cuda(), targets.cuda()
            loss, outputs = criterion_reg.forward_backward_update(inputs, targets, batch_idx)
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            if timer_bank: timer_bank.stop("T_opt")
            desc = ('[%s][LR=%s] Loss: %.3f | Acc: %.3f%% (%d/%d)' %
                    ('Fisher', args.lr, train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
            prog_bar.set_description(desc, refresh=True)
        return train_loss / (batch_idx + 1), 100. * correct / total
    
    train_loss = 0
    correct = 0
    total = 0
    desc = ('[%s][LR=%s] Loss: %.3f | Acc: %.3f%% (%d/%d)' %
            ('Fisher', args.lr, 0, 0, correct, total))

    prog_bar = tqdm(enumerate(clean_val_loader), total=len(clean_val_loader), desc=desc, leave=True)
    for batch_idx, (inputs, targets) in prog_bar:
        if timer_bank: timer_bank.start("T_opt")
        g_step = epoch * len(clean_val_loader) + batch_idx
        inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        
        outputs = net(inputs)
        ce_loss = criterion(outputs, targets)
        reg_loss = criterion_reg._compute_reg_loss(criterion_reg.weight)
        total_loss = ce_loss + reg_loss
        
        progress = min(1.0, g_step / total_steps)
        purify_phase = (progress < 0.30)
        is_purification_batch = purify_phase and (batch_idx % criterion_reg.iter_gap == 0)
        annealed_regF = args.reg_F * (1.0 - progress) ** 2

        if is_purification_batch:
            trace_loss = criterion_reg.get_trace_loss(outputs, targets)
            total_loss += annealed_regF * trace_loss
        else:
            if args.use_usd and defense_criterion is not None and teacher_model is not None:
                current_thresh = args.usd_thresh_start - (args.usd_thresh_start - args.usd_thresh_end) * progress
                defense_criterion.confidence_thresh = current_thresh
                usd_loss = defense_criterion(net, teacher_model, inputs, targets, g_step)
                total_loss += usd_loss
            
            if args.use_fig_ad and kd_criterion is not None:
                kd_loss = kd_criterion(outputs, inputs, targets)
                total_loss += args.lambda_kd * kd_loss
            
            if (args.use_fig_ad or args.use_usd) and teacher_model is not None:
                with torch.no_grad():
                    m = args.ema_tau
                    for param_t, param_s in zip(teacher_model.parameters(), net.parameters()):
                        param_t.data.mul_(m).add_(param_s.data, alpha=1.0 - m)
        
        total_loss.backward()
        if is_purification_batch and guilty_mask:
            grad_snapshot = {n: p.grad.detach().clone() for n, p in net.named_parameters() if p.grad is not None}
            mask_non_guilty_grads_with_snapshot(net, guilty_mask, grad_snapshot, g_step)
        
        optimizer.step()
        if timer_bank: timer_bank.stop("T_opt")

        train_loss += total_loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        desc = ('[%s][LR=%s] Loss: %.3f | Acc: %.3f%% (%d/%d)' %
                ('Fisher', args.lr, train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
        prog_bar.set_description(desc, refresh=True)

    return train_loss / (batch_idx + 1), 100. * correct / total

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
                        choices=['badnets', 'Feature', 'FC', 'SIG', 'Dynamic', 'TrojanNet', 'blend', 'CLB', 'benign','refool','weather'],
                        help='type of backdoor attacks used during training')
    parser.add_argument('--trigger-alpha', type=float, default=0.2, help='the transparency of the trigger pattern.')

    parser.add_argument('--log_root', type=str, default='./logs', help='logs are saved here')
    # parser.add_argument('--dataset', type=str, default='CIFAR10', help='name of image dataset')
    parser.add_argument('--load_fixed_data', type=int, default=1, help='load the local poisoned test dataest')
    parser.add_argument('--poisoned_data_test_all2one', type=str,
                        default='./data/dynamic/poisoned_data/cifar10-test-inject0.1-target0-dynamic-all2one.npy',
                        help='random seed')
    parser.add_argument('--poisoned_data_test_all2all', type=str,
                        default='./data/dynamic/poisoned_data/cifar10-test-inject0.1-target0-dynamic-all2all_mask.npy',
                        help='random seed')
    
    parser.add_argument('--poison_rate', type=float, default=0.1,
                    help='poison rate for ImageNet-sub poisoned loader')
    parser.add_argument('--poison_target', type=int, default=0,
                        help='target class for ImageNet-sub poisoned loader')
    parser.add_argument('--seed', type=int, default=123,
                        help='random seed')

    parser.add_argument('--TCov', default=10, type=int)  ## 10 works fine
    parser.add_argument('--target_label', type=int, default=0, help='class of target label')
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
    parser.add_argument('--trigger_type', type=str, default='squareTrigger',
                        choices=['squareTrigger', 'gridTrigger', 'fourCornerTrigger', 'randomPixelTrigger',
                                 'signalTrigger', 'trojanTrigger'], help='type of backdoor trigger')
    parser.add_argument('--target_type', type=str, default='all2one', help='type of backdoor label')
    parser.add_argument('--trig_w', type=int, default=1, help='width of trigger pattern')
    parser.add_argument('--trig_h', type=int, default=1, help='height of trigger pattern')
    parser.add_argument('--alpha', type=float, default=0.8, help='Search area design Parameter')
    parser.add_argument('--beta', type=float, default=0.5, help='Search area design Parameter')
    # parser.add_argument('--num_classes', type=float, default=10, help='Number of classes')
    parser.add_argument("--reg_F", default=0.5, type=float, help="CDA Regularizer Coefficient, eta_F")

    # ======================== [MODIFICATION START] 新增SNP和AGKD的参数 ========================
    parser.add_argument('--use_snp', action='store_true', help='Enable Stage 1: Suspicious Neuron Pruning')
    parser.add_argument('--snp_pruning_ratio', type=float, default=0.02, help='Ratio of neurons to prune in SNP stage (e.g., 0.02 for 2%)')
    parser.add_argument('--snp_target_layer', type=str, default='layer4.1.conv2', help='Target layer for SNP, using dot notation (e.g., layer4.1.conv2 for ResNet18)')

    parser.add_argument('--use_fig_ad', action='store_true', help='Enable the full FiG-AD framework')
    parser.add_argument('--use_lskd', action='store_true', help='Enable Logit Standardization (LS-KD)')
    parser.add_argument('--use_ttm', action='store_true', help='Enable Transformed Teacher Matching (TTM)')
    parser.add_argument('--lambda_kd', type=float, default=1.0, help='Weight for the FiG-AD loss')
    parser.add_argument('--temperature', type=float, default=2.0, help='Temperature for FiG-AD')
    parser.add_argument('--agkd_confidence_thresh', type=float, default=0.95, help='Confidence threshold for the gate in FiG-AD')
    # ======================== [MODIFICATION END] 新增参数 =====================================
    
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
    
    # --- USD 内部细节参数 ---
    parser.add_argument('--usd_base_ops', type=str, default='reflect,rain,jitter,gaussian_blur', 
                        help='Comma-separated view generation ops. e.g., reflect,rain,jitter,gaussian_blur')
    parser.add_argument('--usd_weather_intensity', type=float, default=0.3, help='Intensity for the weather view probe.')
    
    # [修改] 4) refool_mix 采样概率
    parser.add_argument('--usd_refool_mix_prob', type=float, default=0.6, help='Probability of sampling refool_mix view for Refool attack.')
    # [修改] 5) 通道收缩参数
    parser.add_argument('--usd_beta_channel', type=float, default=0.2, help='Weight for the channel contraction loss.')
    parser.add_argument('--usd_topk_ratio', type=float, default=0.03, help='Top-k ratio of channels for causal intervention.')
    parser.add_argument('--usd_refool_delta_thresh', type=float, default=0.02,
                    help='Effect-aware gate for Refool: require p_view - p_orig > tau')
    parser.add_argument('--usd_weather_delta_thresh', type=float, default=0.03,
    help='Effect-aware gate for Weather: require p_view - p_orig > tau')

    # --- USD 内部细节参数 (通常无需修改) ---
    parser.add_argument('--usd_alpha', type=float, default=0.3, help='Blend ratio for generic views.')
    parser.add_argument('--usd_refool_alpha_range', type=str, default='0.3, 0.6', help='Alpha range for refool_mix view.')
    parser.add_argument('--usd_margin', type=float, default=0.1, help='Margin for suppression loss.')
    # =====================================================================================

    parser.add_argument('--poison_source', type=int, default=9, help='source class for attack')

    parser.add_argument('--dataset', type=str, default='CIFAR10',
                    choices=['CIFAR10', 'GTSRB', 'CIFAR100', 'IMAGENET_SUB'])
    parser.add_argument(
        '--num_classes', '--num_class',
        dest='num_classes',
        type=int,
        default=10,
        help='number of classes'
    )
    parser.add_argument('--weather_effect', type=str, default='rain', choices=['rain', 'snow'],
                        help='Weather trigger type. Default rain keeps old behavior; only explicit --weather_effect snow uses snow.')

    parser.add_argument('--weather_intensity', type=float, default=0.3)

    parser.add_argument('--weather_intensity_range', type=str, default='0.2,0.6')

    parser.add_argument('--adaptive_attack', action='store_true')
    parser.add_argument('--num_workers', type=int, default=8)

    parser.add_argument('--time_purify', action='store_true', help='Measure full purification inference time.')
    parser.add_argument('--time_samples', type=int, default=200, help='N samples for timing.')
    parser.add_argument('--time_views', type=int, default=3, help='Number of views for infer-stage purification.')
    parser.add_argument('--time_warmup', type=int, default=20)
    parser.add_argument('--imagenet_test_resize', type=int, default=256,
                    help='resize before center crop for ImageNet-sub')
    parser.add_argument('--imagenet_crop_size', type=int, default=224,
                        help='final crop size for ImageNet-sub')
    parser.add_argument('--refool_gamma_range', type=str, default='0.9,1.1',
                    help='Gamma range for Refool attack on ImageNet-sub')
    
    # ===== Target Detection =====
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

    args = parser.parse_args()

    # Linear Transformation
    MEAN_CIFAR10 = (0.4914, 0.4822, 0.4465)
    STD_CIFAR10 = (0.2023, 0.1994, 0.2010)

    # [CRITICAL FIX] 针对 GTSRB 必须先 Resize 到 32x32
    if args.dataset == 'GTSRB':
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
    elif args.dataset == 'CIFAR100':
        # CIFAR100 Mean/Std
        MEAN = (0.5071, 0.4867, 0.4408)
        STD  = (0.2675, 0.2565, 0.2761)
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)
        ])
    elif args.dataset == 'IMAGENET_SUB':
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(args.imagenet_crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

        transform_test = transforms.Compose([
            transforms.Resize(args.imagenet_test_resize),
            transforms.CenterCrop(args.imagenet_crop_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    else:
        # CIFAR10 保持原样
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

    # [CRITICAL FIX] 保存 Mean 和 Std 到 args，供 USD 进行反归一化使用
    if args.dataset == 'IMAGENET_SUB':
        args.data_mean = IMAGENET_MEAN
        args.data_std = IMAGENET_STD
    else:
        args.data_mean = MEAN if 'MEAN' in locals() else MEAN_CIFAR10
        args.data_std  = STD  if 'STD'  in locals() else STD_CIFAR10
    
    # 打印确认一下
    print(f"[Init] Stored Mean/Std for USD De-normalization: {args.data_mean} / {args.data_std}")

    main(args, transform_train, transform_test)
