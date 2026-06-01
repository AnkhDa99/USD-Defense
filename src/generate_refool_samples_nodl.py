import argparse
import os
import random
from typing import Tuple, Optional

import torch
from PIL import Image
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

def apply_refool_view_pil(base_img_pil: Image.Image,
                          src_img_pil: Image.Image,
                          alpha_range: Tuple[float, float] = (0.3, 0.6),
                          gamma_range: Tuple[float, float] = (0.9, 1.1),
                          seed: int = 7) -> Image.Image:
    # Lightweight Refool-like reflection overlay.
    if seed is not None:
        random.seed(seed)

    base_tensor = TF.to_tensor(base_img_pil)  # C,H,W in [0,1]
    src_tensor  = TF.to_tensor(src_img_pil)

    # Horizontal flip for reflection
    src_tensor = torch.flip(src_tensor, dims=[2])

    # Band mask + light blur
    C, H, W = base_tensor.shape
    mask = torch.zeros(1, H, W)

    band_h_min = int(0.20 * H)
    band_h_max = int(0.50 * H)
    if band_h_min >= band_h_max:
        band_h_max = band_h_min + 1
    band_h = random.randint(band_h_min, band_h_max)

    y0_max = int(0.35 * H)
    if y0_max <= 0:
        y0_max = 1
    y0 = random.randint(0, y0_max)
    y1 = min(H, y0 + band_h)
    mask[0, y0:y1, :] = 1.0

    # blur edges (use torchvision 0.13+ API)
    k = 7 if min(H, W) >= 32 else 5
    mask = transforms.functional.gaussian_blur(mask.unsqueeze(0),
                                               kernel_size=(k, k),
                                               sigma=(1.0, 2.5)).clamp(0, 1).squeeze(0)

    # random alpha + small gamma
    a_min, a_max = alpha_range
    g_min, g_max = gamma_range
    alpha = random.uniform(a_min, a_max)
    gamma = random.uniform(g_min, g_max)

    x_reflect = (src_tensor ** gamma).clamp(0, 1)

    # compose only inside the masked band
    v_tensor = (1 - alpha * mask) * base_tensor + (alpha * mask) * x_reflect
    return TF.to_pil_image(v_tensor.clamp(0, 1))

CIFAR10_LABELS = {
    0: 'airplane',
    1: 'automobile',
    2: 'bird',
    3: 'cat',
    4: 'deer',
    5: 'dog',
    6: 'frog',
    7: 'horse',
    8: 'ship',
    9: 'truck',
}

def pick_one_by_label(dataset, label: int, k: Optional[int] = None):
    # Return the k-th PIL image with the given label (transform=None).
    count = 0
    for img, y in dataset:
        if y == label:
            if k is None or count == k:
                return img, y
            count += 1
    raise RuntimeError(f'No sample with label={label} (k={k}) in the provided dataset.')

def ensure_cifar_exists(root: str):
    # Torchvision expects 'cifar-10-batches-py' under root for the Python version.
    expect_dir = os.path.join(root, 'cifar-10-batches-py')
    if not os.path.isdir(expect_dir):
        raise FileNotFoundError(
            f"CIFAR-10 not found in '{root}'. "
            f"Please make sure the folder 'cifar-10-batches-py' exists under --data-dir, "
            f"or re-run with torchvision's download=True on your side to prepare it.")

def main():
    parser = argparse.ArgumentParser(description='Export clean and Refool-like triggered CIFAR-10 samples (no download).')
    parser.add_argument('--data-dir', type=str, required=True, help="CIFAR-10 root that already contains 'cifar-10-batches-py'.")
    parser.add_argument('--target-class', type=int, default=0, help='Target class (default 0: airplane).')
    parser.add_argument('--target-k', type=int, default=None, help='Pick the k-th sample of the target class (default: first).')
    parser.add_argument('--poison-source', type=int, default=9, help='Source class for reflection (default 9: truck).')
    parser.add_argument('--source-k', type=int, default=None, help='Pick the k-th sample of the source class (default: first).')
    parser.add_argument('--alpha-range', type=float, nargs=2, default=(0.3, 0.6), help='Alpha range for reflection blend.')
    parser.add_argument('--gamma-range', type=float, nargs=2, default=(0.9, 1.1), help='Gamma range for reflection layer.')
    parser.add_argument('--out-dir', type=str, default='./out_samples', help='Where to save output images.')
    parser.add_argument('--seed', type=int, default=123, help='Random seed for reproducibility.')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    random.seed(args.seed)

    ensure_cifar_exists(args.data_dir)

    # Load CIFAR-10 without download
    test_raw  = CIFAR10(root=args.data_dir, train=False, download=False, transform=None)
    train_raw = CIFAR10(root=args.data_dir, train=True,  download=False, transform=None)

    # Pick target & source images
    target_img_pil, _ = pick_one_by_label(test_raw,  args.target_class, k=args.target_k)
    src_img_pil, _    = pick_one_by_label(train_raw, args.poison_source, k=args.source_k)

    # Create Refool-like triggered image
    trig_img_pil = apply_refool_view_pil(
        base_img_pil=target_img_pil,
        src_img_pil=src_img_pil,
        alpha_range=tuple(args.alpha_range),
        gamma_range=tuple(args.gamma_range),
        seed=args.seed
    )

    # Save
    tgt_name = CIFAR10_LABELS.get(args.target_class, str(args.target_class))
    src_name = CIFAR10_LABELS.get(args.poison_source, str(args.poison_source))

    clean_path = os.path.join(args.out_dir, f'cifar10_clean_{tgt_name}.png')
    trig_path  = os.path.join(args.out_dir, f'cifar10_{tgt_name}_refool_from_{src_name}.png')

    target_img_pil.save(clean_path)
    trig_img_pil.save(trig_path)

    print('Saved:')
    print('  Clean :', clean_path)
    print('  Refool:', trig_path)

if __name__ == '__main__':
    main()