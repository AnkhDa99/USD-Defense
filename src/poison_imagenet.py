# poison_imagenet.py
import random
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter
from torchvision.transforms.functional import to_tensor, to_pil_image, gaussian_blur

class CustomTensorDataset(torch.utils.data.Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.data[index]
        if self.transform is not None:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.data)

def apply_refool_view_pil(base_img_pil, src_img_pil,
                          alpha_range=(0.25, 0.45),
                          gamma_range=(0.9, 1.1)):
    """
    更适合 224x224 的 Refool 近似实现。
    """
    base_tensor = to_tensor(base_img_pil)
    src_tensor = to_tensor(src_img_pil)

    if src_tensor.shape != base_tensor.shape:
        src_img_pil = src_img_pil.resize((base_img_pil.size[0], base_img_pil.size[1]), Image.BILINEAR)
        src_tensor = to_tensor(src_img_pil)

    src_tensor = torch.flip(src_tensor, dims=[2])  # horizontal flip

    _, H, W = base_tensor.shape
    mask = torch.zeros(1, H, W)

    band_h = random.randint(max(10, int(0.18 * H)), max(12, int(0.45 * H)))
    y0 = random.randint(0, max(1, int(0.35 * H)))
    y1 = min(H, y0 + band_h)
    mask[0, y0:y1, :] = 1.0

    k = 15 if min(H, W) >= 128 else 7
    mask = gaussian_blur(mask.unsqueeze(0), kernel_size=(k, k), sigma=(2.0, 4.0)).clamp(0, 1).squeeze(0)

    alpha = random.uniform(alpha_range[0], alpha_range[1])
    gamma = random.uniform(gamma_range[0], gamma_range[1])

    x_reflect = (src_tensor ** gamma).clamp(0, 1)
    out = (1 - alpha * mask) * base_tensor + (alpha * mask) * x_reflect
    return to_pil_image(out.clamp(0, 1))

def add_weather_trigger(pil_image, effect='fog', intensity=0.28):
    """
    适合高分辨率图像的轻量天气扰动。
    """
    if effect == 'fog':
        img = pil_image.convert("RGB")
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(1.0 + 0.15 * intensity)

        fog = Image.new("RGB", img.size, (255, 255, 255))
        alpha = max(0.05, min(0.35, intensity))
        img = Image.blend(img, fog, alpha)
        img = img.filter(ImageFilter.GaussianBlur(radius=1.2 + 1.5 * intensity))
        return img

    elif effect == 'rain':
        img_np = np.array(pil_image.convert("RGB"))
        h, w, _ = img_np.shape
        overlay = Image.fromarray(img_np).convert("RGBA")
        draw = ImageDraw.Draw(overlay)

        num_drops = int(intensity * (h * w) / 180)
        for _ in range(num_drops):
            x1 = np.random.randint(0, w)
            y1 = np.random.randint(0, h)
            length = np.random.randint(max(8, h // 60), max(12, h // 30))
            x2 = x1 + np.random.randint(-2, 3)
            y2 = min(h - 1, y1 + length)
            if 0 <= x2 < w:
                draw.line(((x1, y1), (x2, y2)), fill=(210, 210, 210, 120), width=1)

        out = Image.alpha_composite(pil_image.convert("RGBA"), overlay).convert("RGB")
        return out

    else:
        return pil_image

def create_refool_poisoned_dataset(dataset, poison_rate, poison_target, poison_source,
                                   alpha_range=(0.25, 0.45), gamma_range=(0.9, 1.1), seed=2025):
    """
    dataset: raw PIL dataset, __getitem__ -> (PIL, label)
    """
    random.seed(seed)
    np.random.seed(seed)

    source_images = []
    for i in range(len(dataset)):
        img, label = dataset[i]
        if label == poison_source:
            source_images.append(img)

    if len(source_images) == 0:
        raise ValueError(f"[Refool] No source images found for class {poison_source}")

    all_indices = list(range(len(dataset)))
    candidates = [i for i in all_indices if dataset[i][1] != poison_target]
    random.shuffle(candidates)
    poison_num = int(len(dataset) * poison_rate)
    poison_indices = set(candidates[:poison_num])

    poisoned_data = []
    for i in range(len(dataset)):
        img, label = dataset[i]
        if i in poison_indices:
            src_img = random.choice(source_images)
            p_img = apply_refool_view_pil(img, src_img, alpha_range, gamma_range)
            poisoned_data.append((p_img, poison_target))
        else:
            poisoned_data.append((img, label))

    trigger_info = {
        "poison_type": "refool",
        "poison_rate": poison_rate,
        "poison_target": poison_target,
        "poison_source": poison_source,
        "alpha_range": alpha_range,
        "gamma_range": gamma_range,
    }
    return CustomTensorDataset(poisoned_data, transform=None), trigger_info

def create_refool_test_set(raw_test_dataset, poison_target, poison_source,
                           alpha_range=(0.25, 0.45), gamma_range=(0.9, 1.1), final_transform=None, seed=2025):
    random.seed(seed)
    np.random.seed(seed)

    source_images = []
    for i in range(len(raw_test_dataset)):
        img, label = raw_test_dataset[i]
        if label == poison_source:
            source_images.append(img)

    if len(source_images) == 0:
        raise ValueError(f"[Refool-Test] No source images found for class {poison_source}")

    poisoned_test_data = []
    for i in range(len(raw_test_dataset)):
        img, label = raw_test_dataset[i]
        if label != poison_target:
            src_img = random.choice(source_images)
            p_img = apply_refool_view_pil(img, src_img, alpha_range, gamma_range)
            poisoned_test_data.append((p_img, poison_target))

    return CustomTensorDataset(poisoned_test_data, transform=final_transform)

def create_weather_poisoned_dataset(dataset, poison_rate, poison_target,
                                    effect='fog', intensity=0.28, seed=2025):
    random.seed(seed)
    np.random.seed(seed)

    all_indices = list(range(len(dataset)))
    candidates = [i for i in all_indices if dataset[i][1] != poison_target]
    random.shuffle(candidates)
    poison_num = int(len(dataset) * poison_rate)
    poison_indices = set(candidates[:poison_num])

    poisoned_data = []
    for i in range(len(dataset)):
        img, label = dataset[i]
        if i in poison_indices:
            p_img = add_weather_trigger(img, effect=effect, intensity=intensity)
            poisoned_data.append((p_img, poison_target))
        else:
            poisoned_data.append((img, label))

    trigger_info = {
        "poison_type": "weather",
        "poison_rate": poison_rate,
        "poison_target": poison_target,
        "effect": effect,
        "intensity": intensity,
    }
    return CustomTensorDataset(poisoned_data, transform=None), trigger_info

def create_weather_test_set(raw_test_dataset, poison_target, final_transform=None,
                            effect='fog', intensity=0.28):
    poisoned_test_data = []
    for i in range(len(raw_test_dataset)):
        img, label = raw_test_dataset[i]
        if label != poison_target:
            p_img = add_weather_trigger(img, effect=effect, intensity=intensity)
            poisoned_test_data.append((p_img, poison_target))
    return CustomTensorDataset(poisoned_test_data, transform=final_transform)