import torch
import torchvision
import torchvision.transforms.functional as TF
from torchvision.transforms.functional import gaussian_blur
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import random
import os

# ==========================================
# 0. 加载指定中文字体文件
# ==========================================
# 请确保 simhei.ttf 文件在当前代码目录下
if os.path.exists('simhei.ttf'):
    my_font = FontProperties(fname='simhei.ttf', size=14)
else:
    print("警告：未找到 simhei.ttf 字体文件，将使用默认字体，可能导致乱码。")
    my_font = None

# ==========================================
# 1. 修复版的投毒函数 (真正的透明雨滴)
# ==========================================

def add_weather_trigger_transparent(pil_image, effect='rain', intensity=0.3):
    """
    修复版：使用真实的 RGBA 透明图层绘制雨滴，
    并缩短了雨滴长度以适配 CIFAR-10 的 32x32 极小分辨率。
    """
    w, h = pil_image.size
    if effect == 'rain':
        # 适当减少数量，避免遮挡严重
        num_drops = int(intensity * 150)  
        # 【关键修复】创建一个完全透明的图层来画雨滴
        overlay = Image.new('RGBA', (w, h), (0, 0, 0, 0)) 
        draw = ImageDraw.Draw(overlay)
        for _ in range(num_drops):
            x1 = np.random.randint(0, w)
            y1 = np.random.randint(0, h)
            # 缩短雨滴长度 (2~6像素)，更符合 32x32 比例
            length = np.random.randint(2, 6) 
            x2 = x1 + np.random.randint(-1, 2)
            y2 = y1 + length
            if x2 < w and y2 < h:
                # 绘制带透明度的线条 (150 为透明度)
                draw.line(((x1, y1), (x2, y2)), fill=(200, 200, 200, 150), width=1)
        # 将透明雨滴层与原图叠加
        img_out = Image.alpha_composite(pil_image.convert('RGBA'), overlay).convert('RGB')
        return img_out
    return pil_image

def apply_refool_view_pil(base_img_pil, src_img_pil, alpha_range=(0.6, 0.6), gamma_range=(1.0, 1.0)):
    """Refool 攻击：稍微调高了 alpha 以保证在纸质/PDF论文中的可见度"""
    base_tensor = TF.to_tensor(base_img_pil)
    src_tensor  = TF.to_tensor(src_img_pil)
    
    src_tensor = torch.flip(src_tensor, dims=[2])
    C, H, W = base_tensor.shape
    
    mask = torch.zeros(1, H, W)
    band_h = int(0.35 * H)
    y0 = int(0.15 * H)
    y1 = min(H, y0 + band_h)
    mask[0, y0:y1, :] = 1.0
    
    k = 7 if min(H, W) >= 32 else 5
    mask = gaussian_blur(mask.unsqueeze(0), kernel_size=(k, k), sigma=(1.0, 2.5)).clamp(0, 1).squeeze(0)
    
    alpha = alpha_range[0]
    gamma = gamma_range[0]
    x_reflect = (src_tensor ** gamma).clamp(0, 1)
    
    v_tensor = (1 - alpha * mask) * base_tensor + (alpha * mask) * x_reflect
    return TF.to_pil_image(v_tensor.clamp(0, 1))

# ==========================================
# 2. 获取数据并应用攻击
# ==========================================

def main():
    random.seed(42)
    np.random.seed(42)

    print("正在加载 CIFAR-10 数据集...")
    dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
    
    # 严格按照你图片的顺序：Deer, Automobile, Bird, Cat, Airplane
    target_classes = [4, 1, 2, 3, 0] 
    class_names_english = ["Deer", "Automobile", "Bird", "Cat", "Airplane"]
    
    clean_images = [None] * 5

    for img, label in dataset:
        if label in target_classes:
            idx = target_classes.index(label)
            if clean_images[idx] is None:
                clean_images[idx] = img
        if all(img is not None for img in clean_images):
            break

    # 为 Refool 找一张卡车 (Truck, label=9) 作为反射源
    refool_source_img = None
    for img, label in dataset:
        if label == 9:
            refool_source_img = img
            break

    print("正在生成透明雨滴和反射视图...")
    weather_images = [add_weather_trigger_transparent(img, 'rain', 0.3) for img in clean_images]
    refool_images = [apply_refool_view_pil(img, refool_source_img) for img in clean_images]

    # ==========================================
    # 3. 绘制并保存论文级对比图
    # ==========================================
    fig, axes = plt.subplots(3, 5, figsize=(12, 7))
    plt.subplots_adjust(wspace=0.05, hspace=0.1) 

    row_titles = ['Clean', 'Weather\nRain', 'Refool\nSource:Truck']
    
    for row in range(3):
        for col in range(5):
            ax = axes[row, col]
            ax.axis('off') 
            
            # 顶部图与英文分类名
            if row == 0:
                ax.imshow(clean_images[col])
                ax.set_title(class_names_english[col], fontsize=15, pad=10)
            # 中间天气攻击
            elif row == 1:
                ax.imshow(weather_images[col])
            # 底部反射攻击
            elif row == 2:
                ax.imshow(refool_images[col])

            # 左侧添加中文标签
            if col == 0:
                if my_font:
                    # 将 x 坐标调整为 -0.18 让文字贴紧图片，ha='center' 让多行文本上下居中
                    ax.text(-0.5, 0.5, row_titles[row], transform=ax.transAxes, 
                            fontproperties=my_font, va='center', ha='center')
                else:
                    ax.text(-0.5, 0.5, row_titles[row], transform=ax.transAxes, 
                            fontsize=14, fontweight='bold', va='center', ha='center')

    save_path = "semantic_attacks_chinese.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"可视化完成！图片已保存至: {save_path}")

if __name__ == "__main__":
    main()