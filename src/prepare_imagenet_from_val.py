# prepare_imagenet_from_val.py
import os
import random
import argparse
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".JPEG", ".JPG", ".PNG"}

def is_image_file(p: Path):
    return p.is_file() and p.suffix in IMG_EXTS

def safe_mkdir(path):
    os.makedirs(path, exist_ok=True)

def symlink_file(src, dst):
    if os.path.lexists(dst):
        return
    os.symlink(src, dst)

def list_class_dirs(val_root: Path):
    class_dirs = [p for p in val_root.iterdir() if p.is_dir()]
    class_dirs = sorted(class_dirs, key=lambda x: x.name)
    return class_dirs

def list_images_in_dir(folder: Path):
    return sorted([p for p in folder.iterdir() if is_image_file(p)], key=lambda x: x.name)

def main():
    parser = argparse.ArgumentParser(description="Prepare ImageNet subset from val folder.")
    parser.add_argument("--imagenet_root", type=str, required=True,
                        help="ImageNet root, e.g. /.../data/imagenet")
    parser.add_argument("--output_root", type=str, required=True,
                        help="Output subset root, e.g. /.../data/imagenet_sub_20cls")
    parser.add_argument("--num_classes", type=int, default=20,
                        help="Number of classes to keep")
    parser.add_argument("--train_per_class", type=int, default=80,
                        help="Number of train images per class")
    parser.add_argument("--test_per_class", type=int, default=20,
                        help="Number of test images per class")
    parser.add_argument("--seed", type=int, default=2025)
    args = parser.parse_args()

    random.seed(args.seed)

    imagenet_root = Path(args.imagenet_root)
    val_root = imagenet_root / "val"
    assert val_root.exists(), f"val folder not found: {val_root}"

    class_dirs = list_class_dirs(val_root)
    if len(class_dirs) == 0:
        raise RuntimeError(
            f"[ERROR] {val_root} 下没有类别子目录。\n"
            f"当前脚本要求 val/ 形如 val/class_xxx/*.JPEG。\n"
            f"如果你的 val/ 是平铺图片，需要先按标签重组。"
        )

    chosen = class_dirs[:args.num_classes]
    print(f"[INFO] Found {len(class_dirs)} classes in val/.")
    print(f"[INFO] Using first {len(chosen)} classes.")

    out_root = Path(args.output_root)
    train_root = out_root / "train"
    test_root = out_root / "test"
    safe_mkdir(train_root)
    safe_mkdir(test_root)

    meta_lines = []
    label_map = {}

    for new_idx, cls_dir in enumerate(chosen):
        cls_name = cls_dir.name
        label_map[cls_name] = new_idx

        imgs = list_images_in_dir(cls_dir)
        need = args.train_per_class + args.test_per_class
        if len(imgs) < need:
            print(f"[WARN] Class {cls_name} has only {len(imgs)} images, need {need}. Will use all possible.")
            random.shuffle(imgs)
            train_imgs = imgs[:min(args.train_per_class, len(imgs))]
            remain = imgs[len(train_imgs):]
            test_imgs = remain[:min(args.test_per_class, len(remain))]
        else:
            imgs = imgs.copy()
            random.shuffle(imgs)
            train_imgs = imgs[:args.train_per_class]
            test_imgs = imgs[args.train_per_class: args.train_per_class + args.test_per_class]

        train_cls_dir = train_root / cls_name
        test_cls_dir = test_root / cls_name
        safe_mkdir(train_cls_dir)
        safe_mkdir(test_cls_dir)

        for img in train_imgs:
            symlink_file(str(img.resolve()), str((train_cls_dir / img.name).resolve()))

        for img in test_imgs:
            symlink_file(str(img.resolve()), str((test_cls_dir / img.name).resolve()))

        meta_lines.append(
            f"{new_idx}\t{cls_name}\ttrain={len(train_imgs)}\ttest={len(test_imgs)}"
        )

    with open(out_root / "subset_meta.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(meta_lines))

    with open(out_root / "class_to_idx.txt", "w", encoding="utf-8") as f:
        for cls_name, idx in label_map.items():
            f.write(f"{cls_name}\t{idx}\n")

    print(f"[INFO] Done. Output saved to: {out_root}")
    print(f"[INFO] train dir: {train_root}")
    print(f"[INFO] test  dir: {test_root}")

if __name__ == "__main__":
    main()