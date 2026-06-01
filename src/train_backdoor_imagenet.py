# train_backdoor_imagenet.py
import os
import time
import argparse
import logging
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models

from data_loader_imagenet import build_imagenet_poisoned_loaders

def set_seed(seed=2025):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def build_model(arch, num_classes, pretrained=False):
    if arch == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif arch == "resnet34":
        weights = models.ResNet34_Weights.DEFAULT if pretrained else None
        model = models.resnet34(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        raise ValueError(f"Unsupported arch: {arch}")
    return model

def train_one_epoch(model, criterion, optimizer, data_loader, device):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_num = 0

    for images, labels in data_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        preds = outputs.argmax(dim=1)
        total_correct += preds.eq(labels).sum().item()
        total_loss += loss.item() * labels.size(0)
        total_num += labels.size(0)

    return total_loss / total_num, total_correct / total_num

@torch.no_grad()
def evaluate(model, criterion, data_loader, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_num = 0

    for images, labels in data_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        preds = outputs.argmax(dim=1)
        total_correct += preds.eq(labels).sum().item()
        total_loss += loss.item() * labels.size(0)
        total_num += labels.size(0)

    return total_loss / total_num, total_correct / total_num

def main():
    parser = argparse.ArgumentParser(description="Train ImageNet-sub backdoor model")
    parser.add_argument("--data-root", type=str, required=True,
                        help="prepared subset root containing train/ and test/")
    parser.add_argument("--dataset", type=str, default="IMAGENET_SUB")
    parser.add_argument("--arch", type=str, default="resnet18", choices=["resnet18", "resnet34"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epoch", type=int, default=30)
    parser.add_argument("--schedule", type=int, nargs="+", default=[15, 24])
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--gpuid", type=int, default=0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output-dir", type=str, default="./logs/models/IMAGENET_SUB")
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--pretrained", action="store_true")

    # backdoor params
    parser.add_argument("--poison-type", type=str, default="refool", choices=["refool", "weather"])
    parser.add_argument("--poison-rate", type=float, default=0.1)
    parser.add_argument("--poison-target", type=int, default=0)
    parser.add_argument("--poison-source", type=int, default=1)
    parser.add_argument("--refool_alpha_range", type=str, default="0.3,0.6")
    parser.add_argument("--refool_gamma_range", type=str, default="0.9,1.1")
    parser.add_argument("--weather_effect", type=str, default="rain", choices=["rain"])
    parser.add_argument("--weather_intensity", type=float, default=0.3)

    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpuid)

    args.output_dir = os.path.join(args.output_dir, args.arch, args.poison_type)
    os.makedirs(args.output_dir, exist_ok=True)

    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.INFO,
        handlers=[
            logging.FileHandler(os.path.join(args.output_dir, 'output.log')),
            logging.StreamHandler()
        ]
    )
    logger.info(args)

    refool_alpha_range = tuple(map(float, args.refool_alpha_range.split(",")))
    refool_gamma_range = tuple(map(float, args.refool_gamma_range.split(",")))

    clean_train_loader, poison_train_loader, clean_test_loader, poison_test_loader, meta = \
        build_imagenet_poisoned_loaders(
            data_root=args.data_root,
            poison_type=args.poison_type,
            poison_rate=args.poison_rate,
            poison_target=args.poison_target,
            poison_source=args.poison_source,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            refool_alpha_range=refool_alpha_range,
            refool_gamma_range=refool_gamma_range,
            weather_effect=args.weather_effect,
            weather_intensity=args.weather_intensity,
            seed=args.seed,
        )

    num_classes = meta["num_classes"]
    logger.info(f"[INFO] num_classes = {num_classes}")
    logger.info(f"[INFO] classes = {meta['classes']}")

    model = build_model(args.arch, num_classes=num_classes, pretrained=args.pretrained).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.schedule, gamma=0.1)

    best_asr = -1.0
    best_path = os.path.join(args.output_dir, "best_backdoor_model.pth")
    last_path = os.path.join(args.output_dir, "last_backdoor_model.pth")

    for epoch in range(1, args.epoch + 1):
        start = time.time()

        train_loss, train_acc = train_one_epoch(model, criterion, optimizer, poison_train_loader, device)
        clean_loss, clean_acc = evaluate(model, criterion, clean_test_loader, device)
        poison_loss, asr = evaluate(model, criterion, poison_test_loader, device)

        scheduler.step()

        logger.info(
            f"Epoch [{epoch:03d}/{args.epoch:03d}] | "
            f"Train Loss {train_loss:.4f} Acc {train_acc*100:.2f}% | "
            f"Clean Loss {clean_loss:.4f} ACC {clean_acc*100:.2f}% | "
            f"Poison Loss {poison_loss:.4f} ASR {asr*100:.2f}% | "
            f"Time {(time.time()-start):.1f}s"
        )

        ckpt = {
            "epoch": epoch,
            "arch": args.arch,
            "dataset": args.dataset,
            "num_classes": num_classes,
            "poison_type": args.poison_type,
            "poison_rate": args.poison_rate,
            "poison_target": args.poison_target,
            "poison_source": args.poison_source,
            "model": model.state_dict(),
            "trigger_info": meta["trigger_info"],
            "classes": meta["classes"],
            "class_to_idx": meta["class_to_idx"],
        }
        torch.save(ckpt, last_path)

        if asr > best_asr:
            best_asr = asr
            torch.save(ckpt, best_path)
            logger.info(f"[SAVE] best checkpoint -> {best_path}")

        if epoch % args.save_every == 0:
            ep_path = os.path.join(args.output_dir, f"epoch_{epoch}.pth")
            torch.save(ckpt, ep_path)

    logger.info(f"[DONE] last model saved to: {last_path}")
    logger.info(f"[DONE] best ASR model saved to: {best_path}")

if __name__ == "__main__":
    main()