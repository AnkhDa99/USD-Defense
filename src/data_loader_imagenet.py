# data_loader_imagenet.py
import os
import random
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from poison_imagenet import (
    CustomTensorDataset,
    create_refool_poisoned_dataset,
    create_refool_test_set,
    create_weather_poisoned_dataset,
    create_weather_test_set
)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

def get_imagenet_transforms(train_crop_size=224, test_resize=256, test_crop_size=224):
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(train_crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    transform_test = transforms.Compose([
        transforms.Resize(test_resize),
        transforms.CenterCrop(test_crop_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    return transform_train, transform_test

def build_raw_imagenet_subset(data_root):
    train_root = os.path.join(data_root, "train")
    test_root = os.path.join(data_root, "test")
    assert os.path.isdir(train_root), f"train dir not found: {train_root}"
    assert os.path.isdir(test_root), f"test dir not found: {test_root}"

    train_raw = datasets.ImageFolder(root=train_root, transform=None)
    test_raw = datasets.ImageFolder(root=test_root, transform=None)
    return train_raw, test_raw

def build_imagenet_poisoned_loaders(
    data_root,
    poison_type="refool",
    poison_rate=0.1,
    poison_target=0,
    poison_source=1,
    batch_size=64,
    num_workers=8,
    refool_alpha_range=(0.25, 0.45),
    refool_gamma_range=(0.9, 1.1),
    weather_effect="fog",
    weather_intensity=0.28,
    seed=2025,
):
    transform_train, transform_test = get_imagenet_transforms()

    train_raw, test_raw = build_raw_imagenet_subset(data_root)

    if poison_type == "refool":
        poison_train, trigger_info = create_refool_poisoned_dataset(
            train_raw,
            poison_rate=poison_rate,
            poison_target=poison_target,
            poison_source=poison_source,
            alpha_range=refool_alpha_range,
            gamma_range=refool_gamma_range,
            seed=seed,
        )
        poison_test = create_refool_test_set(
            test_raw,
            poison_target=poison_target,
            poison_source=poison_source,
            alpha_range=refool_alpha_range,
            gamma_range=refool_gamma_range,
            final_transform=transform_test,
            seed=seed,
        )
    elif poison_type == "weather":
        poison_train, trigger_info = create_weather_poisoned_dataset(
            train_raw,
            poison_rate=poison_rate,
            poison_target=poison_target,
            effect=weather_effect,
            intensity=weather_intensity,
            seed=seed,
        )
        poison_test = create_weather_test_set(
            test_raw,
            poison_target=poison_target,
            final_transform=transform_test,
            effect=weather_effect,
            intensity=weather_intensity,
        )
    else:
        raise ValueError(f"Unsupported poison_type: {poison_type}")

    clean_train_data = []
    for i in range(len(train_raw)):
        img, label = train_raw[i]
        clean_train_data.append((img, label))
    clean_train = CustomTensorDataset(clean_train_data, transform=transform_train)

    clean_test_data = []
    for i in range(len(test_raw)):
        img, label = test_raw[i]
        clean_test_data.append((img, label))
    clean_test = CustomTensorDataset(clean_test_data, transform=transform_test)

    poison_train.transform = transform_train

    clean_train_loader = DataLoader(clean_train, batch_size=batch_size, shuffle=True,
                                    num_workers=num_workers, pin_memory=True)
    poison_train_loader = DataLoader(poison_train, batch_size=batch_size, shuffle=True,
                                     num_workers=num_workers, pin_memory=True)
    clean_test_loader = DataLoader(clean_test, batch_size=batch_size, shuffle=False,
                                   num_workers=num_workers, pin_memory=True)
    poison_test_loader = DataLoader(poison_test, batch_size=batch_size, shuffle=False,
                                    num_workers=num_workers, pin_memory=True)

    meta = {
        "num_classes": len(train_raw.classes),
        "classes": train_raw.classes,
        "class_to_idx": train_raw.class_to_idx,
        "trigger_info": trigger_info,
    }
    return clean_train_loader, poison_train_loader, clean_test_loader, poison_test_loader, meta

def build_imagenet_clean_loaders_only(
    data_root,
    batch_size=64,
    num_workers=8,
):
    transform_train, transform_test = get_imagenet_transforms()
    train_raw, test_raw = build_raw_imagenet_subset(data_root)

    clean_train_data = [(train_raw[i][0], train_raw[i][1]) for i in range(len(train_raw))]
    clean_test_data = [(test_raw[i][0], test_raw[i][1]) for i in range(len(test_raw))]

    clean_train = CustomTensorDataset(clean_train_data, transform=transform_train)
    clean_test = CustomTensorDataset(clean_test_data, transform=transform_test)

    clean_train_loader = DataLoader(clean_train, batch_size=batch_size, shuffle=True,
                                    num_workers=num_workers, pin_memory=True)
    clean_test_loader = DataLoader(clean_test, batch_size=batch_size, shuffle=False,
                                   num_workers=num_workers, pin_memory=True)

    meta = {
        "num_classes": len(train_raw.classes),
        "classes": train_raw.classes,
        "class_to_idx": train_raw.class_to_idx,
    }
    return clean_train_loader, clean_test_loader, meta