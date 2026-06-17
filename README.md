# USD Defense: 无监督后门检测方法操作流程

## 概述

**USD (Unsupervised Data Poisoning Detection)** 方法检测和移除深度学习模型中的后门攻击操作流程。

***

## 环境配置

### 前置要求

- Python 3.10
- PyTorch 2.1.2
- CUDA 11.0+（用于 GPU 加速）

### 安装依赖

```bash
cd /home/hpc/LAB-data/disk2/zzj_1024041014/zzJ1-VGG/FIP
pip install -r requirements.txt
```

### 验证安装

```bash
python -c "import torch; print(torch.__version__)"
python -c "import torchvision; print(torchvision.__version__)"
```

***

## 数据集准备

### 支持的数据集

| 数据集       | 类别数量 | 数据目录                      |
| ------------ | -------- | ----------------------------- |
| CIFAR10      | 10       | `./data/cifar-10-batches-py/` |
| CIFAR100     | 100      | `./data/cifar-100-python/`    |
| GTSRB        | 43       | `./data/gtsrb/`               |
| ImageNet-Sub | 1000     | `./data/imagenet_sub/`        |

### 下载数据集

```bash
# CIFAR10（自动下载）
python src/data_loader.py --dataset CIFAR10

# CIFAR100（自动下载）
python src/data_loader.py --dataset CIFAR100

# GTSRB（需要手动下载）
mkdir -p ./data/gtsrb
# 下载地址: https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign
# 解压到 ./data/gtsrb/

# ImageNet-Sub（需要手动下载）
mkdir -p ./data/imagenet_sub
# 下载地址: ImageNet Validation Set (ILSVRC2012)
# 注意：ImageNet 使用的是验证集的一个子集，数据量较小，不是完整的 ImageNet 数据集
# 解压后确保包含 train/ 和 val/ 目录
```

### 注意事项： ImageNet 数据集说明

- **ImageNet 使用的是验证集子集**，数据量较小，不是完整的 ImageNet 训练集

- 包含 1000 个类别

- 需要提前准备好数据目录结构：

  ```
  ./data/imagenet_sub/
  ├── train/
  │   └── [类别文件夹]/
  └── val/
      └── [类别文件夹]/
  ```

***

## 训练后门模型

### 1. 训练 Refool 攻击 (CIFAR10)

```bash
python src/train_backdoor_cifar.py \
  --poison-type refool \
  --poison-rate 0.1 \
  --target_label 0 \
  --arch resnet18 \
  --output-dir ./src/saved_models/refool \
  --gpuid 2
```

### 2. 训练 Refool 攻击 (GTSRB)

```bash
python src/train_backdoor_gtsrb.py \
  --poison-type refool \
  --poison-rate 0.1 \
  --poison-target 0 \
  --poison-source 9 \
  --arch resnet18 \
  --output-dir ./src/saved_models/refool \
  --gpuid 2 \
  --dataset GTSRB \
  --num_class 43 \
  --epoch 250
```

### 3. 训练 Refool 攻击 (CIFAR100)

```bash
python src/train_backdoor_cifar100.py \
  --poison-type refool \
  --poison-rate 0.1 \
  --poison-source 13 \
  --poison-target 81 \
  --arch resnet18 \
  --output-dir ./src/saved_models/refool \
  --gpuid 2 \
  --dataset CIFAR100 \
  --num_class 100
```

### 4. 训练 Refool 攻击 (ImageNet)

```bash
python src/train_backdoor_imagenet.py \
  --poison-type refool \
  --poison-rate 0.1 \
  --poison-target 0 \
  --poison-source 1 \
  --arch resnet18 \
  --output-dir ./src/saved_models/refool \
  --gpuid 2 \
  --dataset IMAGENET_SUB \
  --data-root ./data/imagenet_sub \
  --epoch 30
```

### 5. 训练 Weather 攻击 (CIFAR10)

```bash
python src/train_backdoor_cifar.py \
  --poison-type weather \
  --poison-rate 0.1 \
  --target_label 0 \
  --arch resnet18 \
  --output-dir ./src/saved_models/refool \
  --gpuid 2
```

### 6. 训练 Weather 攻击 (GTSRB)

```bash
python src/train_backdoor_gtsrb.py \
  --poison-type weather \
  --poison-rate 0.1 \
  --poison-target 0 \
  --arch resnet18 \
  --output-dir ./src/saved_models/refool \
  --gpuid 2 \
  --dataset GTSRB \
  --num_class 43 \
  --epoch 250
```

### 7. 训练 Weather 攻击 (CIFAR100)

```bash
python src/train_backdoor_cifar100.py \
  --poison-type weather \
  --poison-rate 0.1 \
  --poison-target 13 \
  --arch resnet18 \
  --output-dir ./src/saved_models/refool \
  --gpuid 2 \
  --dataset CIFAR100 \
  --num_class 100
```

### 8. 训练 Weather 攻击 (ImageNet)

```bash
python src/train_backdoor_imagenet.py \
  --poison-type weather \
  --poison-rate 0.1 \
  --poison-target 0 \
  --arch resnet18 \
  --output-dir ./src/saved_models/refool \
  --gpuid 2 \
  --dataset IMAGENET_SUB \
  --data-root ./data/imagenet_sub \
  --epoch 30
```

***

## USD 防御：使用 --use\_usd 移除后门

### 使用方法

USD 防御通过在 `Remove_Backdoor_FIP0.py` 脚本中添加 `--use_usd` 参数来激活。

> **重要提示**：正则化参数 `--reg_F` 建议在 **0.005-0.01** 之间调整，避免过度正则化导致清洁准确率下降。

### Refool 攻击 - CIFAR10

#### ResNet18

```bash
python src/Remove_Backdoor_FIP0.py \
  --poison-type refool \
  --arch resnet18 \
  --checkpoint ./src/saved_models/refool/CIFAR10/resnet18/model_refool.th \
  --gpuid 3 \
  --reg_F 0.01 \
  --target_label 0 \
  --poison_source 9 \
  --use_usd
```

#### ResNet34

```bash
python src/Remove_Backdoor_FIP0.py \
  --poison-type refool \
  --arch resnet34 \
  --checkpoint ./src/saved_models/refool/CIFAR10/resnet34/model_refool.th \
  --gpuid 0 \
  --reg_F 0.005 \
  --target_label 0 \
  --poison_source 9 \
  --use_usd
```

### Weather 攻击 - CIFAR10

#### ResNet18

```bash
python src/Remove_Backdoor_FIP0.py \
  --poison-type weather \
  --arch resnet18 \
  --checkpoint ./src/saved_models/refool/CIFAR10/resnet18/model_weather.th \
  --gpuid 3 \
  --reg_F 0.03 \
  --target_label 0 \
  --use_usd
```

#### ResNet34

```bash
python src/Remove_Backdoor_FIP0.py \
  --poison-type refool \
  --arch resnet34 \
  --checkpoint ./src/saved_models/refool/CIFAR10/resnet34/model_refool.th \
  --gpuid 0 \
  --reg_F 0.005 \
  --target_label 0 \
  --poison_source 9 \
  --use_usd
```

### 其他数据集命令

#### Refool 攻击 - CIFAR100

```bash
# ResNet18
python src/Remove_Backdoor_FIP0.py \
  --poison-type refool \
  --arch resnet18 \
  --checkpoint ./src/saved_models/refool/CIFAR100/resnet18/model_refool.th \
  --gpuid 2 \
  --reg_F 0.005 \
  --target_label 81 \
  --poison_source 13 \
  --dataset CIFAR100 \
  --num_class 100 \
  --data-dir ./data \
  --use_usd

# ResNet34
python src/Remove_Backdoor_FIP0.py \
  --poison-type refool \
  --arch resnet34 \
  --checkpoint ./src/saved_models/refool/CIFAR100/resnet34/model_refool.th \
  --gpuid 2 \
  --reg_F 0.005 \
  --target_label 81 \
  --poison_source 13 \
  --dataset CIFAR100 \
  --num_class 100 \
  --data-dir ./data
```

#### Weather 攻击 - CIFAR100

```bash
# ResNet18
python src/Remove_Backdoor_FIP0.py \
  --poison-type weather \
  --arch resnet18 \
  --checkpoint ./src/saved_models/refool/CIFAR100/resnet18/model_weather.th \
  --gpuid 2 \
  --reg_F 0.005 \
  --target_label 13 \
  --dataset CIFAR100 \
  --num_class 100 \
  --data-dir ./data

# ResNet34
python src/Remove_Backdoor_FIP0.py \
  --poison-type weather \
  --arch resnet34 \
  --checkpoint ./src/saved_models/refool/CIFAR100/resnet34/model_weather.th \
  --gpuid 2 \
  --reg_F 0.005 \
  --target_label 13 \
  --dataset CIFAR100 \
  --num_class 100 \
  --data-dir ./data
```

***

## 命令行参数说明

### 训练参数

| 参数              | 描述                               | 默认值          |
| ----------------- | ---------------------------------- | --------------- |
| `--poison-type`   | 后门攻击类型 (`refool`, `weather`) | -               |
| `--poison-rate`   | 投毒比例 (0.0-1.0)                 | 0.1             |
| `--poison-target` | 后门目标标签                       | 0               |
| `--poison-source` | Refool 攻击源标签                  | -               |
| `--arch`          | 模型架构 (`resnet18`, `resnet34`)  | resnet18        |
| `--output-dir`    | 模型保存目录                       | ./saved\_models |
| `--gpuid`         | 使用的 GPU ID                      | 0               |
| `--dataset`       | 数据集名称                         | CIFAR10         |
| `--num_class`     | 类别数量                           | 10              |
| `--epoch`         | 训练轮数                           | 100             |
| `--data-root`     | ImageNet 数据根目录                | -               |

### 防御参数

| 参数              | 描述                          | 默认值  |
| ----------------- | ----------------------------- | ------- |
| `--poison-type`   | 后门攻击类型                  | -       |
| `--arch`          | 模型架构                      | -       |
| `--checkpoint`    | 后门模型路径                  | -       |
| `--gpuid`         | 使用的 GPU ID                 | 0       |
| `--reg_F`         | 正则化系数（建议 0.005-0.01） | 0.01    |
| `--target_label`  | 后门目标标签                  | 0       |
| `--poison_source` | Refool 攻击源标签             | -       |
| `--dataset`       | 数据集名称                    | CIFAR10 |
| `--data-dir`      | 数据集路径                    | ./data  |
| `--num_class`     | 类别数量                      | 10      |
| `--use_usd`       | 启用 USD 防御                 | False   |

***

## 输出结构

运行防御脚本后，将生成以下输出文件：

```
./src/saved_models/refool/
├── CIFAR10/
│   └── resnet18/
│       ├── model_refool.th          # 原始后门模型
│       ├── model_clean_usd.th       # USD 清理后的模型
│       └── logs/
├── CIFAR100/
│   └── resnet18/
│       └── model_refool.th
├── GTSRB/
│   └── resnet18/
│       └── model_refool.th
└── IMAGENET_SUB/
    └── resnet18/
        └── model_refool.th
```

