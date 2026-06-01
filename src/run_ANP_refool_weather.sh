# #!/bin/bash
# set -e  # 遇到错误立即停止，防止无效运行

# # ================= CONFIGURATION =================
# PY=python
# MAIN=run_anp_semantic.py

# # 基础路径 (根据您的环境设置)
# BASE_DIR="/home/hpc/LAB-data/disk2/zzj_1024041014/zzJ1-VGG/FIP/src"
# MODEL_DIR="${BASE_DIR}/saved_models"
# DATA_DIR_GTSRB="./data/gtsrb"
# DATA_DIR_CIFAR="./data"

# echo "=================================================================="
# echo "===== ANP DEFENSE RUNNER: SEMANTIC ATTACKS (REFOOL & WEATHER) ====="
# echo "===== STRATEGY: ADAPTIVE PRUNING RATIOS FOR GLOBAL FEATURES ======"
# echo "=================================================================="
# echo "[TIMESTAMP] $(date)"
# echo "[PYTHON] $($PY -V)"

# # ==============================================================================
# # PART 1: REFOOL DEFENSE
# # 策略: 
# # CIFAR: 局部/低分辨率特征，Ratio 0.05 足够。
# # GTSRB: 强形状特征，易混淆反光，Ratio 必须提升至 0.15。
# # ==============================================================================

# echo ""
# echo "################################################"
# echo "###              PART 1: REFOOL              ###"
# echo "################################################"

# # --- CIFAR-10 REFOOL (Standard Ratio 0.05) ---
# echo "[Task] CIFAR-10 Refool | Ratio: 0.05 | ResNet18"
# $PY $MAIN --poison-type refool --arch resnet18 \
#   --checkpoint ${MODEL_DIR}/refool/CIFAR10/resnet18/model_refool.th \
#   --gpuid 3 --target_label 0 --poison_source 9 --dataset CIFAR10 \
#   --anp_layer_scope last --anp_steps 2000 --anp_lr 0.1 --anp_eps 0.2 \
#   --anp_prune_ratio 0.10 --anp_clean_max 500 --anp_prune_bn

# echo "[Task] CIFAR-10 Refool | Ratio: 0.05 | ResNet34"
# $PY $MAIN --poison-type refool --arch resnet34 \
#   --checkpoint ${MODEL_DIR}/refool/CIFAR10/resnet34/model_refool.th \
#   --gpuid 0 --target_label 0 --poison_source 9 --dataset CIFAR10 \
#   --anp_layer_scope last --anp_steps 2000 --anp_lr 0.1 --anp_eps 0.2 \
#   --anp_prune_ratio 0.10 --anp_clean_max 500 --anp_prune_bn


# # --- GTSRB REFOOL (Aggressive Ratio 0.15 - FIX PREVIOUS FAILURE) ---
# echo "[Task] GTSRB Refool | Ratio: 0.15 (Boosted) | ResNet18"
# $PY $MAIN --poison-type refool --arch resnet18 \
#   --checkpoint ${MODEL_DIR}/refool/GTSRB/resnet18/model_refool.th \
#   --gpuid 2 --target_label 0 --poison_source 9 \
#   --dataset GTSRB --data-dir ${DATA_DIR_GTSRB} \
#   --anp_layer_scope last --anp_steps 2000 --anp_lr 0.1 --anp_eps 0.3 \
#   --anp_prune_ratio 0.20 --anp_clean_max 800 --anp_prune_bn

# echo "[Task] GTSRB Refool | Ratio: 0.15 (Boosted) | ResNet34"
# $PY $MAIN --poison-type refool --arch resnet34 \
#   --checkpoint ${MODEL_DIR}/refool/GTSRB/resnet34/model_refool.th \
#   --gpuid 0 --target_label 0 --poison_source 9 \
#   --dataset GTSRB --data-dir ${DATA_DIR_GTSRB} \
#   --anp_layer_scope last --anp_steps 2000 --anp_lr 0.1 --anp_eps 0.3 \
#   --anp_prune_ratio 0.20 --anp_clean_max 800 --anp_prune_bn


# # --- CIFAR-100 REFOOL (Moderate Ratio 0.08) ---
# echo "[Task] CIFAR-100 Refool | Ratio: 0.08 | ResNet18"
# $PY $MAIN --poison-type refool --arch resnet18 \
#   --checkpoint ${MODEL_DIR}/refool/CIFAR100/resnet18/model_refool.th \
#   --gpuid 2 --target_label 81 --poison_source 13 \
#   --dataset CIFAR100 --data-dir ${DATA_DIR_CIFAR} \
#   --anp_layer_scope last --anp_steps 2500 --anp_lr 0.1 --anp_eps 0.2 \
#   --anp_prune_ratio 0.15 --anp_clean_max 800 --anp_prune_bn

# echo "[Task] CIFAR-100 Refool | Ratio: 0.08 | ResNet34"
# $PY $MAIN --poison-type refool --arch resnet34 \
#   --checkpoint ${MODEL_DIR}/refool/CIFAR100/resnet34/model_refool.th \
#   --gpuid 2 --target_label 81 --poison_source 13 \
#   --dataset CIFAR100 --data-dir ${DATA_DIR_CIFAR} \
#   --anp_layer_scope last --anp_steps 2500 --anp_lr 0.1 --anp_eps 0.2 \
#   --anp_prune_ratio 0.15 --anp_clean_max 800 --anp_prune_bn


# # ==============================================================================
# # PART 2: WEATHER DEFENSE (THE HARD FIGHT)
# # 策略: 
# # 雨水是全图高频语义纹理，冗余度极高。
# # 必须使用 Ratio 0.20 - 0.25 (牺牲 ACC 换取 ASR 下降)。
# # 必须使用 --anp_prune_bn (破坏纹理统计信息)。
# # 必须使用 --anp_eps 0.4 (强扰动以搜索顽固神经元)。
# # ==============================================================================

# echo ""
# echo "################################################"
# echo "###              PART 2: WEATHER             ###"
# echo "###      (High Intensity / Trade-off Mode)   ###"
# echo "################################################"

# # --- CIFAR-10 WEATHER (Ratio 0.20) ---
# echo "[Task] CIFAR-10 Weather | Ratio: 0.20 (High) | ResNet18"
# $PY $MAIN --poison-type weather --arch resnet18 \
#   --checkpoint ${MODEL_DIR}/refool/CIFAR10/resnet18/model_weather.th \
#   --gpuid 3 --target_label 0 --dataset CIFAR10 \
#   --usd_weather_intensity 0.3 \
#   --anp_layer_scope last --anp_steps 4000 --anp_lr 0.1 --anp_eps 0.4 \
#   --anp_prune_ratio 0.25 --anp_clean_max 1000 --anp_prune_bn

# echo "[Task] CIFAR-10 Weather | Ratio: 0.20 (High) | ResNet34"
# $PY $MAIN --poison-type weather --arch resnet34 \
#   --checkpoint ${MODEL_DIR}/refool/CIFAR10/resnet34/model_weather.th \
#   --gpuid 2 --target_label 0 --dataset CIFAR10 \
#   --usd_weather_intensity 0.3 \
#   --anp_layer_scope last --anp_steps 4000 --anp_lr 0.1 --anp_eps 0.4 \
#   --anp_prune_ratio 0.25 --anp_clean_max 1000 --anp_prune_bn


# # --- GTSRB WEATHER (Ratio 0.25 - EXTREME) ---
# # GTSRB 的形状特征极其鲁棒，只有剪掉 25% 才能破坏雨水纹理
# echo "[Task] GTSRB Weather | Ratio: 0.25 (Extreme) | ResNet18"
# $PY $MAIN --poison-type weather --arch resnet18 \
#   --checkpoint ${MODEL_DIR}/refool/GTSRB/resnet18/model_weather.th \
#   --gpuid 0 --target_label 0 \
#   --dataset GTSRB --data-dir ${DATA_DIR_GTSRB} \
#   --usd_weather_intensity 0.3 \
#   --anp_layer_scope last --anp_steps 4000 --anp_lr 0.1 --anp_eps 0.4 \
#   --anp_prune_ratio 0.25 --anp_clean_max 1000 --anp_prune_bn

# echo "[Task] GTSRB Weather | Ratio: 0.25 (Extreme) | ResNet34"
# $PY $MAIN --poison-type weather --arch resnet34 \
#   --checkpoint ${MODEL_DIR}/refool/GTSRB/resnet34/model_weather.th \
#   --gpuid 0 --target_label 0 \
#   --dataset GTSRB --data-dir ${DATA_DIR_GTSRB} \
#   --usd_weather_intensity 0.3 \
#   --anp_layer_scope last --anp_steps 4000 --anp_lr 0.1 --anp_eps 0.4 \
#   --anp_prune_ratio 0.25 --anp_clean_max 1000 --anp_prune_bn


# # --- CIFAR-100 WEATHER (Ratio 0.20) ---
# echo "[Task] CIFAR-100 Weather | Ratio: 0.20 (High) | ResNet18"
# $PY $MAIN --poison-type weather --arch resnet18 \
#   --checkpoint ${MODEL_DIR}/refool/CIFAR100/resnet18/model_weather.th \
#   --gpuid 2 --target_label 13 \
#   --dataset CIFAR100 --data-dir ${DATA_DIR_CIFAR} \
#   --usd_weather_intensity 0.3 \
#   --anp_layer_scope last --anp_steps 4000 --anp_lr 0.1 --anp_eps 0.4 \
#   --anp_prune_ratio 0.25 --anp_clean_max 1000 --anp_prune_bn

# echo "[Task] CIFAR-100 Weather | Ratio: 0.20 (High) | ResNet34"
# $PY $MAIN --poison-type weather --arch resnet34 \
#   --checkpoint ${MODEL_DIR}/refool/CIFAR100/resnet34/model_weather.th \
#   --gpuid 2 --target_label 13 \
#   --dataset CIFAR100 --data-dir ${DATA_DIR_CIFAR} \
#   --usd_weather_intensity 0.3 \
#   --anp_layer_scope last --anp_steps 4000 --anp_lr 0.1 --anp_eps 0.4 \
#   --anp_prune_ratio 0.25 --anp_clean_max 1000 --anp_prune_bn

# echo ""
# echo "=========================================================="
# echo "===== ALL TASKS COMPLETED. CHECK ASR/ACC TRADEOFF. ======="
# echo "=========================================================="

#!/bin/bash
set -e

PY=python
MAIN=run_anp_semantic.py

BASE_DIR="/home/hpc/LAB-data/disk2/zzj_1024041014/zzJ1-VGG/FIP/src"
DATA_DIR_IMAGENET="/home/hpc/LAB-data/disk2/zzj_1024041014/zzJ2-ViT/patch/data/imagenet_sub_20cls"

echo "======================================================"
echo "===== ANP DEFENSE RUNNER: IMAGENET_SUB ONLY ====="
echo "======================================================"
echo "[TIMESTAMP] $(date)"
echo "[PYTHON] $($PY -V)"

echo ""
echo "################################################"
echo "###         PART 1: IMAGENET_SUB REFOOL      ###"
echo "################################################"

echo "[Task] IMAGENET_SUB Refool | ResNet18"
$PY $MAIN --poison-type refool --arch resnet18 \
  --checkpoint ${BASE_DIR}/saved_models/imagenet_sub/resnet18/refool/best_backdoor_model.pth \
  --gpuid 0 --target_label 0 --poison_source 1 \
  --dataset IMAGENET_SUB --data-dir ${DATA_DIR_IMAGENET} \
  --anp_layer_scope last --anp_steps 1500 --anp_lr 0.05 --anp_eps 0.15 \
  --anp_prune_ratio 0.03 --anp_clean_max 500 --anp_prune_bn

echo "[Task] IMAGENET_SUB Refool | ResNet34"
$PY $MAIN --poison-type refool --arch resnet34 \
  --checkpoint ${BASE_DIR}/saved_models/imagenet_sub/resnet34/refool/best_backdoor_model.pth \
  --gpuid 0 --target_label 0 --poison_source 1 \
  --dataset IMAGENET_SUB --data-dir ${DATA_DIR_IMAGENET} \
  --anp_layer_scope last --anp_steps 1500 --anp_lr 0.05 --anp_eps 0.15 \
  --anp_prune_ratio 0.02 --anp_clean_max 500 --anp_prune_bn

echo ""
echo "################################################"
echo "###         PART 2: IMAGENET_SUB WEATHER     ###"
echo "################################################"

echo "[Task] IMAGENET_SUB Weather | ResNet18"
$PY $MAIN --poison-type weather --arch resnet18 \
  --checkpoint ${BASE_DIR}/saved_models/imagenet_sub/resnet18/weather/best_backdoor_model.pth \
  --gpuid 0 --target_label 0 \
  --dataset IMAGENET_SUB --data-dir ${DATA_DIR_IMAGENET} \
  --usd_weather_intensity 0.3 \
  --anp_layer_scope last --anp_steps 2000 --anp_lr 0.05 --anp_eps 0.20 \
  --anp_prune_ratio 0.06 --anp_clean_max 500 --anp_prune_bn

echo "[Task] IMAGENET_SUB Weather | ResNet34"
$PY $MAIN --poison-type weather --arch resnet34 \
  --checkpoint ${BASE_DIR}/saved_models/imagenet_sub/resnet34/weather/best_backdoor_model.pth \
  --gpuid 0 --target_label 0 \
  --dataset IMAGENET_SUB --data-dir ${DATA_DIR_IMAGENET} \
  --usd_weather_intensity 0.3 \
  --anp_layer_scope last --anp_steps 2000 --anp_lr 0.05 --anp_eps 0.20 \
  --anp_prune_ratio 0.04 --anp_clean_max 500 --anp_prune_bn

echo ""
echo "======================================================"
echo "===== ANP IMAGENET_SUB TASKS COMPLETED ====="
echo "======================================================"