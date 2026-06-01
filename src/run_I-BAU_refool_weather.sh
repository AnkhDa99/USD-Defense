# #!/bin/bash

# PY=python
# MAIN=Remove_Backdoor_I-BAU_main.py

# echo "============================"
# echo "===== I-BAU : REFOOL ======"
# echo "============================"

# # CIFAR-10
# $PY $MAIN --poison-type refool --arch resnet18 \
#  --checkpoint /home/hpc/LAB-data/disk2/zzj_1024041014/zzJ1-VGG/FIP/src/saved_models/refool/CIFAR10/resnet18/model_refool.th \
#  --gpuid 3 --target_label 0 --poison_source 9 --dataset CIFAR10

# $PY $MAIN --poison-type refool --arch resnet34 \
#  --checkpoint /home/hpc/LAB-data/disk2/zzj_1024041014/zzJ1-VGG/FIP/src/saved_models/refool/CIFAR10/resnet34/model_refool.th \
#  --gpuid 0 --target_label 0 --poison_source 9 --dataset CIFAR10


# # GTSRB
# $PY $MAIN --poison-type refool --arch resnet18 \
#  --checkpoint /home/hpc/LAB-data/disk2/zzj_1024041014/zzJ1-VGG/FIP/src/saved_models/refool/GTSRB/resnet18/model_refool.th \
#  --gpuid 2 --target_label 0 --poison_source 9 \
#  --dataset GTSRB --data-dir ./data/gtsrb

# $PY $MAIN --poison-type refool --arch resnet34 \
#  --checkpoint /home/hpc/LAB-data/disk2/zzj_1024041014/zzJ1-VGG/FIP/src/saved_models/refool/GTSRB/resnet34/model_refool.th \
#  --gpuid 0 --target_label 0 --poison_source 9 \
#  --dataset GTSRB --data-dir ./data/gtsrb


# # CIFAR-100
# $PY $MAIN --poison-type refool --arch resnet18 \
#  --checkpoint /home/hpc/LAB-data/disk2/zzj_1024041014/zzJ1-VGG/FIP/src/saved_models/refool/CIFAR100/resnet18/model_refool.th \
#  --gpuid 2 --target_label 81 --poison_source 13 \
#  --dataset CIFAR100 --data-dir ./data

# $PY $MAIN --poison-type refool --arch resnet34 \
#  --checkpoint /home/hpc/LAB-data/disk2/zzj_1024041014/zzJ1-VGG/FIP/src/saved_models/refool/CIFAR100/resnet34/model_refool.th \
#  --gpuid 2 --target_label 81 --poison_source 13 \
#  --dataset CIFAR100 --data-dir ./data


# echo "============================"
# echo "===== I-BAU : WEATHER ====="
# echo "============================"

# # CIFAR-10
# $PY $MAIN --poison-type weather --arch resnet18 \
#  --checkpoint /home/hpc/LAB-data/disk2/zzj_1024041014/zzJ1-VGG/FIP/src/saved_models/refool/CIFAR10/resnet18/model_weather.th \
#  --gpuid 3 --target_label 0 --dataset CIFAR10

# $PY $MAIN --poison-type weather --arch resnet34 \
#  --checkpoint /home/hpc/LAB-data/disk2/zzj_1024041014/zzJ1-VGG/FIP/src/saved_models/refool/CIFAR10/resnet34/model_weather.th \
#  --gpuid 2 --target_label 0 --dataset CIFAR10


# # GTSRB
# $PY $MAIN --poison-type weather --arch resnet18 \
#  --checkpoint /home/hpc/LAB-data/disk2/zzj_1024041014/zzJ1-VGG/FIP/src/saved_models/refool/GTSRB/resnet18/model_weather.th \
#  --gpuid 0 --target_label 0 \
#  --dataset GTSRB --data-dir ./data/gtsrb

# $PY $MAIN --poison-type weather --arch resnet34 \
#  --checkpoint /home/hpc/LAB-data/disk2/zzj_1024041014/zzJ1-VGG/FIP/src/saved_models/refool/GTSRB/resnet34/model_weather.th \
#  --gpuid 0 --target_label 0 \
#  --dataset GTSRB --data-dir ./data/gtsrb


# # CIFAR-100
# $PY $MAIN --poison-type weather --arch resnet18 \
#  --checkpoint /home/hpc/LAB-data/disk2/zzj_1024041014/zzJ1-VGG/FIP/src/saved_models/refool/CIFAR100/resnet18/model_weather.th \
#  --gpuid 2 --target_label 13 \
#  --dataset CIFAR100 --data-dir ./data

# $PY $MAIN --poison-type weather --arch resnet34 \
#  --checkpoint /home/hpc/LAB-data/disk2/zzj_1024041014/zzJ1-VGG/FIP/src/saved_models/refool/CIFAR100/resnet34/model_weather.th \
#  --gpuid 2 --target_label 13 \
#  --dataset CIFAR100 --data-dir ./data


# echo "===== ALL I-BAU DONE ====="
#!/bin/bash
set -e

PY=python
MAIN=Remove_Backdoor_I-BAU_main.py

BASE_DIR="/home/hpc/LAB-data/disk2/zzj_1024041014/zzJ1-VGG/FIP/src"
DATA_DIR_IMAGENET="/home/hpc/LAB-data/disk2/zzj_1024041014/zzJ2-ViT/patch/data/imagenet_sub_20cls"

echo "=============================================="
echo "===== I-BAU : IMAGENET_SUB ONLY ====="
echo "=============================================="

echo ""
echo "=============================="
echo "===== I-BAU : REFOOL ====="
echo "=============================="

$PY $MAIN --poison-type refool --arch resnet18 \
 --checkpoint ${BASE_DIR}/saved_models/imagenet_sub/resnet18/refool/best_backdoor_model.pth \
 --gpuid 0 --target_label 0 --poison_source 1 \
 --dataset IMAGENET_SUB --data-dir ${DATA_DIR_IMAGENET}

$PY $MAIN --poison-type refool --arch resnet34 \
 --checkpoint ${BASE_DIR}/saved_models/imagenet_sub/resnet34/refool/best_backdoor_model.pth \
 --gpuid 0 --target_label 0 --poison_source 1 \
 --dataset IMAGENET_SUB --data-dir ${DATA_DIR_IMAGENET}

echo ""
echo "=============================="
echo "===== I-BAU : WEATHER ====="
echo "=============================="

$PY $MAIN --poison-type weather --arch resnet18 \
 --checkpoint ${BASE_DIR}/saved_models/imagenet_sub/resnet18/weather/best_backdoor_model.pth \
 --gpuid 0 --target_label 0 \
 --dataset IMAGENET_SUB --data-dir ${DATA_DIR_IMAGENET}

$PY $MAIN --poison-type weather --arch resnet34 \
 --checkpoint ${BASE_DIR}/saved_models/imagenet_sub/resnet34/weather/best_backdoor_model.pth \
 --gpuid 0 --target_label 0 \
 --dataset IMAGENET_SUB --data-dir ${DATA_DIR_IMAGENET}

echo "===== ALL I-BAU IMAGENET_SUB DONE ====="