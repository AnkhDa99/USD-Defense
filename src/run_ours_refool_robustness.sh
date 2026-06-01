#!/usr/bin/env bash
set -euo pipefail

# =========================================================
# Ours 方法参数鲁棒性分析（含 reg_F）
# Dataset: CIFAR10
# Model:   ResNet-18
# Attack:  Refool
# Output:  CSV + 5张单图 + 1张2x3总图
# =========================================================

PROJECT_ROOT="/home/hpc/LAB-data/disk2/zzj_1024041014/zzJ1-VGG/FIP/src"
PYTHON_BIN="python"

# 物理 GPU 编号
GPU_ID=2

CHECKPOINT="/home/hpc/LAB-data/disk2/zzj_1024041014/zzJ1-VGG/FIP/src/saved_models/refool/CIFAR10/resnet18/model_refool.th"

POISON_TYPE="refool"
ARCH="resnet18"
DATASET="CIFAR10"
TARGET_LABEL=0
POISON_SOURCE=9
DATA_DIR="${PROJECT_ROOT}/data"

# 输出目录
EXP_ROOT="${PROJECT_ROOT}/temp/ours_refool_robustness_v2"
LOG_DIR="${EXP_ROOT}/logs"
RUN_DIR="${EXP_ROOT}/runs"
mkdir -p "${LOG_DIR}" "${RUN_DIR}"

RESULT_CSV="${EXP_ROOT}/robustness_results.csv"
TOTAL_PLOT="${EXP_ROOT}/robustness_2x3.png"

# 单图
PLOT1="${EXP_ROOT}/reg_F.png"
PLOT2="${EXP_ROOT}/mask_lambda_clean.png"
PLOT3="${EXP_ROOT}/mask_num_views.png"
PLOT4="${EXP_ROOT}/usd_lambda_suppress.png"
PLOT5="${EXP_ROOT}/usd_thresh.png"

# ---------------------------------------------------------
# 扫描参数
# ---------------------------------------------------------
# reg_F 对 Fisher 平滑强度敏感，建议论文中强调需谨慎设置
REG_F_VALUES=(0.001 0.003 0.005 0.01 0.02)

MASK_LAMBDA_CLEAN_VALUES=(0.5 1.0 1.5 2.0)
MASK_NUM_VIEWS_VALUES=(1 2 3 4)
USD_LAMBDA_SUPPRESS_VALUES=(0.2 0.4 0.6 0.8)
USD_THRESH_VALUES=(0.85 0.88 0.90 0.92)

# 默认基准值
BASE_REG_F=0.01
BASE_MASK_LAMBDA_CLEAN=1.0
BASE_MASK_NUM_VIEWS=3
BASE_USD_LAMBDA_SUPPRESS=0.4
BASE_USD_THRESH=0.90

# 你当前主代码支持这两个参数
NB_EPOCHS=1000
EPOCH_AGG=250
BATCH_SIZE=128
LR=0.005

echo "param_name,param_value,final_acc,final_asr,best_acc,best_asr,run_dir" > "${RESULT_CSV}"

log() {
  echo "[$(date '+%F %T')] $*"
}

run_one() {
  local param_name="$1"
  local param_value="$2"

  local reg_F="${BASE_REG_F}"
  local mask_lambda_clean="${BASE_MASK_LAMBDA_CLEAN}"
  local mask_num_views="${BASE_MASK_NUM_VIEWS}"
  local usd_lambda_suppress="${BASE_USD_LAMBDA_SUPPRESS}"
  local usd_thresh="${BASE_USD_THRESH}"

  if [[ "${param_name}" == "reg_F" ]]; then
    reg_F="${param_value}"
  elif [[ "${param_name}" == "mask_lambda_clean" ]]; then
    mask_lambda_clean="${param_value}"
  elif [[ "${param_name}" == "mask_num_views" ]]; then
    mask_num_views="${param_value}"
  elif [[ "${param_name}" == "usd_lambda_suppress" ]]; then
    usd_lambda_suppress="${param_value}"
  elif [[ "${param_name}" == "usd_thresh" ]]; then
    usd_thresh="${param_value}"
  else
    echo "[ERROR] Unknown param_name=${param_name}"
    exit 1
  fi

  local tag="${param_name}_${param_value}"
  local out_dir="${RUN_DIR}/${tag}"
  local log_file="${LOG_DIR}/${tag}.log"
  mkdir -p "${out_dir}"

  log "Running ${tag}"

  CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" "${PROJECT_ROOT}/Remove_Backdoor_FIP0.py" \
    --poison-type "${POISON_TYPE}" \
    --arch "${ARCH}" \
    --checkpoint "${CHECKPOINT}" \
    --gpuid 0 \
    --reg_F "${reg_F}" \
    --target_label "${TARGET_LABEL}" \
    --poison_source "${POISON_SOURCE}" \
    --dataset "${DATASET}" \
    --data-dir "${DATA_DIR}" \
    --output-dir "${out_dir}" \
    --batch-size "${BATCH_SIZE}" \
    --lr "${LR}" \
    --mask_strategy ours \
    --use_usd \
    --defense_preset balanced \
    --mask_lambda_clean "${mask_lambda_clean}" \
    --mask_num_views "${mask_num_views}" \
    --usd_lambda_suppress "${usd_lambda_suppress}" \
    --usd_thresh_start "${usd_thresh}" \
    --usd_thresh_end "${usd_thresh}" \
    > "${log_file}" 2>&1

  local npz_file="${out_dir}/remove_model_refool_CIFAR10_.npz"

  if [[ ! -f "${npz_file}" ]]; then
    echo "[ERROR] NPZ result not found: ${npz_file}"
    echo "[ERROR] Check log: ${log_file}"
    exit 1
  fi

  python - <<PY
import numpy as np
import pandas as pd

npz_path = r"${npz_file}"
csv_path = r"${RESULT_CSV}"
param_name = "${param_name}"
param_value = "${param_value}"
run_dir = r"${out_dir}"

data = np.load(npz_path)
acc_arr = data["cl_test"]
asr_arr = data["po_acc"]

final_acc = float(acc_arr[-1] * 100.0)
final_asr = float(asr_arr[-1] * 100.0)
best_acc = float(acc_arr.max() * 100.0)
best_asr = float(asr_arr.min() * 100.0)

row = pd.DataFrame([{
    "param_name": param_name,
    "param_value": param_value,
    "final_acc": round(final_acc, 4),
    "final_asr": round(final_asr, 4),
    "best_acc": round(best_acc, 4),
    "best_asr": round(best_asr, 4),
    "run_dir": run_dir
}])
row.to_csv(csv_path, mode="a", header=False, index=False)
print(row.to_string(index=False))
PY
}

# ---------------------------------------------------------
# 开始扫描
# ---------------------------------------------------------
for v in "${REG_F_VALUES[@]}"; do
  run_one "reg_F" "${v}"
done

for v in "${MASK_LAMBDA_CLEAN_VALUES[@]}"; do
  run_one "mask_lambda_clean" "${v}"
done

for v in "${MASK_NUM_VIEWS_VALUES[@]}"; do
  run_one "mask_num_views" "${v}"
done

for v in "${USD_LAMBDA_SUPPRESS_VALUES[@]}"; do
  run_one "usd_lambda_suppress" "${v}"
done

for v in "${USD_THRESH_VALUES[@]}"; do
  run_one "usd_thresh" "${v}"
done

# ---------------------------------------------------------
# 画图：5张单图 + 1张2x3总图
# ---------------------------------------------------------
python - <<PY
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager

csv_path = r"${RESULT_CSV}"
total_plot = r"${TOTAL_PLOT}"
plot_map = {
    "reg_F": r"${PLOT1}",
    "mask_lambda_clean": r"${PLOT2}",
    "mask_num_views": r"${PLOT3}",
    "usd_lambda_suppress": r"${PLOT4}",
    "usd_thresh": r"${PLOT5}",
}

df = pd.read_csv(csv_path)

# 显式指定中文字体，避免 fallback 到 DejaVu Sans
font_candidates = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/truetype/arphic/uming.ttc",
]
font_path = None
for fp in font_candidates:
    import os
    if os.path.exists(fp):
        font_path = fp
        break

if font_path is None:
    raise RuntimeError("未找到可用中文字体，请先安装 Noto Sans CJK 或文泉驿字体。")

font_prop = font_manager.FontProperties(fname=font_path)
plt.rcParams["axes.unicode_minus"] = False

param_info = [
    ("reg_F", "Fisher正则强度", "#1f77b4", "#ff7f0e"),
    ("mask_lambda_clean", "责任通道平衡系数", "#2ca02c", "#8c564b"),
    ("mask_num_views", "责任通道多视图数", "#d62728", "#e377c2"),
    ("usd_lambda_suppress", "伪目标抑制权重", "#9467bd", "#7f7f7f"),
    ("usd_thresh", "高置信门控阈值", "#17becf", "#bcbd22"),
]

def annotate_points(ax, xs, ys, color, fmt="{:.2f}", y_offset=0.6):
    for x, y in zip(xs, ys):
        ax.text(
            x, y + y_offset, fmt.format(y),
            color=color, fontsize=8, ha="center", va="bottom",
            fontproperties=font_prop
        )

# 单图
for pname, title, c1, c2 in param_info:
    sub = df[df["param_name"] == pname].copy()
    sub["param_value_num"] = sub["param_value"].astype(float)
    sub = sub.sort_values("param_value_num")

    xs = sub["param_value_num"].tolist()
    accs = sub["final_acc"].tolist()
    asrs = sub["final_asr"].tolist()

    plt.figure(figsize=(7, 5))
    plt.plot(xs, accs, marker="o", linewidth=2, color=c1, label="ACC")
    plt.plot(xs, asrs, marker="s", linewidth=2, color=c2, label="ASR")

    annotate_points(plt.gca(), xs, accs, c1, fmt="{:.2f}", y_offset=0.8)
    annotate_points(plt.gca(), xs, asrs, c2, fmt="{:.2f}", y_offset=0.3)

    plt.title(title, fontproperties=font_prop, fontsize=14)
    plt.xlabel("参数取值", fontproperties=font_prop, fontsize=12)
    plt.ylabel("指标值（%）", fontproperties=font_prop, fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(prop=font_prop)
    plt.tight_layout()
    plt.savefig(plot_map[pname], dpi=300, bbox_inches="tight")
    plt.close()

# 2x3 总图
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
axes = axes.flatten()

for idx, (pname, title, c1, c2) in enumerate(param_info):
    ax = axes[idx]
    sub = df[df["param_name"] == pname].copy()
    sub["param_value_num"] = sub["param_value"].astype(float)
    sub = sub.sort_values("param_value_num")

    xs = sub["param_value_num"].tolist()
    accs = sub["final_acc"].tolist()
    asrs = sub["final_asr"].tolist()

    ax.plot(xs, accs, marker="o", linewidth=2, color=c1, label="ACC")
    ax.plot(xs, asrs, marker="s", linewidth=2, color=c2, label="ASR")

    for x, y in zip(xs, accs):
        ax.text(
            x, y + 0.8, f"{y:.2f}",
            color=c1, fontsize=7, ha="center", va="bottom",
            fontproperties=font_prop
        )
    for x, y in zip(xs, asrs):
        ax.text(
            x, y + 0.3, f"{y:.2f}",
            color=c2, fontsize=7, ha="center", va="bottom",
            fontproperties=font_prop
        )

    ax.set_title(title, fontproperties=font_prop, fontsize=12)
    ax.set_xlabel("参数取值", fontproperties=font_prop, fontsize=11)
    ax.set_ylabel("指标值（%）", fontproperties=font_prop, fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(prop=font_prop)

# 第6个子图留空
axes[-1].axis("off")

plt.tight_layout()
plt.savefig(total_plot, dpi=300, bbox_inches="tight")
plt.close()

print("图片生成完成")
print("单图路径：")
for k, v in plot_map.items():
    print(k, "->", v)
print("总图路径：", total_plot)
PY

log "Done."
echo "结果 CSV: ${RESULT_CSV}"
echo "单图1: ${PLOT1}"
echo "单图2: ${PLOT2}"
echo "单图3: ${PLOT3}"
echo "单图4: ${PLOT4}"
echo "单图5: ${PLOT5}"
echo "总图 : ${TOTAL_PLOT}"