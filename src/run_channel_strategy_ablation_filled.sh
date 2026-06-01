#!/usr/bin/env bash
set -euo pipefail

# ===== 可按需覆盖的环境变量 =====
PYTHON_BIN=${PYTHON_BIN:-python}
SCRIPT=${SCRIPT:-Remove_Backdoor_FIP0.py}

DATASET=${DATASET:-CIFAR10}
ARCH=${ARCH:-resnet18}
ATTACK=${ATTACK:-refool}
CHECKPOINT=${CHECKPOINT:-/home/hpc/LAB-data/disk2/zzj_1024041014/zzJ1-VGG/FIP/src/saved_models/refool/CIFAR10/resnet18/model_refool.th}
GPUID=${GPUID:-1}
REG_F=${REG_F:-0.01}
TARGET_LABEL=${TARGET_LABEL:-0}
POISON_SOURCE=${POISON_SOURCE:-9}
DATA_DIR=${DATA_DIR:-../data}
OUTPUT_ROOT=${OUTPUT_ROOT:-./save/channel_strategy_ablation}
NUM_CLASS=${NUM_CLASS:-10}
USE_USD=${USE_USD:-1}

# 这份脚本适配你当前这版 Remove_Backdoor_FIP0.py 的参数接口
# 它支持 --epoch，不支持 --nb-epochs / --epoch-aggregation
EPOCH=${EPOCH:-2000}
BATCH_SIZE=${BATCH_SIZE:-128}
LR=${LR:-0.005}
VAL_RATIO=${VAL_RATIO:-0.1}

MASK_NUM_VIEWS=${MASK_NUM_VIEWS:-3}
MASK_LAMBDA_CLEAN=${MASK_LAMBDA_CLEAN:-1.0}
MASK_RANDOM_SEED=${MASK_RANDOM_SEED:-123}

mkdir -p "$OUTPUT_ROOT"

run_one() {
  local strategy="$1"
  local outdir="$OUTPUT_ROOT/${DATASET}/${ARCH}/${ATTACK}/${strategy}"
  mkdir -p "$outdir"

  echo "============================================================"
  echo "[RUN] strategy=${strategy}"
  echo "[RUN] dataset=${DATASET}, arch=${ARCH}, attack=${ATTACK}"
  echo "[RUN] checkpoint=${CHECKPOINT}"
  echo "[RUN] output=${outdir}"
  echo "============================================================"

  CMD=(
    "$PYTHON_BIN" "$SCRIPT"
    --poison-type "$ATTACK"
    --arch "$ARCH"
    --checkpoint "$CHECKPOINT"
    --gpuid "$GPUID"
    --reg_F "$REG_F"
    --target_label "$TARGET_LABEL"
    --dataset "$DATASET"
    --data-dir "$DATA_DIR"
    --output-dir "$outdir"
    --mask_strategy "$strategy"
    --mask_num_views "$MASK_NUM_VIEWS"
    --mask_lambda_clean "$MASK_LAMBDA_CLEAN"
    --mask_random_seed "$MASK_RANDOM_SEED"
    --epoch "$EPOCH"
    --batch-size "$BATCH_SIZE"
    --lr "$LR"
    --val-ratio "$VAL_RATIO"
  )

  if [[ "$NUM_CLASS" != "" ]]; then
    CMD+=(--num_class "$NUM_CLASS")
  fi

  if [[ "$POISON_SOURCE" != "" ]]; then
    CMD+=(--poison_source "$POISON_SOURCE")
  fi

  if [[ "$USE_USD" == "1" ]]; then
    CMD+=(--use_usd)
  fi

  "${CMD[@]}" | tee "$outdir/run.log"
}

run_one random
run_one high_response
run_one ours

SUMMARY_CSV="$OUTPUT_ROOT/${DATASET}_${ARCH}_${ATTACK}_channel_strategy_summary.csv"
SUMMARY_MD="$OUTPUT_ROOT/${DATASET}_${ARCH}_${ATTACK}_channel_strategy_summary.md"
SUMMARY_TXT="$OUTPUT_ROOT/${DATASET}_${ARCH}_${ATTACK}_channel_strategy_summary.txt"

export OUTPUT_ROOT DATASET ARCH ATTACK SUMMARY_CSV SUMMARY_MD SUMMARY_TXT

python - <<'PY'
import os
import numpy as np
import pandas as pd

output_root = os.environ['OUTPUT_ROOT']
dataset = os.environ['DATASET']
arch = os.environ['ARCH']
attack = os.environ['ATTACK']
summary_csv = os.environ['SUMMARY_CSV']
summary_md = os.environ['SUMMARY_MD']
summary_txt = os.environ['SUMMARY_TXT']

mapping = [
    ('random', 'Random'),
    ('high_response', 'High-response'),
    ('ours', 'Ours'),
]

rows = []
for key, name in mapping:
    npz_path = os.path.join(output_root, dataset, arch, attack, key, f'remove_model_{attack}_{dataset}_.npz')
    if not os.path.exists(npz_path):
        rows.append({'Strategy': name, 'ACC(%)': 'NA', 'ASR(%)': 'NA'})
        continue
    data = np.load(npz_path)
    acc = float(data['cl_test'][-1]) * 100.0
    asr = float(data['po_acc'][-1]) * 100.0
    rows.append({'Strategy': name, 'ACC(%)': round(acc, 2), 'ASR(%)': round(asr, 2)})

df = pd.DataFrame(rows)
df.to_csv(summary_csv, index=False)

md_lines = []
md_lines.append(f"表 X {dataset}数据集下不同通道选择策略的净化结果")
md_lines.append(f"Table X Purification results under different channel selection strategies on {dataset}")
md_lines.append("")
md_lines.append("| Strategy | ACC(%) | ASR(%) |")
md_lines.append("|---|---:|---:|")
for _, r in df.iterrows():
    md_lines.append(f"| {r['Strategy']} | {r['ACC(%)']} | {r['ASR(%)']} |")

with open(summary_md, 'w', encoding='utf-8') as f:
    f.write('\n'.join(md_lines))
with open(summary_txt, 'w', encoding='utf-8') as f:
    f.write('\n'.join(md_lines))

print('\n'.join(md_lines))
print(f"\n[Saved] {summary_csv}")
print(f"[Saved] {summary_md}")
print(f"[Saved] {summary_txt}")
PY
