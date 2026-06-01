#!/usr/bin/env bash
set -euo pipefail

# ========= 可覆盖环境变量 =========
PYTHON_BIN=${PYTHON_BIN:-python}
SCRIPT=${SCRIPT:-Remove_Backdoor_FIP0.py}

DATASET=${DATASET:-CIFAR10}
ARCH=${ARCH:-resnet18}
ATTACK=${ATTACK:-refool}
CHECKPOINT=${CHECKPOINT:-/home/hpc/LAB-data/disk2/zzj_1024041014/zzJ1-VGG/FIP/src/saved_models/refool/CIFAR10/resnet18/model_refool.th}
GPUID=${GPUID:-2}
TARGET_LABEL=${TARGET_LABEL:-0}
POISON_SOURCE=${POISON_SOURCE:-9}
DATA_DIR=${DATA_DIR:-../data}
OUTPUT_ROOT=${OUTPUT_ROOT:-./save/channel_intervention}
NUM_CLASSES=${NUM_CLASSES:-10}

MASK_NUM_VIEWS=${MASK_NUM_VIEWS:-3}
MASK_LAMBDA_CLEAN=${MASK_LAMBDA_CLEAN:-1.0}
MASK_RANDOM_SEED=${MASK_RANDOM_SEED:-123}

INTERVENTION_BATCHES=${INTERVENTION_BATCHES:-50}
INTERVENTION_MODE=${INTERVENTION_MODE:-zero}
INTERVENTION_SAVE_NAME=${INTERVENTION_SAVE_NAME:-channel_intervention_result.csv}

mkdir -p "$OUTPUT_ROOT"

run_one() {
  local strategy="$1"
  local outdir="$OUTPUT_ROOT/${DATASET}/${ARCH}/${ATTACK}/${strategy}"
  mkdir -p "$outdir"

  echo "============================================================"
  echo "[RUN] strategy=${strategy}"
  echo "[RUN] mode=Channel-Intervention"
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
    --target_label "$TARGET_LABEL"
    --dataset "$DATASET"
    --data-dir "$DATA_DIR"
    --output-dir "$outdir"
    --poison_source "$POISON_SOURCE"
    --num_class "$NUM_CLASSES"
    --mask_strategy "$strategy"
    --mask_num_views "$MASK_NUM_VIEWS"
    --mask_lambda_clean "$MASK_LAMBDA_CLEAN"
    --mask_random_seed "$MASK_RANDOM_SEED"
    --run_channel_intervention
    --intervention_batches "$INTERVENTION_BATCHES"
    --intervention_mode "$INTERVENTION_MODE"
    --intervention_save_name "$INTERVENTION_SAVE_NAME"
  )

  "${CMD[@]}" | tee "$outdir/run.log"
}

run_one random
run_one high_response
run_one ours

SUMMARY_CSV="$OUTPUT_ROOT/${DATASET}_${ARCH}_${ATTACK}_channel_intervention_summary.csv"
SUMMARY_MD="$OUTPUT_ROOT/${DATASET}_${ARCH}_${ATTACK}_channel_intervention_summary.md"
SUMMARY_TXT="$OUTPUT_ROOT/${DATASET}_${ARCH}_${ATTACK}_channel_intervention_summary.txt"

export OUTPUT_ROOT DATASET ARCH ATTACK INTERVENTION_SAVE_NAME SUMMARY_CSV SUMMARY_MD SUMMARY_TXT

python - <<'PY'
import os
import pandas as pd

output_root = os.environ['OUTPUT_ROOT']
dataset = os.environ['DATASET']
arch = os.environ['ARCH']
attack = os.environ['ATTACK']
intervention_save_name = os.environ['INTERVENTION_SAVE_NAME']
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
    csv_path = os.path.join(output_root, dataset, arch, attack, key, intervention_save_name)
    if not os.path.exists(csv_path):
        rows.append({
            'Strategy': name,
            'Clean ACC Drop(%)': 'NA',
            'Poison ASR Drop(%)': 'NA',
            'Clean GT Logit Drop': 'NA',
            'Poison Target Logit Drop': 'NA'
        })
        continue

    df = pd.read_csv(csv_path)
    r = df.iloc[0]
    rows.append({
        'Strategy': name,
        'Clean ACC Drop(%)': round(float(r['clean_acc_drop']) * 100.0, 2),
        'Poison ASR Drop(%)': round(float(r['poison_asr_drop']) * 100.0, 2),
        'Clean GT Logit Drop': round(float(r['clean_gt_logit_drop']), 4),
        'Poison Target Logit Drop': round(float(r['poison_target_logit_drop']), 4),
    })

out_df = pd.DataFrame(rows)
out_df.to_csv(summary_csv, index=False)

lines = []
lines.append(f"表 X {dataset}数据集下不同通道选择策略的通道干预结果")
lines.append(f"Table X Channel intervention results under different channel selection strategies on {dataset}")
lines.append("")
lines.append("| Strategy | Clean ACC Drop(%) | Poison ASR Drop(%) | Clean GT Logit Drop | Poison Target Logit Drop |")
lines.append("|---|---:|---:|---:|---:|")
for _, r in out_df.iterrows():
    lines.append(
        f"| {r['Strategy']} | {r['Clean ACC Drop(%)']} | {r['Poison ASR Drop(%)']} | {r['Clean GT Logit Drop']} | {r['Poison Target Logit Drop']} |"
    )

with open(summary_md, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
with open(summary_txt, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

print("\n".join(lines))
print(f"\n[Saved] {summary_csv}")
print(f"[Saved] {summary_md}")
print(f"[Saved] {summary_txt}")
PY
