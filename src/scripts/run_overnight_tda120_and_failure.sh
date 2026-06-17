#!/usr/bin/env bash
set -uo pipefail

# ============================================================
# Overnight script:
#   Task-1: 120 pseudo-target detection experiments
#   Task-2: failure-impact analysis on Refool
#
# Safe output:
#   All new models/results are saved under temp/overnight_<timestamp>/
#   It will NOT overwrite src/saved_models/refool/
# ============================================================

# ---------------- Basic paths ----------------
PROJECT_ROOT="${PROJECT_ROOT:-/home/hpc/LAB-data/disk2/zzj_1024041014/zzJ1-VGG/FIP/src}"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_ID="${GPU_ID:-0}"

# ImageNet-Sub 数据路径：如果你的路径不同，运行时用环境变量覆盖
IMAGENET_DATA_ROOT="${IMAGENET_DATA_ROOT:-/home/hpc/LAB-data/disk2/zzj_1024041014/zzJ2-ViT/patch/data/imagenet_sub_20cls}"

STAMP="${STAMP:-$(date '+%Y%m%d_%H%M%S')}"
OUT_ROOT="${OUT_ROOT:-${PROJECT_ROOT}/temp/overnight_${STAMP}}"

# 是否运行两个任务
RUN_TDA120="${RUN_TDA120:-1}"
RUN_FAILURE="${RUN_FAILURE:-1}"

# 训练参数
EPOCHS="${EPOCHS:-100}"
POISON_RATE="${POISON_RATE:-0.1}"

# ImageNet 训练参数
IMAGENET_BATCH_SIZE="${IMAGENET_BATCH_SIZE:-32}"
IMAGENET_LR="${IMAGENET_LR:-0.01}"
IMAGENET_SCHEDULE_1="${IMAGENET_SCHEDULE_1:-75}"
IMAGENET_SCHEDULE_2="${IMAGENET_SCHEDULE_2:-120}"
IMAGENET_NUM_WORKERS="${IMAGENET_NUM_WORKERS:-8}"

# 目标检测参数
DET_MAX_PER_CLASS="${DET_MAX_PER_CLASS:-20}"
DET_NUM_VIEWS="${DET_NUM_VIEWS:-3}"
DET_BATCH_SIZE="${DET_BATCH_SIZE:-64}"
DET_GAMMA="${DET_GAMMA:-0.0}"

# ---------------- Output dirs ----------------
TDA_MODEL_ROOT="${OUT_ROOT}/saved_models_tda120"
TDA_DETECT_ROOT="${OUT_ROOT}/detect_outputs_tda120"
FAIL_ROOT="${OUT_ROOT}/failure_impact"
LOG_ROOT="${OUT_ROOT}/logs"

mkdir -p "${TDA_MODEL_ROOT}" "${TDA_DETECT_ROOT}" "${FAIL_ROOT}" "${LOG_ROOT}"

cd "${PROJECT_ROOT}"

# ---------------- Auto-detect parser flags ----------------
if "${PYTHON_BIN}" Remove_Backdoor_FIP0.py -h 2>&1 | grep -q -- "--num_classes"; then
  NUM_CLASS_FLAG="--num_classes"
else
  NUM_CLASS_FLAG="--num_class"
fi

if "${PYTHON_BIN}" Remove_Backdoor_FIP0.py -h 2>&1 | grep -q -- "--defense_target_label"; then
  DEFENSE_TARGET_FLAG="--defense_target_label"
elif "${PYTHON_BIN}" Remove_Backdoor_FIP0.py -h 2>&1 | grep -q -- "--override_target_label"; then
  DEFENSE_TARGET_FLAG="--override_target_label"
else
  DEFENSE_TARGET_FLAG=""
fi

echo "============================================================"
echo "[Config]"
echo "PROJECT_ROOT       = ${PROJECT_ROOT}"
echo "OUT_ROOT           = ${OUT_ROOT}"
echo "GPU_ID             = ${GPU_ID}"
echo "EPOCHS             = ${EPOCHS}"
echo "RUN_TDA120         = ${RUN_TDA120}"
echo "RUN_FAILURE        = ${RUN_FAILURE}"
echo "IMAGENET_DATA_ROOT = ${IMAGENET_DATA_ROOT}"
echo "NUM_CLASS_FLAG     = ${NUM_CLASS_FLAG}"
echo "DEFENSE_TARGET_FLAG= ${DEFENSE_TARGET_FLAG}"
echo "============================================================"

"${PYTHON_BIN}" -m py_compile Remove_Backdoor_FIP0.py || {
  echo "[ERROR] Remove_Backdoor_FIP0.py syntax check failed."
  exit 1
}

log() {
  echo "[$(date '+%F %T')] $*"
}

get_num_classes() {
  case "$1" in
    CIFAR10) echo 10 ;;
    CIFAR100) echo 100 ;;
    GTSRB) echo 43 ;;
    IMAGENET_SUB) echo 20 ;;
    *) echo 10 ;;
  esac
}

get_data_dir() {
  case "$1" in
    GTSRB) echo "./data/gtsrb" ;;
    IMAGENET_SUB) echo "${IMAGENET_DATA_ROOT}" ;;
    *) echo "./data" ;;
  esac
}

get_source() {
  local target="$1"
  local num_classes="$2"
  local s=$(( (target + 7) % num_classes ))
  if [[ "${s}" -eq "${target}" ]]; then
    s=$(( (target + 1) % num_classes ))
  fi
  echo "${s}"
}

get_reg_f() {
  local dataset="$1"
  local arch="$2"
  local attack="$3"

  if [[ "${dataset}" == "CIFAR10" && "${attack}" == "weather" && "${arch}" == "resnet18" ]]; then
    echo 0.03
  elif [[ "${dataset}" == "CIFAR10" && "${attack}" == "weather" && "${arch}" == "resnet34" ]]; then
    echo 0.01
  elif [[ "${dataset}" == "CIFAR10" && "${attack}" == "refool" && "${arch}" == "resnet18" ]]; then
    echo 0.01
  else
    echo 0.005
  fi
}

find_checkpoint() {
  local search_root="$1"
  find "${search_root}" -type f \( \
      -name 'model_refool.th' -o \
      -name 'model_weather.th' -o \
      -name 'best_backdoor_model.pth' -o \
      -name '*.pth' -o \
      -name '*.th' \
    \) -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n 1 | cut -d' ' -f2-
}


# ============================================================
# Training functions
# ============================================================

train_backdoor_model() {
  local dataset="$1"
  local arch="$2"
  local attack="$3"
  local target="$4"
  local source="$5"
  local num_classes="$6"
  local outdir="$7"

  mkdir -p "${outdir}"

  local existing
  existing=$(find_checkpoint "${outdir}")
  if [[ -n "${existing}" ]]; then
    log "[Skip Train] Existing checkpoint found: ${existing}"
    return 0
  fi

  log "[Train] dataset=${dataset}, arch=${arch}, attack=${attack}, target=${target}, source=${source}"
  log "[Train Save Dir] ${outdir}"

  if [[ "${dataset}" == "CIFAR10" ]]; then
    if [[ "${attack}" == "weather" ]]; then
      CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" train_backdoor_cifar.py \
        --poison-type weather \
        --poison-rate "${POISON_RATE}" \
        --target_label "${target}" \
        --poison-target "${target}" \
        --arch "${arch}" \
        --epoch "${EPOCHS}" \
        --save-every "${EPOCHS}" \
        --output-dir "${outdir}" \
        --gpuid 0
    else
      CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" train_backdoor_cifar.py \
        --poison-type refool \
        --poison-rate "${POISON_RATE}" \
        --target_label "${target}" \
        --poison-target "${target}" \
        --poison_source "${source}" \
        --poison-source "${source}" \
        --trigger_alpha 0.5 \
        --arch "${arch}" \
        --epoch "${EPOCHS}" \
        --save-every "${EPOCHS}" \
        --output-dir "${outdir}" \
        --gpuid 0
    fi

  elif [[ "${dataset}" == "GTSRB" ]]; then
    if [[ "${attack}" == "weather" ]]; then
      CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" train_backdoor_gtsrb.py \
        --poison-type weather \
        --poison-rate "${POISON_RATE}" \
        --poison-target "${target}" \
        --arch "${arch}" \
        --output-dir "${outdir}" \
        --gpuid 0 \
        --dataset GTSRB \
        --num_class "${num_classes}" \
        --epoch "${EPOCHS}" \
        --save-every "${EPOCHS}"
    else
      CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" train_backdoor_gtsrb.py \
        --poison-type refool \
        --poison-rate "${POISON_RATE}" \
        --poison-target "${target}" \
        --poison-source "${source}" \
        --trigger_alpha 0.5 \
        --arch "${arch}" \
        --output-dir "${outdir}" \
        --gpuid 0 \
        --dataset GTSRB \
        --num_class "${num_classes}" \
        --epoch "${EPOCHS}" \
        --save-every "${EPOCHS}"
    fi

  elif [[ "${dataset}" == "CIFAR100" ]]; then
    if [[ "${attack}" == "weather" ]]; then
      CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" train_backdoor_cifar100.py \
        --poison-type weather \
        --poison-rate "${POISON_RATE}" \
        --poison-target "${target}" \
        --arch "${arch}" \
        --output-dir "${outdir}" \
        --gpuid 0 \
        --dataset CIFAR100 \
        --num_class "${num_classes}" \
        --epoch "${EPOCHS}" \
        --save-every "${EPOCHS}"
    else
      CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" train_backdoor_cifar100.py \
        --poison-type refool \
        --poison-rate "${POISON_RATE}" \
        --poison-source "${source}" \
        --poison-target "${target}" \
        --arch "${arch}" \
        --output-dir "${outdir}" \
        --gpuid 0 \
        --dataset CIFAR100 \
        --num_class "${num_classes}" \
        --epoch "${EPOCHS}" \
        --save-every "${EPOCHS}"
    fi

  elif [[ "${dataset}" == "IMAGENET_SUB" ]]; then
    if [[ "${attack}" == "weather" ]]; then
      CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" train_backdoor_imagenet.py \
        --data-root "${IMAGENET_DATA_ROOT}" \
        --dataset IMAGENET_SUB \
        --arch "${arch}" \
        --batch-size "${IMAGENET_BATCH_SIZE}" \
        --epoch "${EPOCHS}" \
        --save-every "${EPOCHS}" \
        --schedule "${IMAGENET_SCHEDULE_1}" "${IMAGENET_SCHEDULE_2}" \
        --lr "${IMAGENET_LR}" \
        --num_workers "${IMAGENET_NUM_WORKERS}" \
        --poison-type weather \
        --poison-rate "${POISON_RATE}" \
        --poison-target "${target}" \
        --weather_effect rain \
        --weather_intensity 0.3 \
        --output-dir "${outdir}" \
        --gpuid 0 \
        --seed 123 \
        --pretrained
    else
      CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" train_backdoor_imagenet.py \
        --data-root "${IMAGENET_DATA_ROOT}" \
        --dataset IMAGENET_SUB \
        --arch "${arch}" \
        --batch-size "${IMAGENET_BATCH_SIZE}" \
        --epoch "${EPOCHS}" \
        --save-every "${EPOCHS}" \
        --schedule "${IMAGENET_SCHEDULE_1}" "${IMAGENET_SCHEDULE_2}" \
        --lr "${IMAGENET_LR}" \
        --num_workers "${IMAGENET_NUM_WORKERS}" \
        --poison-type refool \
        --poison-rate "${POISON_RATE}" \
        --poison-target "${target}" \
        --poison-source "${source}" \
        --refool_alpha_range 0.4,0.7 \
        --refool_gamma_range 0.9,1.1 \
        --output-dir "${outdir}" \
        --gpuid 0 \
        --seed 123 \
        --pretrained
    fi
  fi
}


# ============================================================
# Detection function
# ============================================================

run_target_detection() {
  local dataset="$1"
  local arch="$2"
  local attack="$3"
  local target="$4"
  local source="$5"
  local checkpoint="$6"
  local outdir="$7"

  local num_classes data_dir reg_f
  num_classes=$(get_num_classes "${dataset}")
  data_dir=$(get_data_dir "${dataset}")
  reg_f=$(get_reg_f "${dataset}" "${arch}" "${attack}")

  mkdir -p "${outdir}"

  local extra_args=()
  if [[ "${attack}" == "refool" ]]; then
    extra_args+=(--poison_source "${source}")
  fi

  log "[Detect] dataset=${dataset}, arch=${arch}, attack=${attack}, target=${target}, ckpt=${checkpoint}"

  CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" Remove_Backdoor_FIP0.py \
    --poison-type "${attack}" \
    --arch "${arch}" \
    --checkpoint "${checkpoint}" \
    --gpuid 0 \
    --reg_F "${reg_f}" \
    --target_label "${target}" \
    --dataset "${dataset}" \
    "${NUM_CLASS_FLAG}" "${num_classes}" \
    --data-dir "${data_dir}" \
    --run_target_detection \
    --target_detect_max_per_class "${DET_MAX_PER_CLASS}" \
    --target_detect_num_views "${DET_NUM_VIEWS}" \
    --target_detect_batch_size "${DET_BATCH_SIZE}" \
    --target_detect_gamma "${DET_GAMMA}" \
    --target_detect_save_name "target_detection_result.csv" \
    --output-dir "${outdir}" \
    "${extra_args[@]}"
}

# ============================================================
# Task 1: 120 target-detection experiments
# 4 datasets × 2 archs × 2 attacks × targets
# target counts: CIFAR10=8, CIFAR100=8, GTSRB=7, IMAGENET_SUB=7
# total = (8+8+7+7) × 2 × 2 = 120
# ============================================================

run_tda120() {
  log "========== TASK 1: Start 120 target-detection experiments =========="

  local status_csv="${OUT_ROOT}/tda120_status.csv"
  local detail_csv="${OUT_ROOT}/tda120_all_results.csv"
  local summary_csv="${OUT_ROOT}/tda120_group_summary.csv"

  cat > "${status_csv}" <<CSV
tag,dataset,arch,attack,target,source,status,checkpoint,result_csv,log_file
CSV

  local ARCHS=(resnet18 resnet34)
  local ATTACKS=(refool weather)

  local CIFAR10_TARGETS=(0 1 2 3 4 5 6 7)
  local CIFAR100_TARGETS=(3 13 17 24 35 42 58 81)
  local GTSRB_TARGETS=(0 1 2 5 14 25 33)
  local IMAGENET_TARGETS=(0 1 2 3 4 5 6)

  run_one_tda() {
    local dataset="$1"
    local arch="$2"
    local attack="$3"
    local target="$4"

    local num_classes source tag train_dir detect_dir log_file checkpoint result_csv
    num_classes=$(get_num_classes "${dataset}")
    source=$(get_source "${target}" "${num_classes}")

    tag="${dataset}_${arch}_${attack}_t${target}_s${source}"
    train_dir="${TDA_MODEL_ROOT}/${dataset}/${arch}/${attack}/target_${target}_source_${source}_epoch${EPOCHS}"
    detect_dir="${TDA_DETECT_ROOT}/${dataset}/${arch}/${attack}/target_${target}_source_${source}"
    log_file="${LOG_ROOT}/tda120_${tag}.log"
    result_csv="${detect_dir}/target_detection_result.csv"

    if [[ -f "${result_csv}" ]]; then
      log "[Skip TDA] Existing result found: ${result_csv}"
      echo "${tag},${dataset},${arch},${attack},${target},${source},success,SKIP_EXISTING,${result_csv},${log_file}" >> "${status_csv}"
      return 0
    fi

    {
      log "============================================================"
      log "[TDA120] ${tag}"
      log "Model dir : ${train_dir}"
      log "Detect dir: ${detect_dir}"

      train_backdoor_model "${dataset}" "${arch}" "${attack}" "${target}" "${source}" "${num_classes}" "${train_dir}"

      checkpoint=$(find_checkpoint "${train_dir}")
      if [[ -z "${checkpoint}" ]]; then
        log "[ERROR] checkpoint not found under ${train_dir}"
        echo "${tag},${dataset},${arch},${attack},${target},${source},missing_checkpoint,,${result_csv},${log_file}" >> "${status_csv}"
        return 0
      fi

      run_target_detection "${dataset}" "${arch}" "${attack}" "${target}" "${source}" "${checkpoint}" "${detect_dir}"

      if [[ -f "${result_csv}" ]]; then
        echo "${tag},${dataset},${arch},${attack},${target},${source},success,${checkpoint},${result_csv},${log_file}" >> "${status_csv}"
        log "[SUCCESS] ${tag}"
        log "[Cleanup] Removing TDA checkpoint dir to save disk: ${train_dir}"
        rm -rf "${train_dir}"
      else
        echo "${tag},${dataset},${arch},${attack},${target},${source},no_result_csv,${checkpoint},${result_csv},${log_file}" >> "${status_csv}"
        log "[ERROR] result csv not found: ${result_csv}"
      fi
    } 2>&1 | tee "${log_file}"
  }

  for arch in "${ARCHS[@]}"; do
    for attack in "${ATTACKS[@]}"; do
      for target in "${CIFAR10_TARGETS[@]}"; do
        run_one_tda "CIFAR10" "${arch}" "${attack}" "${target}"
      done
      for target in "${CIFAR100_TARGETS[@]}"; do
        run_one_tda "CIFAR100" "${arch}" "${attack}" "${target}"
      done
      for target in "${GTSRB_TARGETS[@]}"; do
        run_one_tda "GTSRB" "${arch}" "${attack}" "${target}"
      done
      for target in "${IMAGENET_TARGETS[@]}"; do
        run_one_tda "IMAGENET_SUB" "${arch}" "${attack}" "${target}"
      done
    done
  done

  "${PYTHON_BIN}" - <<PY
import os, glob
import pandas as pd

detect_root = r"${TDA_DETECT_ROOT}"
detail_csv = r"${detail_csv}"
summary_csv = r"${summary_csv}"

files = glob.glob(os.path.join(detect_root, "**", "target_detection_result.csv"), recursive=True)
rows = []

for f in files:
    try:
        df = pd.read_csv(f)
        df["result_file"] = f
        rows.append(df)
    except Exception as e:
        print("[WARN]", f, e)

if not rows:
    print("[ERROR] No target detection results found.")
    raise SystemExit(0)

all_df = pd.concat(rows, ignore_index=True)

if "hit_top3" not in all_df.columns:
    all_df["hit_top3"] = all_df["hit_top1"]
if "top1_top2_margin" not in all_df.columns:
    all_df["top1_top2_margin"] = -1.0
if "T_target_detect_s" not in all_df.columns:
    all_df["T_target_detect_s"] = -1.0
if "top5_classes" not in all_df.columns:
    all_df["top5_classes"] = ""

all_df.to_csv(detail_csv, index=False)

group = all_df.groupby(["dataset", "arch", "attack"], as_index=False).agg(
    trials=("hit_top1", "count"),
    top1_hits=("hit_top1", "sum"),
    top3_hits=("hit_top3", "sum"),
    avg_margin=("top1_top2_margin", "mean"),
    avg_time_s=("T_target_detect_s", "mean"),
)

group["top1_tda_percent"] = group["top1_hits"] / group["trials"] * 100.0
group["top3_tda_percent"] = group["top3_hits"] / group["trials"] * 100.0

overall = pd.DataFrame([{
    "dataset": "Overall",
    "arch": "-",
    "attack": "-",
    "trials": len(all_df),
    "top1_hits": int(all_df["hit_top1"].sum()),
    "top3_hits": int(all_df["hit_top3"].sum()),
    "avg_margin": float(all_df["top1_top2_margin"].mean()),
    "avg_time_s": float(all_df["T_target_detect_s"].mean()),
    "top1_tda_percent": float(all_df["hit_top1"].mean() * 100.0),
    "top3_tda_percent": float(all_df["hit_top3"].mean() * 100.0),
}])

group = pd.concat([group, overall], ignore_index=True)
group.to_csv(summary_csv, index=False)

print("\n===== TDA120 GROUP SUMMARY =====")
print(group.to_string(index=False))
print(f"\nSaved detail : {detail_csv}")
print(f"Saved summary: {summary_csv}")
PY

  log "========== TASK 1 Done =========="
  log "[TDA models saved to] ${TDA_MODEL_ROOT}"
  log "[TDA results saved to] ${TDA_DETECT_ROOT}"
}

# ============================================================
# Task 2: Failure-impact analysis
# 6 base configs:
#   CIFAR10 / CIFAR100 / IMAGENET_SUB × resnet18 / resnet34 × Refool
# For each config:
#   1) detect pseudo target
#   2) purify with true/pseudo target
#   3) purify with wrong target = top2 non-true candidate
# ============================================================

# Known existing checkpoints. If not found, script will train a fresh Refool model under FAIL_ROOT/models/.
get_existing_refool_ckpt() {
  local dataset="$1"
  local arch="$2"

  if [[ "${dataset}" == "CIFAR10" && "${arch}" == "resnet18" ]]; then
    echo "/home/hpc/LAB-data/disk2/zzj_1024041014/zzJ1-VGG/FIP/src/saved_models/refool/CIFAR10/resnet18/model_refool.th"
  elif [[ "${dataset}" == "CIFAR10" && "${arch}" == "resnet34" ]]; then
    echo "/home/hpc/LAB-data/disk2/zzj_1024041014/zzJ1-VGG/FIP/src/saved_models/refool/CIFAR10/resnet34/model_refool.th"
  elif [[ "${dataset}" == "CIFAR100" && "${arch}" == "resnet18" ]]; then
    echo "/home/hpc/LAB-data/disk2/zzj_1024041014/zzJ1-VGG/FIP/src/saved_models/refool/CIFAR100/resnet18/model_refool.th"
  elif [[ "${dataset}" == "CIFAR100" && "${arch}" == "resnet34" ]]; then
    echo "/home/hpc/LAB-data/disk2/zzj_1024041014/zzJ1-VGG/FIP/src/saved_models/refool/CIFAR100/resnet34/model_refool.th"
  else
    echo ""
  fi
}

get_failure_target() {
  local dataset="$1"
  if [[ "${dataset}" == "CIFAR10" ]]; then
    echo 0
  elif [[ "${dataset}" == "CIFAR100" ]]; then
    echo 81
  elif [[ "${dataset}" == "IMAGENET_SUB" ]]; then
    echo 0
  fi
}

ensure_failure_checkpoint() {
  local dataset="$1"
  local arch="$2"
  local target="$3"
  local source="$4"
  local num_classes="$5"

  local existing
  existing=$(get_existing_refool_ckpt "${dataset}" "${arch}")

  if [[ -n "${existing}" && -f "${existing}" ]]; then
    echo "${existing}"
    return 0
  fi

  # Try to reuse model from TDA120 if it was trained.
  local tda_dir="${TDA_MODEL_ROOT}/${dataset}/${arch}/refool/target_${target}_source_${source}_epoch${EPOCHS}"
  local tda_ckpt
  tda_ckpt=$(find_checkpoint "${tda_dir}")
  if [[ -n "${tda_ckpt}" ]]; then
    echo "${tda_ckpt}"
    return 0
  fi

  # Otherwise train a separate failure-impact model.
  local fail_model_dir="${FAIL_ROOT}/models/${dataset}/${arch}/refool/target_${target}_source_${source}_epoch${EPOCHS}"
  train_backdoor_model "${dataset}" "${arch}" "refool" "${target}" "${source}" "${num_classes}" "${fail_model_dir}" >&2

  local ckpt
  ckpt=$(find_checkpoint "${fail_model_dir}")
  echo "${ckpt}"
}

run_purification_failure() {
  local dataset="$1"
  local arch="$2"
  local true_target="$3"
  local source="$4"
  local defense_target="$5"
  local checkpoint="$6"
  local mode="$7"
  local outdir="$8"

  local num_classes data_dir reg_f
  num_classes=$(get_num_classes "${dataset}")
  data_dir=$(get_data_dir "${dataset}")
  reg_f=$(get_reg_f "${dataset}" "${arch}" "refool")

  mkdir -p "${outdir}"

  local override_args=()
  if [[ -n "${DEFENSE_TARGET_FLAG}" ]]; then
    override_args+=("${DEFENSE_TARGET_FLAG}" "${defense_target}")
  fi

  log "[Failure Purify] dataset=${dataset}, arch=${arch}, mode=${mode}, true=${true_target}, defense=${defense_target}"

  for _d in "${outdir:-}" "${purify_dir:-}" "${run_dir:-}"; do
    if [[ -n "${_d}" && -f "${_d}/purification_summary_manual.csv" ]]; then
      log "[Skip Failure Purify] Existing summary found: ${_d}/purification_summary_manual.csv"
      return 0
    fi
  done

  CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" Remove_Backdoor_FIP0.py \
    --poison-type refool \
    --arch "${arch}" \
    --checkpoint "${checkpoint}" \
    --gpuid 0 \
    --reg_F "${reg_f}" \
    --target_label "${true_target}" \
    --poison_source "${source}" \
    --dataset "${dataset}" \
    "${NUM_CLASS_FLAG}" "${num_classes}" \
    --data-dir "${data_dir}" \
    --use_usd \
    --output-dir "${outdir}" \
    "${override_args[@]}"
}

run_failure_impact() {
  log "========== TASK 2: Start failure-impact analysis =========="

  local status_csv="${OUT_ROOT}/failure_status.csv"
  local summary_csv="${OUT_ROOT}/failure_summary.csv"
  cat > "${status_csv}" <<CSV
tag,dataset,arch,true_target,pseudo_target,wrong_target,mode,status,checkpoint,out_dir,log_file
CSV

  local DATASETS=(CIFAR10 CIFAR100 IMAGENET_SUB)
  local ARCHS=(resnet18 resnet34)

  for dataset in "${DATASETS[@]}"; do
    for arch in "${ARCHS[@]}"; do
      local num_classes target source checkpoint base_tag detect_dir detect_csv score_csv wrong_target pseudo_target
      num_classes=$(get_num_classes "${dataset}")
      target=$(get_failure_target "${dataset}")
      source=$(get_source "${target}" "${num_classes}")

      base_tag="${dataset}_${arch}_refool_t${target}_s${source}"
      log "============================================================"
      log "[Failure Base] ${base_tag}"

      checkpoint=$(ensure_failure_checkpoint "${dataset}" "${arch}" "${target}" "${source}" "${num_classes}")
      if [[ -z "${checkpoint}" || ! -f "${checkpoint}" ]]; then
        log "[ERROR] Failure checkpoint not found for ${base_tag}"
        echo "${base_tag},${dataset},${arch},${target},-1,-1,base,missing_checkpoint,,," >> "${status_csv}"
        continue
      fi

      detect_dir="${FAIL_ROOT}/${dataset}/${arch}/refool/target_${target}_source_${source}/detect"
      detect_csv="${detect_dir}/target_detection_result.csv"
      score_csv="${detect_dir}/target_detection_result_scores.csv"

      run_target_detection "${dataset}" "${arch}" "refool" "${target}" "${source}" "${checkpoint}" "${detect_dir}" \
        2>&1 | tee "${LOG_ROOT}/failure_detect_${base_tag}.log"

      if [[ ! -f "${detect_csv}" ]]; then
        log "[ERROR] Detection result not found: ${detect_csv}"
        echo "${base_tag},${dataset},${arch},${target},-1,-1,detect,no_result_csv,${checkpoint},${detect_dir},${LOG_ROOT}/failure_detect_${base_tag}.log" >> "${status_csv}"
        continue
      fi

      pseudo_target=$("${PYTHON_BIN}" - <<PY
import pandas as pd
df = pd.read_csv(r"${detect_csv}")
print(int(df.loc[0, "pred_target_top1"]))
PY
)

      if [[ -f "${score_csv}" ]]; then
        wrong_target=$("${PYTHON_BIN}" - <<PY
import pandas as pd
true_t = int("${target}")
df = pd.read_csv(r"${score_csv}").sort_values("rank")
wrong = None
for c in df["class_id"].tolist():
    c = int(c)
    if c != true_t:
        wrong = c
        break
if wrong is None:
    wrong = (true_t + 1) % int("${num_classes}")
print(wrong)
PY
)
      else
        wrong_target=$("${PYTHON_BIN}" - <<PY
import ast, pandas as pd
true_t = int("${target}")
num_classes = int("${num_classes}")
df = pd.read_csv(r"${detect_csv}")
wrong = None
if "top5_classes" in df.columns:
    try:
        top5 = ast.literal_eval(str(df.loc[0, "top5_classes"]))
        for c in top5:
            c = int(c)
            if c != true_t:
                wrong = c
                break
    except Exception:
        pass
if wrong is None:
    wrong = (true_t + 1) % num_classes
print(wrong)
PY
)
      fi

      log "[Failure Target] true=${target}, pseudo=${pseudo_target}, wrong=${wrong_target}"

      # real/pseudo target purification
      local out_real log_real
      out_real="${FAIL_ROOT}/${dataset}/${arch}/refool/target_${target}_source_${source}/purify_real_pseudo"
      log_real="${LOG_ROOT}/failure_purify_real_${base_tag}.log"

      run_purification_failure "${dataset}" "${arch}" "${target}" "${source}" "${pseudo_target}" "${checkpoint}" "real_pseudo" "${out_real}" \
        2>&1 | tee "${log_real}"

      if [[ ${PIPESTATUS[0]} -eq 0 ]]; then
        echo "${base_tag},${dataset},${arch},${target},${pseudo_target},${wrong_target},real_pseudo,success,${checkpoint},${out_real},${log_real}" >> "${status_csv}"
      else
        echo "${base_tag},${dataset},${arch},${target},${pseudo_target},${wrong_target},real_pseudo,failed,${checkpoint},${out_real},${log_real}" >> "${status_csv}"
      fi

      # wrong target purification
      local out_wrong log_wrong
      out_wrong="${FAIL_ROOT}/${dataset}/${arch}/refool/target_${target}_source_${source}/purify_wrong_${wrong_target}"
      log_wrong="${LOG_ROOT}/failure_purify_wrong_${base_tag}.log"

      run_purification_failure "${dataset}" "${arch}" "${target}" "${source}" "${wrong_target}" "${checkpoint}" "wrong" "${out_wrong}" \
        2>&1 | tee "${log_wrong}"

      if [[ ${PIPESTATUS[0]} -eq 0 ]]; then
        echo "${base_tag},${dataset},${arch},${target},${pseudo_target},${wrong_target},wrong,success,${checkpoint},${out_wrong},${log_wrong}" >> "${status_csv}"
      else
        echo "${base_tag},${dataset},${arch},${target},${pseudo_target},${wrong_target},wrong,failed,${checkpoint},${out_wrong},${log_wrong}" >> "${status_csv}"
      fi
    done
  done

  "${PYTHON_BIN}" - <<PY
import os, glob
import numpy as np
import pandas as pd

fail_root = r"${FAIL_ROOT}"
summary_csv = r"${summary_csv}"
status_csv = r"${status_csv}"

rows = []
for npz in glob.glob(os.path.join(fail_root, "**", "remove_model_*.npz"), recursive=True):
    try:
        z = np.load(npz)
        run_dir = os.path.dirname(npz)
        rows.append({
            "run_dir": run_dir,
            "npz": npz,
            "final_ASR_percent": float(z["po_acc"][-1] * 100.0),
            "final_ACC_percent": float(z["cl_test"][-1] * 100.0),
        })
    except Exception as e:
        print("[WARN]", npz, e)

df = pd.DataFrame(rows)
if os.path.exists(status_csv):
    st = pd.read_csv(status_csv)
    # attach mode by matching out_dir suffix
    if len(df) > 0:
        merged = []
        for _, r in df.iterrows():
            match = st[st["out_dir"] == r["run_dir"]]
            if len(match) > 0:
                d = {**match.iloc[0].to_dict(), **r.to_dict()}
            else:
                d = r.to_dict()
            merged.append(d)
        df = pd.DataFrame(merged)

if len(df) > 0:
    df.to_csv(summary_csv, index=False)
    print("\n===== FAILURE IMPACT SUMMARY =====")
    cols = [c for c in [
        "dataset", "arch", "true_target", "pseudo_target", "wrong_target",
        "mode", "final_ASR_percent", "final_ACC_percent", "run_dir"
    ] if c in df.columns]
    print(df[cols].to_string(index=False))
    print(f"\nSaved failure summary: {summary_csv}")
else:
    print("[WARN] No failure purification npz found.")
PY

  log "========== TASK 2 Done =========="
  log "[Failure outputs saved to] ${FAIL_ROOT}"
}

# ============================================================
# Main
# ============================================================

if [[ "${RUN_TDA120}" == "1" ]]; then
  run_tda120
else
  log "[Skip] Task 1 TDA120"
fi

if [[ "${RUN_FAILURE}" == "1" ]]; then
  run_failure_impact
else
  log "[Skip] Task 2 Failure impact"
fi

log "============================================================"
log "[ALL DONE]"
log "OUT_ROOT=${OUT_ROOT}"
log "TDA models: ${TDA_MODEL_ROOT}"
log "TDA results: ${TDA_DETECT_ROOT}"
log "Failure results: ${FAIL_ROOT}"
log "============================================================"