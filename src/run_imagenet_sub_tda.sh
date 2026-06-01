#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# ImageNet-sub 20类, ResNet-18/34, Refool/Weather
# 训练后门模型 -> 运行目标检测 -> 统计 TDA
# 所有临时结果保存在 temp 目录下, 避免覆盖已有模型
# ============================================================

# ---------- 可按需修改的基础配置 ----------
GPU_ID="${GPU_ID:-0}"
PROJECT_ROOT="${PROJECT_ROOT:-/home/hpc/LAB-data/disk2/zzj_1024041014/zzJ1-VGG/FIP/src}"
DATA_ROOT="${DATA_ROOT:-/home/hpc/LAB-data/disk2/zzj_1024041014/zzJ2-ViT/patch/data/imagenet_sub_20cls}"
TEMP_ROOT="${TEMP_ROOT:-${PROJECT_ROOT}/temp/tda_imagenet_sub}"
PYTHON_BIN="${PYTHON_BIN:-python}"

# 训练轮数尽量少一些, 先用于 TDA 统计
EPOCHS="${EPOCHS:-30}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-8}"
LR="${LR:-0.01}"
SCHEDULE_1="${SCHEDULE_1:-15}"
SCHEDULE_2="${SCHEDULE_2:-24}"
SEED="${SEED:-123}"

# 目标检测参数
DET_MAX_PER_CLASS="${DET_MAX_PER_CLASS:-20}"
DET_NUM_VIEWS="${DET_NUM_VIEWS:-3}"
DET_BATCH_SIZE="${DET_BATCH_SIZE:-64}"
DET_GAMMA="${DET_GAMMA:-0.0}"

# 待统计的目标类别
# 为了控制总训练量, 默认每种攻击每个模型先跑 3 个 target
# 随机采样配置
NUM_CLASSES="${NUM_CLASSES:-20}"
NUM_TARGETS="${NUM_TARGETS:-5}"
RANDOM_SEED="${RANDOM_SEED:-123}"
SAMPLE_RECORD="${TEMP_ROOT}/sampled_targets_and_sources.csv"

ARCHS=(resnet18 resnet34)
ATTACKS=(weather refool)

# ---------- 环境准备 ----------
mkdir -p "${TEMP_ROOT}"
mkdir -p "${TEMP_ROOT}/logs"
mkdir -p "${TEMP_ROOT}/saved_models"
mkdir -p "${TEMP_ROOT}/detect_outputs"
cd "${PROJECT_ROOT}"

SUMMARY_CSV="${TEMP_ROOT}/tda_detection_summary.csv"
REPORT_CSV="${TEMP_ROOT}/tda_report.csv"

cat > "${SAMPLE_RECORD}" <<CSV
arch,attack,target_label,source_label
CSV

log() {
  echo "[$(date '+%F %T')] $*"
}

sample_targets_for_arch() {
  local arch="$1"
  python - <<PY
import random

num_classes = int("${NUM_CLASSES}")
num_targets = int("${NUM_TARGETS}")
base_seed = int("${RANDOM_SEED}")

# 不同架构使用不同种子，保证 resnet18 / resnet34 分别随机
arch = "${arch}"
arch_offset = sum(ord(c) for c in arch)
rng = random.Random(base_seed + arch_offset)

targets = rng.sample(list(range(num_classes)), num_targets)
print(" ".join(str(x) for x in targets))
PY
}

sample_refool_sources() {
  local arch="$1"
  shift
  local targets=("$@")

  python - <<PY
import random

num_classes = int("${NUM_CLASSES}")
base_seed = int("${RANDOM_SEED}")
arch = "${arch}"
targets = [int(x) for x in "${targets[*]}".split()]

arch_offset = sum(ord(c) for c in arch)
rng = random.Random(base_seed + arch_offset + 999)

sources = []
for t in targets:
    candidates = [x for x in range(num_classes) if x != t]
    s = rng.choice(candidates)
    sources.append(s)

print(" ".join(str(x) for x in sources))
PY
}

find_checkpoint() {
  local search_root="$1"
  local ckpt=""

  # 优先找常见的最佳模型名
  ckpt=$(find "${search_root}" -type f \( -name 'best_backdoor_model.pth' -o -name 'model_refool.th' -o -name 'model_weather.th' \) | sort | head -n 1 || true)
  if [[ -n "${ckpt}" ]]; then
    echo "${ckpt}"
    return 0
  fi

  # 兜底, 找任意 pth/th 文件
  ckpt=$(find "${search_root}" -type f \( -name '*.pth' -o -name '*.th' \) | sort | head -n 1 || true)
  if [[ -n "${ckpt}" ]]; then
    echo "${ckpt}"
    return 0
  fi

  return 1
}

run_train() {
  local arch="$1"
  local attack="$2"
  local target="$3"
  local source="$4"
  local outdir="$5"

  mkdir -p "${outdir}"

  if [[ "${attack}" == "weather" ]]; then
    log "Train ${arch} ${attack} target=${target}"
    CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" train_backdoor_imagenet.py \
      --data-root "${DATA_ROOT}" \
      --dataset IMAGENET_SUB \
      --arch "${arch}" \
      --batch-size "${BATCH_SIZE}" \
      --epoch "${EPOCHS}" \
      --schedule "${SCHEDULE_1}" "${SCHEDULE_2}" \
      --lr "${LR}" \
      --num_workers "${NUM_WORKERS}" \
      --poison-type weather \
      --poison-rate 0.1 \
      --poison-target "${target}" \
      --weather_effect rain \
      --weather_intensity 0.3 \
      --output-dir "${outdir}" \
      --gpuid 0 \
      --seed "${SEED}" \
      --pretrained
  else
    log "Train ${arch} ${attack} target=${target} source=${source}"
    CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" train_backdoor_imagenet.py \
      --data-root "${DATA_ROOT}" \
      --dataset IMAGENET_SUB \
      --arch "${arch}" \
      --batch-size "${BATCH_SIZE}" \
      --epoch "${EPOCHS}" \
      --schedule "${SCHEDULE_1}" "${SCHEDULE_2}" \
      --lr "${LR}" \
      --num_workers "${NUM_WORKERS}" \
      --poison-type refool \
      --poison-rate 0.2 \
      --poison-target "${target}" \
      --poison-source "${source}" \
      --refool_alpha_range 0.4,0.7 \
      --refool_gamma_range 0.9,1.1 \
      --output-dir "${outdir}" \
      --gpuid 0 \
      --seed "${SEED}" \
      --pretrained
  fi
}

run_detect() {
  local arch="$1"
  local attack="$2"
  local target="$3"
  local source="$4"
  local checkpoint="$5"
  local detect_outdir="$6"

  mkdir -p "${detect_outdir}"

  local extra_args=()
  if [[ "${attack}" == "refool" ]]; then
    extra_args+=(--poison_source "${source}")
  fi

  log "Detect ${arch} ${attack} target=${target} checkpoint=$(basename "${checkpoint}")"
  CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" Remove_Backdoor_FIP0.py \
    --poison-type "${attack}" \
    --arch "${arch}" \
    --checkpoint "${checkpoint}" \
    --gpuid 0 \
    --reg_F 0.005 \
    --target_label "${target}" \
    --dataset IMAGENET_SUB \
    --data-dir "${DATA_ROOT}" \
    --use_usd \
    --run_target_detection \
    --target_detect_max_per_class "${DET_MAX_PER_CLASS}" \
    --target_detect_num_views "${DET_NUM_VIEWS}" \
    --target_detect_batch_size "${DET_BATCH_SIZE}" \
    --target_detect_gamma "${DET_GAMMA}" \
    --output-dir "${detect_outdir}" \
    "${extra_args[@]}"
}

append_summary() {
  local arch="$1"
  local attack="$2"
  local target="$3"
  local source="$4"
  local checkpoint="$5"
  local result_csv="$6"

  local pred hit
  pred=$(python - <<PY
import pandas as pd
p = pd.read_csv(r"${result_csv}")
print(int(p.loc[0, 'pred_target_top1']))
PY
)
  hit=$(python - <<PY
import pandas as pd
p = pd.read_csv(r"${result_csv}")
print(int(p.loc[0, 'hit_top1']))
PY
)

  echo "${arch},${attack},${target},${source},${checkpoint},${pred},${hit},${result_csv}" >> "${SUMMARY_CSV}"
}

# ---------- 主流程 ----------
log "PROJECT_ROOT=${PROJECT_ROOT}"
log "DATA_ROOT=${DATA_ROOT}"
log "TEMP_ROOT=${TEMP_ROOT}"
log "GPU_ID=${GPU_ID}"
log "EPOCHS=${EPOCHS}"

for arch in "${ARCHS[@]}"; do
  # 每个架构单独随机采样 5 个 target
  read -r -a ARCH_TARGETS <<< "$(sample_targets_for_arch "${arch}")"
  read -r -a ARCH_REFOOL_SOURCES <<< "$(sample_refool_sources "${arch}" "${ARCH_TARGETS[@]}")"

  log "Sampled targets for ${arch}: ${ARCH_TARGETS[*]}"
  log "Sampled refool sources for ${arch}: ${ARCH_REFOOL_SOURCES[*]}"

  for attack in "${ATTACKS[@]}"; do
    for idx in "${!ARCH_TARGETS[@]}"; do
      target="${ARCH_TARGETS[$idx]}"
      source="-1"

      if [[ "${attack}" == "refool" ]]; then
        source="${ARCH_REFOOL_SOURCES[$idx]}"
      fi

      echo "${arch},${attack},${target},${source}" >> "${SAMPLE_RECORD}"

      train_outdir="${TEMP_ROOT}/saved_models/${arch}/${attack}/target_${target}"
      detect_outdir="${TEMP_ROOT}/detect_outputs/${arch}/${attack}/target_${target}"

      run_train "${arch}" "${attack}" "${target}" "${source}" "${train_outdir}" \
        2>&1 | tee "${TEMP_ROOT}/logs/train_${arch}_${attack}_target${target}.log"

      checkpoint="$(find_checkpoint "${train_outdir}")"
      if [[ -z "${checkpoint}" ]]; then
        log "[ERROR] No checkpoint found under ${train_outdir}"
        exit 1
      fi
      log "Found checkpoint: ${checkpoint}"

      run_detect "${arch}" "${attack}" "${target}" "${source}" "${checkpoint}" "${detect_outdir}" \
        2>&1 | tee "${TEMP_ROOT}/logs/detect_${arch}_${attack}_target${target}.log"

      result_csv="${detect_outdir}/target_detection_result.csv"
      if [[ ! -f "${result_csv}" ]]; then
        log "[ERROR] target_detection_result.csv not found in ${detect_outdir}"
        exit 1
      fi

      append_summary "${arch}" "${attack}" "${target}" "${source}" "${checkpoint}" "${result_csv}"
    done
  done
done
# ---------- 汇总 TDA ----------
python - <<PY
import pandas as pd

sampled = pd.read_csv(r"${SAMPLE_RECORD}")
summary = pd.read_csv(r"${SUMMARY_CSV}")

report = summary.groupby(["arch", "attack"], as_index=False).agg(
    num_models=("hit_top1", "count"),
    num_hits=("hit_top1", "sum")
)
report["tda_percent"] = report["num_hits"] / report["num_models"] * 100.0

report.to_csv(r"${REPORT_CSV}", index=False)

print("\n===== SAMPLED TARGETS & SOURCES =====")
print(sampled.to_string(index=False))

print("\n===== TDA SUMMARY =====")
print(report.to_string(index=False))

print(f"\nSample record saved to: ${SAMPLE_RECORD}")
print(f"Detailed summary saved to: ${SUMMARY_CSV}")
print(f"Aggregated report saved to: ${REPORT_CSV}")
PY

log "Done."
