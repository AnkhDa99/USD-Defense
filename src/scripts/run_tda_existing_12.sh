#!/usr/bin/env bash
set -uo pipefail

# ============================================================
# One-click pseudo-target detection for existing checkpoints
# Datasets: CIFAR10 / GTSRB / CIFAR100
# Attacks : Refool / Weather
# Archs   : ResNet18 / ResNet34
# Total   : 12 runs
# ============================================================

PROJECT_ROOT="${PROJECT_ROOT:-/home/hpc/LAB-data/disk2/zzj_1024041014/zzJ1-VGG/FIP/src}"
PYTHON_BIN="${PYTHON_BIN:-python}"

OUT_ROOT="${OUT_ROOT:-${PROJECT_ROOT}/temp/tda_existing_12}"
MANIFEST="${OUT_ROOT}/tda_manifest.csv"
STATUS_CSV="${OUT_ROOT}/run_status.csv"
DETAIL_CSV="${OUT_ROOT}/tda_all_results.csv"
SUMMARY_CSV="${OUT_ROOT}/tda_group_summary.csv"

DET_MAX_PER_CLASS="${DET_MAX_PER_CLASS:-20}"
DET_NUM_VIEWS="${DET_NUM_VIEWS:-3}"
DET_BATCH_SIZE="${DET_BATCH_SIZE:-64}"
DET_GAMMA="${DET_GAMMA:-0.0}"

mkdir -p "${OUT_ROOT}"
mkdir -p "${OUT_ROOT}/logs"
mkdir -p "${OUT_ROOT}/outputs"

cd "${PROJECT_ROOT}"

echo "[Check] PROJECT_ROOT=${PROJECT_ROOT}"
echo "[Check] OUT_ROOT=${OUT_ROOT}"
echo "[Check] PYTHON_BIN=${PYTHON_BIN}"

# ---------- 0. 语法检查 ----------
echo "[Check] Python syntax..."
"${PYTHON_BIN}" -m py_compile Remove_Backdoor_FIP0.py
if [[ $? -ne 0 ]]; then
  echo "[ERROR] Remove_Backdoor_FIP0.py has syntax error."
  exit 1
fi

# ---------- 1. 写入实验清单 ----------
cat > "${MANIFEST}" <<CSV
tag,dataset,arch,attack,target_label,poison_source,num_classes,data_dir,checkpoint,gpuid,reg_F
cifar10_refool_r18_t0,CIFAR10,resnet18,refool,0,9,10,./data,/home/hpc/LAB-data/disk2/zzj_1024041014/zzJ1-VGG/FIP/src/saved_models/refool/CIFAR10/resnet18/model_refool.th,3,0.01
cifar10_refool_r34_t0,CIFAR10,resnet34,refool,0,9,10,./data,/home/hpc/LAB-data/disk2/zzj/zzJ1-VGG/FIP/src/saved_models/refool/CIFAR10/resnet34/model_refool.th,0,0.005
gtsrb_refool_r18_t0,GTSRB,resnet18,refool,0,9,43,./data/gtsrb,/home/hpc/LAB-data/disk2/zzj_1024041014/zzJ1-VGG/FIP/src/saved_models/refool/GTSRB/resnet18/model_refool.th,2,0.005
gtsrb_refool_r34_t0,GTSRB,resnet34,refool,0,9,43,./data/gtsrb,/home/hpc/LAB-data/disk2/zzj/zzJ1-VGG/FIP/src/saved_models/refool/GTSRB/resnet34/model_refool.th,0,0.005
cifar100_refool_r18_t81,CIFAR100,resnet18,refool,81,13,100,./data,/home/hpc/LAB-data/disk2/zzj_1024041014/zzJ1-VGG/FIP/src/saved_models/refool/CIFAR100/resnet18/model_refool.th,2,0.005
cifar100_refool_r34_t81,CIFAR100,resnet34,refool,81,13,100,./data,/home/hpc/LAB-data/disk2/zzj_1024041014/zzJ1-VGG/FIP/src/saved_models/refool/CIFAR100/resnet34/model_refool.th,2,0.005
cifar10_weather_r18_t0,CIFAR10,resnet18,weather,0,-1,10,./data,/home/hpc/LAB-data/disk2/zzj_1024041014/zzJ1-VGG/FIP/src/saved_models/refool/CIFAR10/resnet18/model_weather.th,3,0.03
cifar10_weather_r34_t0,CIFAR10,resnet34,weather,0,-1,10,./data,/home/hpc/LAB-data/disk2/zzj_1024041014/zzJ1-VGG/FIP/src/saved_models/refool/CIFAR10/resnet34/model_weather.th,2,0.01
gtsrb_weather_r18_t0,GTSRB,resnet18,weather,0,-1,43,./data/gtsrb,/home/hpc/LAB-data/disk2/zzj_1024041014/zzJ1-VGG/FIP/src/saved_models/refool/GTSRB/resnet18/model_weather.th,0,0.005
gtsrb_weather_r34_t0,GTSRB,resnet34,weather,0,-1,43,./data/gtsrb,/home/hpc/LAB-data/disk2/zzj/zzJ1-VGG/FIP/src/saved_models/refool/GTSRB/resnet34/model_weather.th,0,0.005
cifar100_weather_r18_t13,CIFAR100,resnet18,weather,13,-1,100,./data,/home/hpc/LAB-data/disk2/zzj_1024041014/zzJ1-VGG/FIP/src/saved_models/refool/CIFAR100/resnet18/model_weather.th,2,0.005
cifar100_weather_r34_t13,CIFAR100,resnet34,weather,13,-1,100,./data,/home/hpc/LAB-data/disk2/zzj_1024041014/zzJ1-VGG/FIP/src/saved_models/refool/CIFAR100/resnet34/model_weather.th,2,0.005
CSV

cat > "${STATUS_CSV}" <<CSV
tag,dataset,arch,attack,target_label,checkpoint,status,result_csv,log_file
CSV

# ---------- 2. 单组运行函数 ----------
run_one() {
  local tag="$1"
  local dataset="$2"
  local arch="$3"
  local attack="$4"
  local target="$5"
  local source="$6"
  local num_classes="$7"
  local data_dir="$8"
  local checkpoint="$9"
  local gpuid="${10}"
  local reg_F="${11}"

  local out_dir="${OUT_ROOT}/outputs/${tag}"
  local log_file="${OUT_ROOT}/logs/${tag}.log"
  local result_csv="${out_dir}/target_detection_result.csv"

  mkdir -p "${out_dir}"

  echo "============================================================"
  echo "[RUN] ${tag}"
  echo "dataset=${dataset}, arch=${arch}, attack=${attack}, target=${target}"
  echo "checkpoint=${checkpoint}"
  echo "gpuid=${gpuid}"
  echo "============================================================"

  if [[ ! -f "${checkpoint}" ]]; then
    echo "[ERROR] checkpoint not found: ${checkpoint}" | tee "${log_file}"
    echo "${tag},${dataset},${arch},${attack},${target},${checkpoint},missing_checkpoint,${result_csv},${log_file}" >> "${STATUS_CSV}"
    return 0
  fi

  local extra_args=()
  if [[ "${attack}" == "refool" ]]; then
    extra_args+=(--poison_source "${source}")
  fi

  "${PYTHON_BIN}" Remove_Backdoor_FIP0.py \
    --poison-type "${attack}" \
    --arch "${arch}" \
    --checkpoint "${checkpoint}" \
    --gpuid "${gpuid}" \
    --reg_F "${reg_F}" \
    --target_label "${target}" \
    --dataset "${dataset}" \
    --num_classes "${num_classes}" \
    --data-dir "${data_dir}" \
    --run_target_detection \
    --target_detect_max_per_class "${DET_MAX_PER_CLASS}" \
    --target_detect_num_views "${DET_NUM_VIEWS}" \
    --target_detect_batch_size "${DET_BATCH_SIZE}" \
    --target_detect_gamma "${DET_GAMMA}" \
    --experiment_tag "${tag}" \
    --output-dir "${out_dir}" \
    "${extra_args[@]}" \
    2>&1 | tee "${log_file}"

  local exit_code=${PIPESTATUS[0]}

  if [[ ${exit_code} -ne 0 ]]; then
    echo "[FAILED] ${tag}, exit_code=${exit_code}"
    echo "${tag},${dataset},${arch},${attack},${target},${checkpoint},failed,${result_csv},${log_file}" >> "${STATUS_CSV}"
    return 0
  fi

  if [[ ! -f "${result_csv}" ]]; then
    echo "[FAILED] ${tag}, result csv not found."
    echo "${tag},${dataset},${arch},${attack},${target},${checkpoint},no_result_csv,${result_csv},${log_file}" >> "${STATUS_CSV}"
    return 0
  fi

  echo "[SUCCESS] ${tag}"
  echo "${tag},${dataset},${arch},${attack},${target},${checkpoint},success,${result_csv},${log_file}" >> "${STATUS_CSV}"
}

# ---------- 3. 开始批量运行 ----------
tail -n +2 "${MANIFEST}" | while IFS=',' read -r tag dataset arch attack target source num_classes data_dir checkpoint gpuid reg_F
do
  run_one "${tag}" "${dataset}" "${arch}" "${attack}" "${target}" "${source}" "${num_classes}" "${data_dir}" "${checkpoint}" "${gpuid}" "${reg_F}"
done

# ---------- 4. 汇总结果 ----------
"${PYTHON_BIN}" - <<PY
import os
import glob
import pandas as pd

out_root = r"${OUT_ROOT}"
detail_csv = r"${DETAIL_CSV}"
summary_csv = r"${SUMMARY_CSV}"
status_csv = r"${STATUS_CSV}"

files = glob.glob(os.path.join(out_root, "outputs", "*", "target_detection_result.csv"))

rows = []
for f in files:
    try:
        df = pd.read_csv(f)
        df["result_file"] = f
        rows.append(df)
    except Exception as e:
        print("[WARN] cannot read", f, e)

print("\n===== RUN STATUS =====")
if os.path.exists(status_csv):
    status = pd.read_csv(status_csv)
    print(status.to_string(index=False))
else:
    print("No status file found.")

if not rows:
    print("\n[ERROR] No target_detection_result.csv found.")
    raise SystemExit(0)

all_df = pd.concat(rows, ignore_index=True)
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

group.to_csv(summary_csv, index=False)

print("\n===== DETAIL RESULTS =====")
cols = [
    "dataset", "arch", "attack", "true_target",
    "pred_target_top1", "hit_top1", "hit_top3",
    "top1_top2_margin", "top5_classes"
]
print(all_df[cols].to_string(index=False))

print("\n===== GROUP SUMMARY =====")
print(group.to_string(index=False))

print(f"\n[Saved] detail : {detail_csv}")
print(f"[Saved] summary: {summary_csv}")
print(f"[Saved] status : {status_csv}")
PY

echo "[DONE] All target detection experiments finished."
echo "[OUT] ${OUT_ROOT}"