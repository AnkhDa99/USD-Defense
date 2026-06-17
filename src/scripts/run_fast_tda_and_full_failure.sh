#!/usr/bin/env bash
set -uo pipefail

# ============================================================
# Fast TDA + Full Failure Impact
#
# Phase 1:
#   120 target-detection experiments with shorter training epochs.
#
# Phase 2:
#   Failure-impact analysis with full training epochs.
#
# All outputs are saved under temp/overnight_fast_<timestamp>/
# No original checkpoint under saved_models/refool/ will be overwritten.
# ============================================================

PROJECT_ROOT="${PROJECT_ROOT:-/home/hpc/LAB-data/disk2/zzj_1024041014/zzJ1-VGG/FIP/src}"
GPU_ID="${GPU_ID:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"

# 目标检测训练轮数：快
TDA_EPOCHS="${TDA_EPOCHS:-70}"

# 失败影响分析训练轮数：完整
FAILURE_EPOCHS="${FAILURE_EPOCHS:-170}"

# ImageNet-Sub 数据路径
IMAGENET_DATA_ROOT="${IMAGENET_DATA_ROOT:-/home/hpc/LAB-data/disk2/zzj_1024041014/zzJ2-ViT/patch/data/imagenet_sub_20cls}"

STAMP="${STAMP:-$(date '+%Y%m%d_%H%M%S')}"
BASE_OUT_ROOT="${BASE_OUT_ROOT:-${PROJECT_ROOT}/temp/overnight_fast_${STAMP}}"

TDA_OUT_ROOT="${BASE_OUT_ROOT}/phase1_tda120_e${TDA_EPOCHS}"
FAIL_OUT_ROOT="${BASE_OUT_ROOT}/phase2_failure_e${FAILURE_EPOCHS}"

MAIN_SCRIPT="${PROJECT_ROOT}/scripts/run_overnight_tda120_and_failure.sh"

cd "${PROJECT_ROOT}"

echo "============================================================"
echo "[Fast Overnight Wrapper]"
echo "PROJECT_ROOT       = ${PROJECT_ROOT}"
echo "GPU_ID             = ${GPU_ID}"
echo "TDA_EPOCHS         = ${TDA_EPOCHS}"
echo "FAILURE_EPOCHS     = ${FAILURE_EPOCHS}"
echo "BASE_OUT_ROOT      = ${BASE_OUT_ROOT}"
echo "TDA_OUT_ROOT       = ${TDA_OUT_ROOT}"
echo "FAIL_OUT_ROOT      = ${FAIL_OUT_ROOT}"
echo "IMAGENET_DATA_ROOT = ${IMAGENET_DATA_ROOT}"
echo "============================================================"

if [[ ! -f "${MAIN_SCRIPT}" ]]; then
  echo "[ERROR] Cannot find ${MAIN_SCRIPT}"
  exit 1
fi

# -----------------------------
# Phase 1: fast target detection
# -----------------------------
echo ""
echo "============================================================"
echo "[PHASE 1] Start TDA120 with EPOCHS=${TDA_EPOCHS}"
echo "============================================================"

GPU_ID="${GPU_ID}" \
PYTHON_BIN="${PYTHON_BIN}" \
IMAGENET_DATA_ROOT="${IMAGENET_DATA_ROOT}" \
EPOCHS="${TDA_EPOCHS}" \
RUN_TDA120=1 \
RUN_FAILURE=0 \
OUT_ROOT="${TDA_OUT_ROOT}" \
bash "${MAIN_SCRIPT}"

echo ""
echo "============================================================"
echo "[PHASE 1 DONE]"
echo "TDA output root: ${TDA_OUT_ROOT}"
echo "============================================================"

TDA_COUNT=$(find "${TDA_OUT_ROOT}/detect_outputs_tda120" -name "target_detection_result.csv" 2>/dev/null | wc -l)
echo "[CHECK] TDA result count = ${TDA_COUNT}/120"

if [[ "${TDA_COUNT}" -lt 120 ]]; then
  echo "[ERROR] TDA120 is incomplete. Stop before failure-impact analysis."
  echo "[ERROR] Please fix TDA phase first."
  exit 2
fi

# -----------------------------
# Phase 2: full failure impact
# -----------------------------
echo ""
echo "============================================================"
echo "[PHASE 2] Start Failure Impact with EPOCHS=${FAILURE_EPOCHS}"
echo "============================================================"

GPU_ID="${GPU_ID}" \
PYTHON_BIN="${PYTHON_BIN}" \
IMAGENET_DATA_ROOT="${IMAGENET_DATA_ROOT}" \
EPOCHS="${FAILURE_EPOCHS}" \
RUN_TDA120=0 \
RUN_FAILURE=1 \
OUT_ROOT="${FAIL_OUT_ROOT}" \
bash "${MAIN_SCRIPT}"

echo ""
echo "============================================================"
echo "[ALL DONE]"
echo "BASE_OUT_ROOT  = ${BASE_OUT_ROOT}"
echo "TDA_OUT_ROOT   = ${TDA_OUT_ROOT}"
echo "FAIL_OUT_ROOT  = ${FAIL_OUT_ROOT}"
echo "============================================================"