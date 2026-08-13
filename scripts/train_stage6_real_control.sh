#!/usr/bin/env bash
set -euo pipefail

ISAACLAB_ROOT="${ISAACLAB_ROOT:-/home/mage/IsaacLab}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-wheeled_legged_stage1}"
NUM_ENVS="${NUM_ENVS:-2048}"
VIZ="${VIZ:-kit}"
ITER_PER_STAGE="${ITER_PER_STAGE:-1200}"

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <load_run> <checkpoint>"
  echo
  echo "Example:"
  echo "  $0 2026-07-30_XX-XX-XX_adaptive_stage5_yaw_from_stage3 model_4497.pt"
  echo
  echo "Optional environment variables:"
  echo "  NUM_ENVS=2048 VIZ=kit ITER_PER_STAGE=1200 EXPERIMENT_NAME=wheeled_legged_stage1"
  exit 1
fi

LOAD_RUN="$1"
CHECKPOINT="$2"
LOG_ROOT="${ISAACLAB_ROOT}/logs/rsl_rl/${EXPERIMENT_NAME}"

latest_checkpoint() {
  local run_dir="$1"
  find "${run_dir}" -maxdepth 1 -type f -name 'model_*.pt' -printf '%f\n' \
    | sort -V \
    | tail -n 1
}

latest_run_dir() {
  local run_name="$1"
  find "${LOG_ROOT}" -maxdepth 1 -type d -name "*_${run_name}" \
    | sort \
    | tail -n 1
}

train_stage() {
  local task="$1"
  local run_name="$2"
  local load_run="$3"
  local checkpoint="$4"
  local new_run_dir
  local new_checkpoint

  echo
  echo "============================================================"
  echo "Training ${task}"
  echo "  load_run:    ${load_run}"
  echo "  checkpoint:  ${checkpoint}"
  echo "  run_name:    ${run_name}"
  echo "  extra iters: ${ITER_PER_STAGE}"
  echo "============================================================"

  cd "${ISAACLAB_ROOT}"
  PYTHONPATH="${PROJECT_ROOT}" \
  ./isaaclab.sh train --rl_library rsl_rl \
    --task "${task}" \
    --external_callback wheeled_legged_rl.tasks.register.register_tasks \
    --resume \
    --experiment_name "${EXPERIMENT_NAME}" \
    --load_run "${load_run}" \
    --checkpoint "${checkpoint}" \
    --run_name "${run_name}" \
    --num_envs "${NUM_ENVS}" \
    --max_iterations "${ITER_PER_STAGE}" \
    --viz "${VIZ}"

  new_run_dir="$(latest_run_dir "${run_name}")"
  if [[ -z "${new_run_dir}" ]]; then
    echo "Could not find output run directory for run_name=${run_name}" >&2
    exit 1
  fi

  new_checkpoint="$(latest_checkpoint "${new_run_dir}")"
  if [[ -z "${new_checkpoint}" ]]; then
    echo "Could not find checkpoint in ${new_run_dir}" >&2
    exit 1
  fi

  LOAD_RUN="$(basename "${new_run_dir}")"
  CHECKPOINT="${new_checkpoint}"
  echo "Next resume point: ${LOAD_RUN}/${CHECKPOINT}"
}

train_stage "Isaac-WheeledLegged-Stage6-v0" "adaptive_stage6_real_control_from_stage5" "${LOAD_RUN}" "${CHECKPOINT}"

echo
echo "Stage6 adaptive real-control group finished."
echo "Final checkpoint: ${LOG_ROOT}/${LOAD_RUN}/${CHECKPOINT}"
