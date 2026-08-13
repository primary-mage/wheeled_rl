#!/usr/bin/env bash
set -euo pipefail

ISAACLAB_ROOT="${ISAACLAB_ROOT:-/home/mage/IsaacLab}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-wheeled_legged_stage1}"
NUM_ENVS="${NUM_ENVS:-2048}"
VIZ="${VIZ:-kit}"
ITERATIONS="${ITERATIONS:-1200}"

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <load_run> <checkpoint>"
  echo
  echo "Example:"
  echo "  $0 2026-07-30_XX-XX-XX_stage2 model_800.pt"
  exit 1
fi

cd "${ISAACLAB_ROOT}"
PYTHONPATH="${PROJECT_ROOT}" \
./isaaclab.sh train --rl_library rsl_rl \
  --task Isaac-WheeledLegged-Stage3-v0 \
  --external_callback wheeled_legged_rl.tasks.register.register_tasks \
  --resume \
  --experiment_name "${EXPERIMENT_NAME}" \
  --load_run "$1" \
  --checkpoint "$2" \
  --run_name adaptive_stage3_height \
  --num_envs "${NUM_ENVS}" \
  --max_iterations "${ITERATIONS}" \
  --viz "${VIZ}"
