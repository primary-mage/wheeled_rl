#!/usr/bin/env bash
set -euo pipefail

# Resume Stage 5 and Stage 6 from a Stage 3 checkpoint. Each stage is trained
# in chunks until its per-environment yaw curriculum has converged.
ISAACLAB_ROOT="${ISAACLAB_ROOT:-/home/mage/IsaacLab}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-wheeled_legged_stage1}"
NUM_ENVS="${NUM_ENVS:-2048}"
VIZ="${VIZ:-kit}"
CHUNK_ITERATIONS="${CHUNK_ITERATIONS:-300}"
MAX_CHUNKS="${MAX_CHUNKS:-20}"
MIN_MAX_LEVEL_FRACTION="${MIN_MAX_LEVEL_FRACTION:-0.90}"
REQUIRED_READY_CHECKS="${REQUIRED_READY_CHECKS:-2}"
SESSION_NAME="${SESSION_NAME:-stage56_from_stage3_contact_2nm}"
RESET_CURRICULUM_STATE="${RESET_CURRICULUM_STATE:-0}"

SOURCE_CHECKPOINT="${1:-${PROJECT_ROOT}/logs/stage3_stance/model_399_stage3_contact_2nm.pt}"
LOG_ROOT="${ISAACLAB_ROOT}/logs/rsl_rl/${EXPERIMENT_NAME}"
STATE_DIR="${AUTO_STATE_DIR:-${PROJECT_ROOT}/logs/${SESSION_NAME}_auto}"
MANIFEST="${STATE_DIR}/resume.env"
COMPLETED_MARKER="${STATE_DIR}/completed"
STAGE5_STATE="${STATE_DIR}/stage5_yaw_levels.pt"
STAGE6_STATE="${STATE_DIR}/stage6_yaw_levels.pt"

if [[ "${SOURCE_CHECKPOINT}" != /* ]]; then
  SOURCE_CHECKPOINT="$(cd -- "$(dirname -- "${SOURCE_CHECKPOINT}")" && pwd)/$(basename -- "${SOURCE_CHECKPOINT}")"
fi

if [[ ! -x "${ISAACLAB_ROOT}/isaaclab.sh" ]]; then
  echo "Isaac Lab launcher not found: ${ISAACLAB_ROOT}/isaaclab.sh" >&2
  echo "Set ISAACLAB_ROOT to the Isaac Lab installation directory." >&2
  exit 1
fi

if [[ ! -f "${SOURCE_CHECKPOINT}" ]]; then
  echo "Stage 3 checkpoint not found: ${SOURCE_CHECKPOINT}" >&2
  exit 1
fi

mkdir -p "${STATE_DIR}" "${LOG_ROOT}"
if [[ "${RESET_CURRICULUM_STATE}" == "1" ]]; then
  rm -f -- "${MANIFEST}" "${COMPLETED_MARKER}" "${STAGE5_STATE}" "${STAGE6_STATE}"
fi

if [[ -f "${COMPLETED_MARKER}" ]]; then
  echo "Stage 5 and Stage 6 are already complete for session ${SESSION_NAME}."
  exit 0
fi

write_manifest() {
  printf 'LOAD_RUN=%q\nCHECKPOINT=%q\n' "${LOAD_RUN}" "${CHECKPOINT}" > "${MANIFEST}"
}

initialize_resume_point() {
  if [[ -f "${MANIFEST}" ]]; then
    # shellcheck disable=SC1090
    source "${MANIFEST}"
    return
  fi

  local bootstrap_dir
  bootstrap_dir="${LOG_ROOT}/bootstrap_${SESSION_NAME}"
  mkdir -p "${bootstrap_dir}"
  CHECKPOINT="$(basename "${SOURCE_CHECKPOINT}")"
  ln -sfn "${SOURCE_CHECKPOINT}" "${bootstrap_dir}/${CHECKPOINT}"
  LOAD_RUN="$(basename "${bootstrap_dir}")"
  write_manifest
}

latest_run_dir() {
  local run_name="$1"
  find "${LOG_ROOT}" -maxdepth 1 -type d -name "*_${run_name}" -printf '%T@ %p\n' \
    | sort -n \
    | tail -n 1 \
    | cut -d' ' -f2-
}

latest_checkpoint() {
  local run_dir="$1"
  find "${run_dir}" -maxdepth 1 -type f -name 'model_*.pt' -printf '%f\n' \
    | sort -V \
    | tail -n 1
}

run_chunk() {
  local task="$1"
  local run_name="$2"
  local state_path="$3"
  local output_dir

  echo
  echo "============================================================"
  echo "Training ${task}"
  echo "  resume point: ${LOAD_RUN}/${CHECKPOINT}"
  echo "  chunk size:   ${CHUNK_ITERATIONS} iterations"
  echo "============================================================"

  cd "${ISAACLAB_ROOT}"
  WHEELED_RL_YAW_CURRICULUM_STATE="${state_path}" \
  PYTHONPATH="${PROJECT_ROOT}" \
  ./isaaclab.sh train --rl_library rsl_rl \
    --task "${task}" \
    --external_callback wheeled_legged_rl.tasks.register.register_tasks \
    --resume \
    --experiment_name "${EXPERIMENT_NAME}" \
    --load_run "${LOAD_RUN}" \
    --checkpoint "${CHECKPOINT}" \
    --run_name "${run_name}" \
    --num_envs "${NUM_ENVS}" \
    --max_iterations "${CHUNK_ITERATIONS}" \
    --viz "${VIZ}"

  output_dir="$(latest_run_dir "${run_name}")"
  if [[ -z "${output_dir}" ]]; then
    echo "No output run directory found for ${run_name}" >&2
    exit 1
  fi
  CHECKPOINT="$(latest_checkpoint "${output_dir}")"
  if [[ -z "${CHECKPOINT}" ]]; then
    echo "No checkpoint found in ${output_dir}" >&2
    exit 1
  fi
  LOAD_RUN="$(basename "${output_dir}")"
  write_manifest
}

curriculum_is_ready() {
  local state_path="$1"
  "${ISAACLAB_ROOT}/isaaclab.sh" -p "${PROJECT_ROOT}/scripts/check_curriculum_state.py" \
    "${state_path}" \
    --expected-num-envs "${NUM_ENVS}" \
    --min-max-fraction "${MIN_MAX_LEVEL_FRACTION}"
}

train_until_ready() {
  local task="$1"
  local run_name="$2"
  local state_path="$3"
  local ready_checks=0
  local chunk

  for ((chunk = 1; chunk <= MAX_CHUNKS; chunk += 1)); do
    run_chunk "${task}" "${run_name}" "${state_path}"
    if curriculum_is_ready "${state_path}"; then
      ready_checks=$((ready_checks + 1))
      echo "Curriculum ready check ${ready_checks}/${REQUIRED_READY_CHECKS} passed."
      if ((ready_checks >= REQUIRED_READY_CHECKS)); then
        return 0
      fi
    else
      ready_checks=0
      echo "Curriculum not complete; continuing ${task}."
    fi
  done

  echo "${task} did not satisfy the completion threshold after ${MAX_CHUNKS} chunks." >&2
  exit 1
}

initialize_resume_point
train_until_ready "Isaac-WheeledLegged-Stage5-v0" "${SESSION_NAME}_stage5" "${STAGE5_STATE}"
train_until_ready "Isaac-WheeledLegged-Stage6-v0" "${SESSION_NAME}_stage6" "${STAGE6_STATE}"

echo
echo "Stage 5 and Stage 6 training completed."
echo "Final checkpoint: ${LOG_ROOT}/${LOAD_RUN}/${CHECKPOINT}"
date -Is > "${COMPLETED_MARKER}"
