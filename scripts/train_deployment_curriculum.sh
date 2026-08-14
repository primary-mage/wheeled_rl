#!/usr/bin/env bash
set -euo pipefail

# Train the P1-P6 curriculum sequentially.  Promotion is global: the current
# checkpoint must pass the finite MuJoCo acceptance suite after a training
# chunk before it can seed the next task.  There is no per-environment command
# difficulty promotion or hidden curriculum state.

ISAACLAB_ROOT="${ISAACLAB_ROOT:-/home/mage/IsaacLab}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-wheeled_legged_deployment}"
SESSION_NAME="${SESSION_NAME:-p1_p6_deployment}"
NUM_ENVS="${NUM_ENVS:-4096}"
VIZ="${VIZ:-kit}"
CHUNK_ITERATIONS="${CHUNK_ITERATIONS:-400}"
MAX_CHUNKS_PER_PHASE="${MAX_CHUNKS_PER_PHASE:-16}"
REQUIRED_PASS_CHECKS="${REQUIRED_PASS_CHECKS:-2}"
EVAL_PYTHON="${EVAL_PYTHON:-python}"
RESET_SESSION="${RESET_SESSION:-0}"

LOG_ROOT="${ISAACLAB_ROOT}/logs/rsl_rl/${EXPERIMENT_NAME}"
STATE_DIR="${AUTO_STATE_DIR:-${PROJECT_ROOT}/logs/${SESSION_NAME}_auto}"
MANIFEST="${STATE_DIR}/resume.env"
COMPLETED_MARKER="${STATE_DIR}/completed"

PHASES=(p1 p2 p3 p4a p4b p4c p5 p6)
TASKS=(
  Isaac-WheeledLegged-P1-v0
  Isaac-WheeledLegged-P2-v0
  Isaac-WheeledLegged-P3-v0
  Isaac-WheeledLegged-P4A-v0
  Isaac-WheeledLegged-P4B-v0
  Isaac-WheeledLegged-P4C-v0
  Isaac-WheeledLegged-P5-v0
  Isaac-WheeledLegged-P6-v0
)

if [[ ! -x "${ISAACLAB_ROOT}/isaaclab.sh" ]]; then
  echo "Isaac Lab launcher not found: ${ISAACLAB_ROOT}/isaaclab.sh" >&2
  exit 1
fi
if ! "${EVAL_PYTHON}" -c 'import mujoco, torch' >/dev/null 2>&1; then
  echo "${EVAL_PYTHON} must provide both mujoco and torch for the promotion gate." >&2
  echo "Set EVAL_PYTHON to the deployment/MuJoCo Python environment." >&2
  exit 1
fi

mkdir -p "${STATE_DIR}" "${LOG_ROOT}"
if [[ "${RESET_SESSION}" == "1" ]]; then
  rm -f -- "${MANIFEST}" "${COMPLETED_MARKER}" "${STATE_DIR}"/p*_gate_*.json
fi
if [[ -f "${COMPLETED_MARKER}" ]]; then
  echo "P1-P6 session ${SESSION_NAME} is already complete."
  exit 0
fi

CURRENT_INDEX=0
LOAD_RUN=""
CHECKPOINT=""
if [[ -f "${MANIFEST}" ]]; then
  # shellcheck disable=SC1090
  source "${MANIFEST}"
fi

write_manifest() {
  printf 'CURRENT_INDEX=%q\nLOAD_RUN=%q\nCHECKPOINT=%q\n' \
    "${CURRENT_INDEX}" "${LOAD_RUN}" "${CHECKPOINT}" > "${MANIFEST}"
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
  local phase="$1"
  local task="$2"
  local chunk="$3"
  local run_name="${SESSION_NAME}_${phase}_chunk_${chunk}"
  local output_dir
  local -a train_args=(
    train --rl_library rsl_rl
    --task "${task}"
    --external_callback wheeled_legged_rl.tasks.register.register_tasks
    --experiment_name "${EXPERIMENT_NAME}"
    --run_name "${run_name}"
    --num_envs "${NUM_ENVS}"
    --max_iterations "${CHUNK_ITERATIONS}"
    --viz "${VIZ}"
  )

  if [[ -n "${LOAD_RUN}" ]]; then
    train_args+=(--resume --load_run "${LOAD_RUN}" --checkpoint "${CHECKPOINT}")
  fi

  echo
  echo "============================================================"
  echo "phase=${phase} task=${task} chunk=${chunk}"
  if [[ -n "${LOAD_RUN}" ]]; then
    echo "resume=${LOAD_RUN}/${CHECKPOINT}"
  else
    echo "starting P1 from random initialization"
  fi
  echo "============================================================"

  cd "${ISAACLAB_ROOT}"
  PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
    ./isaaclab.sh "${train_args[@]}"

  output_dir="$(latest_run_dir "${run_name}")"
  if [[ -z "${output_dir}" ]]; then
    echo "No output directory found for ${run_name}" >&2
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

evaluate_gate() {
  local phase="$1"
  local chunk="$2"
  local metrics_path="${STATE_DIR}/${phase}_gate_${chunk}.json"
  local checkpoint_path="${LOG_ROOT}/${LOAD_RUN}/${CHECKPOINT}"

  "${EVAL_PYTHON}" "${PROJECT_ROOT}/scripts/evaluate_mujoco_policy.py" \
    --checkpoint "${checkpoint_path}" \
    --phase "${phase}" \
    --output "${metrics_path}"
  "${EVAL_PYTHON}" "${PROJECT_ROOT}/scripts/check_policy_gate.py" \
    "${metrics_path}" --phase "${phase}"
}

for ((index = CURRENT_INDEX; index < ${#PHASES[@]}; index += 1)); do
  phase="${PHASES[index]}"
  task="${TASKS[index]}"
  pass_checks=0
  for ((chunk = 1; chunk <= MAX_CHUNKS_PER_PHASE; chunk += 1)); do
    run_chunk "${phase}" "${task}" "${chunk}"
    if evaluate_gate "${phase}" "${chunk}"; then
      pass_checks=$((pass_checks + 1))
      echo "${phase} promotion gate passed (${pass_checks}/${REQUIRED_PASS_CHECKS})."
      if ((pass_checks >= REQUIRED_PASS_CHECKS)); then
        CURRENT_INDEX=$((index + 1))
        write_manifest
        break
      fi
    else
      pass_checks=0
      echo "${phase} remains active; continuing from its latest checkpoint."
    fi
  done
  if ((CURRENT_INDEX != index + 1)); then
    echo "${phase} failed its promotion gate after ${MAX_CHUNKS_PER_PHASE} chunks." >&2
    exit 1
  fi
done

date -Is > "${COMPLETED_MARKER}"
echo "P1-P6 curriculum complete."
echo "Final checkpoint: ${LOG_ROOT}/${LOAD_RUN}/${CHECKPOINT}"
