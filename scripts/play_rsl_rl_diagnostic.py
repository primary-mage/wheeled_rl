#!/usr/bin/env python3
"""Run Isaac Lab RSL-RL play while exporting one CSV row per environment step."""

from __future__ import annotations

import csv
import os
import runpy
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ISAACLAB_ROOT = Path(os.environ.get("ISAACLAB_ROOT", "/home/mage/IsaacLab")).expanduser()
ISAACLAB_PLAY = ISAACLAB_ROOT / "scripts/reinforcement_learning/rsl_rl/play.py"
LOG_PATH = Path(os.environ.get("WHEELED_RL_DIAGNOSTIC_LOG", "isaac_play_diagnostic.csv")).expanduser()
JOINT_NAMES = ("servo2", "servo1", "servo4", "servo3", "wheel1", "wheel2")


def _array(value):
    """Convert Isaac Lab tensor wrappers to a CPU NumPy array without assumptions about device."""
    value = getattr(value, "torch", value)
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return value


def _first_env(value):
    value = _array(value)
    if value is None:
        return []
    if getattr(value, "ndim", 0) > 1:
        value = value[0]
    return value.reshape(-1).tolist()


def _policy_obs(value):
    if isinstance(value, dict):
        value = value.get("policy", next(iter(value.values())))
    return value


class DiagnosticLogger:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.file = path.open("w", newline="")
        self.writer = None
        self.env = None
        self.obs = None
        self.step_index = 0

    def close(self):
        self.file.close()

    def set_env(self, env):
        self.env = env

    def set_obs(self, obs):
        self.obs = obs

    def _state(self):
        robot = self.env.scene["robot"]
        data = robot.data
        state = {
            "root_pos_w": _first_env(data.root_pos_w),
            "root_quat_w": _first_env(data.root_quat_w),
            "root_lin_vel_b": _first_env(getattr(data, "root_lin_vel_b", None)),
            "root_ang_vel_b": _first_env(getattr(data, "root_ang_vel_b", None)),
            "joint_pos": [],
            "joint_vel": [],
            "applied_torque": _first_env(getattr(data, "applied_torque", None)),
            "net_forces_w": _first_env(
                getattr(data, "net_forces_w", getattr(data, "contact_forces_w", None))
            ),
        }
        joint_pos = _array(data.joint_pos)
        joint_vel = _array(data.joint_vel)
        for name in JOINT_NAMES:
            joint_ids = robot.find_joints(name, preserve_order=True)[0]
            state["joint_pos"].append(_first_env(joint_pos[:, joint_ids]))
            state["joint_vel"].append(_first_env(joint_vel[:, joint_ids]))
        state["joint_pos"] = [item[0] if item else float("nan") for item in state["joint_pos"]]
        state["joint_vel"] = [item[0] if item else float("nan") for item in state["joint_vel"]]
        return state

    def write_step(self, action):
        if self.env is None:
            return
        state = self._state()
        obs = _first_env(_policy_obs(self.obs))
        action = _first_env(action)
        if self.writer is None:
            state_columns = [
                *(f"obs_{i}" for i in range(len(obs))),
                *(f"action_{i}" for i in range(len(action))),
                *(f"root_pos_w_{i}" for i in range(len(state["root_pos_w"]))),
                *(f"root_quat_w_{i}" for i in range(len(state["root_quat_w"]))),
                *(f"root_lin_vel_b_{i}" for i in range(len(state["root_lin_vel_b"]))),
                *(f"root_ang_vel_b_{i}" for i in range(len(state["root_ang_vel_b"]))),
                *(f"joint_pos_{name}" for name in JOINT_NAMES),
                *(f"joint_vel_{name}" for name in JOINT_NAMES),
                *(f"applied_torque_{i}" for i in range(len(state["applied_torque"]))),
                *(f"net_force_w_{i}" for i in range(len(state["net_forces_w"]))),
            ]
            self.writer = csv.writer(self.file)
            self.writer.writerow(["step", "sim_time", *state_columns])
        row = [
            self.step_index,
            (self.step_index + 1) * float(self.env.step_dt),
            *obs,
            *action,
            *state["root_pos_w"],
            *state["root_quat_w"],
            *state["root_lin_vel_b"],
            *state["root_ang_vel_b"],
            *state["joint_pos"],
            *state["joint_vel"],
            *state["applied_torque"],
            *state["net_forces_w"],
        ]
        self.writer.writerow(row)
        self.file.flush()
        self.step_index += 1


def main():
    if not ISAACLAB_PLAY.exists():
        raise FileNotFoundError(f"Isaac Lab play.py not found: {ISAACLAB_PLAY}")
    sys.path.insert(0, str(PROJECT_ROOT))
    sys.path.insert(0, str(ISAACLAB_PLAY.parent))

    from isaaclab.envs import ManagerBasedRLEnv

    logger = DiagnosticLogger(LOG_PATH)
    original_reset = ManagerBasedRLEnv.reset
    original_step = ManagerBasedRLEnv.step

    def reset(env, *args, **kwargs):
        result = original_reset(env, *args, **kwargs)
        logger.set_env(env)
        logger.set_obs(_policy_obs(result[0] if isinstance(result, tuple) else result))
        return result

    def step(env, action):
        logger.set_env(env)
        result = original_step(env, action)
        logger.write_step(action)
        logger.set_obs(_policy_obs(result[0] if isinstance(result, tuple) else result))
        return result

    ManagerBasedRLEnv.reset = reset
    ManagerBasedRLEnv.step = step
    try:
        import wheeled_legged_rl.tasks.velocity  # noqa: F401

        runpy.run_path(str(ISAACLAB_PLAY), run_name="__main__")
    finally:
        ManagerBasedRLEnv.reset = original_reset
        ManagerBasedRLEnv.step = original_step
        logger.close()
        print(f"Diagnostic CSV written to: {LOG_PATH}")


if __name__ == "__main__":
    main()
