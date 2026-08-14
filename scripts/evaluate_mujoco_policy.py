#!/usr/bin/env python3
"""Run the deterministic P1-P6 acceptance suite in headless MuJoCo.

The script intentionally exercises the exact deployment observation builder,
policy action conversion, 2 Nm MJCF actuator limits, and configured command
delay.  Its JSON output is consumed by ``train_deployment_curriculum.sh`` as a
global phase-transition gate.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import mujoco
import numpy as np
import torch

from play_mujoco import (
    DEFAULT_JOINT_POS,
    DEFAULT_MODEL,
    JOINT_NAMES,
    LEG_ACTION_SCALE,
    LEG_LIMITS,
    OBS_JOINT_NAMES,
    POLICY_DT,
    build_observation,
    load_actor,
    reset,
)


PHASES = ("p1", "p2", "p3", "p4a", "p4b", "p4c", "p5", "p6")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--phase", choices=PHASES, required=True)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--action-delay-steps", type=int, default=4)
    return parser.parse_args()


def _profile(phase: str) -> list[tuple[float, float, float]]:
    """Return ``(duration_s, vx_mps, yaw_rate_radps)`` test segments."""
    stationary = [(3.0, 0.0, 0.0), (3.0, 0.0, 0.0), (3.0, 0.0, 0.0)]
    if phase == "p1":
        return stationary
    if phase == "p2":
        return stationary + [(3.0, 0.15, 0.0), (3.0, -0.15, 0.0), (3.0, 0.0, 0.0)]
    if phase == "p3":
        return stationary + [(2.0, 0.45, 0.0), (1.0, 0.0, 0.0), (2.0, -0.45, 0.0), (2.0, 0.0, 0.0)]

    speed = {"p4a": 0.60, "p4b": 0.80, "p4c": 1.00, "p5": 1.00, "p6": 1.00}[phase]
    profile = stationary + [
        (2.0, speed, 0.0),
        (1.0, 0.0, 0.0),
        (2.0, -speed, 0.0),
        (1.0, 0.0, 0.0),
        (2.0, 0.55 * speed, 0.0),
    ]
    if phase in {"p5", "p6"}:
        profile.extend([(2.0, 0.45, 0.50), (2.0, -0.45, -0.50), (2.0, 0.0, 0.0)])
    return profile


def _roll_pitch(data: mujoco.MjData, base_id: int) -> tuple[float, float]:
    rotation = np.asarray(data.xmat[base_id], dtype=np.float64).reshape(3, 3)
    roll = math.atan2(rotation[2, 1], rotation[2, 2])
    pitch = math.atan2(-rotation[2, 0], math.hypot(rotation[2, 1], rotation[2, 2]))
    return roll, pitch


def run(args: argparse.Namespace) -> dict[str, float | int | str]:
    if args.action_delay_steps < 0:
        raise ValueError("--action-delay-steps must be non-negative")

    torch.set_num_threads(1)
    model = mujoco.MjModel.from_xml_path(str(args.model))
    data = mujoco.MjData(model)
    joint_qpos_ids = [
        model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)] for name in JOINT_NAMES
    ]
    obs_joint_qpos_ids = [
        model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)] for name in OBS_JOINT_NAMES
    ]
    obs_joint_dof_ids = [
        model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)] for name in OBS_JOINT_NAMES
    ]
    actuator_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        for name in ("pos_servo2", "pos_servo1", "pos_servo4", "pos_servo3", "vel_wheel1", "vel_wheel2")
    ]
    if min(actuator_ids) < 0:
        raise RuntimeError("MuJoCo model is missing policy actuators")

    actor = load_actor(args.checkpoint)
    reset(model, data, joint_qpos_ids)
    desired_ctrl = data.ctrl.copy()
    desired_ctrl[actuator_ids[:4]] = DEFAULT_JOINT_POS[:4].numpy()
    desired_ctrl[actuator_ids[4:]] = 0.0
    delayed_ctrl = [desired_ctrl.copy() for _ in range(args.action_delay_steps)]
    last_action = torch.zeros(6, dtype=torch.float32)
    physics_steps_per_policy = round(POLICY_DT / model.opt.timestep)
    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")

    vx_errors: list[float] = []
    yaw_errors: list[float] = []
    stationary_speeds: list[float] = []
    min_height = float("inf")
    max_abs_roll = 0.0
    max_abs_pitch = 0.0
    falls = 0
    policy_steps = 0

    for segment_index, (duration_s, vx_cmd, yaw_cmd) in enumerate(_profile(args.phase)):
        command = torch.tensor((vx_cmd, 0.0, yaw_cmd, 0.270, 0.0), dtype=torch.float32)
        segment_steps = round(duration_s / model.opt.timestep)
        for physics_step in range(segment_steps):
            if physics_step % physics_steps_per_policy == 0:
                observation = build_observation(
                    model, data, obs_joint_qpos_ids, obs_joint_dof_ids, command, last_action
                )
                with torch.inference_mode():
                    action = actor(observation).clamp(-10.0, 10.0)
                leg_target = DEFAULT_JOINT_POS[:4] + action[:4] * LEG_ACTION_SCALE
                leg_target = torch.maximum(torch.minimum(leg_target, LEG_LIMITS[:, 1]), LEG_LIMITS[:, 0])
                desired_ctrl[actuator_ids[:4]] = leg_target.numpy()
                desired_ctrl[actuator_ids[4:]] = (action[4:] * 8.0).numpy()
                last_action = action

                base_lin_vel, base_ang_vel = _body_velocity(model, data, base_id)
                vx_errors.append(float(base_lin_vel[0] - vx_cmd))
                yaw_errors.append(float(base_ang_vel[2] - yaw_cmd))
                if abs(vx_cmd) < 1.0e-6 and abs(yaw_cmd) < 1.0e-6:
                    stationary_speeds.append(float(torch.linalg.vector_norm(base_lin_vel[:2])))
                policy_steps += 1

            delayed_ctrl.append(desired_ctrl.copy())
            data.ctrl[:] = delayed_ctrl.pop(0)
            mujoco.mj_step(model, data)
            roll, pitch = _roll_pitch(data, base_id)
            height = float(data.xpos[base_id, 2])
            min_height = min(min_height, height)
            max_abs_roll = max(max_abs_roll, abs(roll))
            max_abs_pitch = max(max_abs_pitch, abs(pitch))
            if not np.isfinite(data.qpos).all() or height < 0.15 or abs(roll) > math.radians(35.0) or abs(pitch) > math.radians(35.0):
                falls += 1
                break
        if falls:
            break

        # P1 explicitly includes repeatable recovery pushes.  Applying these
        # between zero-command segments keeps the profile deterministic.
        if args.phase == "p1" and segment_index < 2:
            data.qvel[:3] += np.array((0.10 if segment_index == 0 else -0.10, 0.04, 0.0))
            data.qvel[3:6] += np.array((0.0, 0.0, 0.12 if segment_index == 0 else -0.12))

    metrics: dict[str, float | int | str] = {
        "phase": args.phase,
        "checkpoint": str(args.checkpoint),
        "policy_steps": policy_steps,
        "falls": falls,
        "vx_rmse": float(np.sqrt(np.mean(np.square(vx_errors)))) if vx_errors else float("inf"),
        "yaw_rmse": float(np.sqrt(np.mean(np.square(yaw_errors)))) if yaw_errors else float("inf"),
        "stationary_speed_rms": float(np.sqrt(np.mean(np.square(stationary_speeds)))) if stationary_speeds else 0.0,
        "min_base_height": min_height,
        "max_abs_roll": max_abs_roll,
        "max_abs_pitch": max_abs_pitch,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n")
    return metrics


def _body_velocity(model: mujoco.MjModel, data: mujoco.MjData, body_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    velocity = np.zeros(6, dtype=np.float64)
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, body_id, velocity, 1)
    return torch.from_numpy(velocity[3:]).float(), torch.from_numpy(velocity[:3]).float()


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps(result, sort_keys=True))
