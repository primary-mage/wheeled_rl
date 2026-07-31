#!/usr/bin/env python3
"""Replay an RSL-RL policy in MuJoCo without Isaac Lab."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = PROJECT_ROOT / "asset" / "wheeled_robot.xml"
DEFAULT_CHECKPOINT = PROJECT_ROOT / "logs" / "6d" / "model_5691.pt"
JOINT_NAMES = ("servo2", "servo1", "servo4", "servo3", "wheel1", "wheel2")
DEFAULT_JOINT_POS = torch.tensor((0.9, -1.9, 0.9, -1.9, 0.0, 0.0), dtype=torch.float32)
LEG_ACTION_SCALE = torch.tensor((1.6, 1.25, 1.6, 1.25), dtype=torch.float32)
LEG_LIMITS = torch.tensor(((-1.57, 1.57), (-3.14, 0.0), (-1.57, 1.57), (-3.14, 0.0)))
POLICY_DT = 0.02  # Isaac Lab: dt=0.005, decimation=4.


class Actor(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(34, 256),
            torch.nn.ELU(),
            torch.nn.Linear(256, 128),
            torch.nn.ELU(),
            torch.nn.Linear(128, 64),
            torch.nn.ELU(),
            torch.nn.Linear(64, 6),
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.mlp(observation)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--vx", type=float, default=0.0, help="Forward command in m/s.")
    parser.add_argument("--vy", type=float, default=0.0, help="Lateral command in m/s.")
    parser.add_argument("--yaw-rate", type=float, default=0.0, help="Yaw-rate command in rad/s.")
    parser.add_argument("--height", type=float, default=0.27, help="Base-height command in m.")
    parser.add_argument("--roll", type=float, default=0.0, help="Roll command in rad.")
    parser.add_argument("--duration", type=float, default=60.0, help="Replay duration in seconds.")
    parser.add_argument("--headless", action="store_true", help="Run without opening the MuJoCo viewer.")
    parser.add_argument("--realtime", action="store_true", help="Throttle the simulation to wall-clock time.")
    parser.add_argument(
        "--time-scale",
        type=float,
        default=1.0,
        help="Wall-clock playback speed when --realtime is enabled (0.25 = quarter speed).",
    )
    parser.add_argument("--render-fps", type=float, default=60.0, help="Viewer refresh rate; independent of physics rate.")
    parser.add_argument("--torch-threads", type=int, default=1, help="CPU threads used by the small policy network.")
    parser.add_argument(
        "--action-delay-steps",
        type=int,
        default=4,
        help="Actuator command delay in MuJoCo physics steps; Isaac training samples 0-4 steps.",
    )
    return parser.parse_args()


def load_actor(checkpoint_path: Path) -> Actor:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    actor = Actor()
    # RSL-RL checkpoints also store the Gaussian exploration standard deviation.
    actor_state = {name: value for name, value in checkpoint["actor_state_dict"].items() if name.startswith("mlp.")}
    actor.load_state_dict(actor_state)
    actor.eval()
    return actor


def body_velocity(model: mujoco.MjModel, data: mujoco.MjData, body_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    velocity = torch.zeros(6, dtype=torch.float64).numpy()
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, body_id, velocity, 1)
    # MuJoCo returns angular velocity followed by linear velocity in the requested local frame.
    return torch.from_numpy(velocity[3:]).float(), torch.from_numpy(velocity[:3]).float()


def projected_gravity(data: mujoco.MjData) -> torch.Tensor:
    # Rotate the unit world gravity vector into the root body frame.
    rotation = torch.tensor(data.xmat[1], dtype=torch.float32).reshape(3, 3)
    return rotation.T @ torch.tensor((0.0, 0.0, -1.0))


def build_observation(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    joint_qpos_ids: list[int],
    joint_dof_ids: list[int],
    command: torch.Tensor,
    last_action: torch.Tensor,
) -> torch.Tensor:
    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    base_lin_vel, base_ang_vel = body_velocity(model, data, base_id)
    joint_pos = torch.tensor(data.qpos[joint_qpos_ids], dtype=torch.float32) - DEFAULT_JOINT_POS
    joint_vel = torch.tensor(data.qvel[joint_dof_ids], dtype=torch.float32)
    root_z = torch.tensor((data.xpos[base_id, 2],), dtype=torch.float32)
    root_rotation = torch.tensor(data.xmat[base_id], dtype=torch.float32).reshape(3, 3)
    roll = torch.atan2(root_rotation[2, 1], root_rotation[2, 2]).reshape(1)
    observation = torch.cat((
        base_lin_vel,
        base_ang_vel,
        projected_gravity(data),
        command[:3],
        command[3:4],
        command[4:5],
        root_z,
        roll,
        joint_pos,
        joint_vel,
        last_action,
    ))
    if observation.shape != (34,):
        raise RuntimeError(f"Expected a 34D policy observation, got {tuple(observation.shape)}")
    return observation


def reset(model: mujoco.MjModel, data: mujoco.MjData, joint_qpos_ids: list[int]) -> None:
    mujoco.mj_resetData(model, data)
    data.qpos[:7] = (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
    data.qpos[2] = 0.27
    data.qpos[joint_qpos_ids] = DEFAULT_JOINT_POS.numpy()
    mujoco.mj_forward(data.model, data)


def run(args: argparse.Namespace) -> None:
    if args.render_fps <= 0:
        raise ValueError("--render-fps must be positive")
    if args.time_scale <= 0:
        raise ValueError("--time-scale must be positive")
    if args.action_delay_steps < 0:
        raise ValueError("--action-delay-steps must be non-negative")
    torch.set_num_threads(args.torch_threads)
    model = mujoco.MjModel.from_xml_path(str(args.model))
    data = mujoco.MjData(model)
    joint_qpos_ids = [model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)] for name in JOINT_NAMES]
    joint_dof_ids = [model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)] for name in JOINT_NAMES]
    actuator_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in (
        "pos_servo2", "pos_servo1", "pos_servo4", "pos_servo3", "vel_wheel1", "vel_wheel2"
    )]
    if min(actuator_ids) < 0:
        raise RuntimeError("MuJoCo model is missing one or more policy actuators")

    actor = load_actor(args.checkpoint)
    command = torch.tensor((args.vx, args.vy, args.yaw_rate, args.height, args.roll), dtype=torch.float32)
    last_action = torch.zeros(6, dtype=torch.float32)
    reset(model, data, joint_qpos_ids)
    print(
        f"Loaded {args.checkpoint.name}; physics dt={model.opt.timestep:.3f}s, "
        f"policy dt={POLICY_DT:.3f}s, action delay={args.action_delay_steps} physics steps, "
        f"torch threads={torch.get_num_threads()}"
    )

    desired_ctrl = data.ctrl.copy()
    desired_ctrl[actuator_ids[:4]] = DEFAULT_JOINT_POS[:4].numpy()
    desired_ctrl[actuator_ids[4:]] = 0.0
    delayed_ctrl = [desired_ctrl.copy() for _ in range(args.action_delay_steps)]

    def step_policy() -> None:
        nonlocal last_action
        observation = build_observation(model, data, joint_qpos_ids, joint_dof_ids, command, last_action)
        inference_start = time.perf_counter()
        with torch.inference_mode():
            action = actor(observation).clamp(-10.0, 10.0)
        inference_times.append(time.perf_counter() - inference_start)
        leg_target = DEFAULT_JOINT_POS[:4] + action[:4] * LEG_ACTION_SCALE
        leg_target = torch.maximum(torch.minimum(leg_target, LEG_LIMITS[:, 1]), LEG_LIMITS[:, 0])
        desired_ctrl[actuator_ids[:4]] = leg_target.numpy()
        desired_ctrl[actuator_ids[4:]] = (action[4:] * 8.0).numpy()
        last_action = action

    physics_steps_per_policy = round(POLICY_DT / model.opt.timestep)
    total_steps = round(args.duration / model.opt.timestep)
    viewer = None
    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    min_height = float("inf")
    max_height = float("-inf")
    inference_times: list[float] = []
    max_realtime_lag = 0.0
    if not args.headless:
        from mujoco import viewer as mujoco_viewer

        viewer = mujoco_viewer.launch_passive(model, data)
    sim_start = data.time
    wall_start = time.perf_counter()
    next_render_time = data.time
    try:
        for physics_step in range(total_steps):
            if physics_step % physics_steps_per_policy == 0:
                step_policy()
            delayed_ctrl.append(desired_ctrl.copy())
            data.ctrl[:] = delayed_ctrl.pop(0)
            mujoco.mj_step(model, data)
            base_height = float(data.xpos[base_id, 2])
            min_height = min(min_height, base_height)
            max_height = max(max_height, base_height)
            if not torch.isfinite(torch.tensor(data.qpos)).all():
                raise RuntimeError("MuJoCo state became non-finite")
            # Rendering at every 2 ms physics step forces simulation down to display refresh rate.
            if viewer is not None and data.time >= next_render_time:
                viewer.sync()
                next_render_time = data.time + 1.0 / args.render_fps
            if args.realtime:
                deadline = wall_start + (data.time - sim_start) / args.time_scale
                remaining = deadline - time.perf_counter()
                if remaining > 0:
                    time.sleep(remaining)
                else:
                    max_realtime_lag = max(max_realtime_lag, -remaining)
    finally:
        if viewer is not None:
            viewer.close()
    wall_elapsed = time.perf_counter() - wall_start
    print(
        f"Completed {args.duration:.1f}s; base height range: "
        f"{min_height:.3f} to {max_height:.3f} m; final height: {data.xpos[base_id, 2]:.3f} m; "
        f"wall time: {wall_elapsed:.3f}s"
    )
    sorted_inference = sorted(inference_times)
    mean_inference = sum(sorted_inference) / len(sorted_inference)
    p99_inference = sorted_inference[round(0.99 * (len(sorted_inference) - 1))]
    print(
        f"Policy inference: mean={mean_inference * 1e3:.3f} ms, p99={p99_inference * 1e3:.3f} ms; "
        f"max real-time lag={max_realtime_lag * 1e3:.3f} ms"
    )


if __name__ == "__main__":
    run(parse_args())
