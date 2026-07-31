"""Reward terms for wheeled-legged velocity tasks."""

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import euler_xyz_from_quat, quat_apply_inverse


def track_base_height_exp(
    env,
    std: float,
    command_name: str | None = None,
    target_height: float | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward root height tracking with an exponential kernel."""
    asset = env.scene[asset_cfg.name]
    if command_name is not None:
        target = env.command_manager.get_command(command_name)[:, 0]
    elif target_height is not None:
        target = target_height
    else:
        raise ValueError("Either command_name or target_height must be provided.")
    error = torch.square(asset.data.root_pos_w.torch[:, 2] - target)
    return torch.exp(-error / std**2)


def track_base_roll_exp(
    env,
    std: float,
    command_name: str | None = None,
    target_roll: float | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward root roll tracking with an exponential kernel."""
    asset = env.scene[asset_cfg.name]
    roll, _, _ = euler_xyz_from_quat(asset.data.root_quat_w.torch)
    if command_name is not None:
        target = env.command_manager.get_command(command_name)[:, 0]
    elif target_roll is not None:
        target = target_roll
    else:
        raise ValueError("Either command_name or target_roll must be provided.")
    error = torch.square(roll - target)
    return torch.exp(-error / std**2)


def pitch_l2(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize pitch angle."""
    asset = env.scene[asset_cfg.name]
    _, pitch, _ = euler_xyz_from_quat(asset.data.root_quat_w.torch)
    return torch.square(pitch)


def lateral_lin_vel_l2(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize lateral velocity for a non-holonomic two-wheel platform."""
    asset = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_lin_vel_b.torch[:, 1])


def wheel_forward_alignment_l2(
    env,
    forward_axis: int,
    asset_cfg: SceneEntityCfg = SceneEntityCfg(
        "robot",
        body_names=["left_wheel", "right_wheel"],
        preserve_order=True,
    ),
) -> torch.Tensor:
    """Penalize fore-aft wheel offset in the robot base frame."""
    asset = env.scene[asset_cfg.name]
    wheel_pos_w = asset.data.body_pos_w.torch[:, asset_cfg.body_ids, :]
    wheel_pos_b = quat_apply_inverse(
        asset.data.root_quat_w.torch[:, None, :].expand(-1, wheel_pos_w.shape[1], -1),
        wheel_pos_w - asset.data.root_pos_w.torch[:, None, :],
    )
    return torch.square(wheel_pos_b[:, 0, forward_axis] - wheel_pos_b[:, 1, forward_axis])


def leg_joint_symmetry_l2(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg(
        "robot",
        joint_names=["servo2", "servo1", "servo4", "servo3"],
        preserve_order=True,
    ),
) -> torch.Tensor:
    """Penalize left-right leg joint mismatch in the nominally symmetric pose."""
    asset = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    return torch.square(joint_pos[:, 0] - joint_pos[:, 2]) + torch.square(joint_pos[:, 1] - joint_pos[:, 3])
