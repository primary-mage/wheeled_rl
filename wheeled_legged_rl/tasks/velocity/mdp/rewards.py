"""Reward terms for wheeled-legged velocity tasks."""

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import euler_xyz_from_quat


def track_base_height_exp(
    env,
    target_height: float,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward root height tracking with an exponential kernel."""
    asset = env.scene[asset_cfg.name]
    error = torch.square(asset.data.root_pos_w.torch[:, 2] - target_height)
    return torch.exp(-error / std**2)


def track_base_roll_exp(
    env,
    target_roll: float,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward root roll tracking with an exponential kernel."""
    asset = env.scene[asset_cfg.name]
    roll, _, _ = euler_xyz_from_quat(asset.data.root_quat_w.torch)
    error = torch.square(roll - target_roll)
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

