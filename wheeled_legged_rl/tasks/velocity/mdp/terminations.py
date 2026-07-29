"""Termination terms for wheeled-legged velocity tasks."""

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import euler_xyz_from_quat


def root_height_out_of_bounds(
    env,
    bounds: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate when root height leaves the configured interval."""
    asset = env.scene[asset_cfg.name]
    height = asset.data.root_pos_w.torch[:, 2]
    return torch.logical_or(height < bounds[0], height > bounds[1])


def root_orientation_out_of_bounds(
    env,
    roll_limit: float,
    pitch_limit: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate when roll or pitch becomes too large."""
    asset = env.scene[asset_cfg.name]
    roll, pitch, _ = euler_xyz_from_quat(asset.data.root_quat_w.torch)
    return torch.logical_or(torch.abs(roll) > roll_limit, torch.abs(pitch) > pitch_limit)

