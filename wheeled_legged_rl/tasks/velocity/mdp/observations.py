"""Observation terms for wheeled-legged velocity tasks."""

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import euler_xyz_from_quat


def base_height(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Root height above the world frame."""
    asset = env.scene[asset_cfg.name]
    return asset.data.root_pos_w.torch[:, 2:3]


def base_roll(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Root roll angle in radians."""
    asset = env.scene[asset_cfg.name]
    roll, _, _ = euler_xyz_from_quat(asset.data.root_quat_w.torch)
    return roll.unsqueeze(-1)


def constant_command(env, value: float) -> torch.Tensor:
    """A scalar command term used for staged fixed height/roll targets."""
    return torch.full((env.num_envs, 1), value, device=env.device)

