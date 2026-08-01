"""Termination terms for wheeled-legged velocity tasks."""

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import euler_xyz_from_quat, quat_apply_inverse


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


def wheel_forward_offset_too_large(
    env,
    max_offset: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg(
        "robot",
        body_names=["left_wheel", "right_wheel"],
        preserve_order=True,
    ),
) -> torch.Tensor:
    """Terminate when the wheel centers form a large fore-aft scissor offset."""
    asset = env.scene[asset_cfg.name]
    wheel_pos_w = asset.data.body_pos_w.torch[:, asset_cfg.body_ids, :]
    wheel_pos_b = quat_apply_inverse(
        asset.data.root_quat_w.torch[:, None, :].expand(-1, wheel_pos_w.shape[1], -1),
        wheel_pos_w - asset.data.root_pos_w.torch[:, None, :],
    )
    return torch.abs(wheel_pos_b[:, 0, 0] - wheel_pos_b[:, 1, 0]) > max_offset


def leg_or_foot_contact(env, force_threshold: float, sensor_names: tuple[str, ...]) -> torch.Tensor:
    """Terminate when any configured leg or foot contact sensor exceeds the force threshold."""
    in_contact = []
    for sensor_name in sensor_names:
        contact_sensor = env.scene.sensors[sensor_name]
        force_history = contact_sensor.data.net_forces_w_history
        max_force = torch.linalg.vector_norm(force_history, dim=-1).amax(dim=(1, 2))
        in_contact.append(max_force > force_threshold)
    return torch.stack(in_contact, dim=1).any(dim=1)
