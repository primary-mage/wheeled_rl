"""Custom reset events for the wheeled-legged robot tasks."""

from isaaclab.managers import SceneEntityCfg


def set_joint_targets_to_default(env, env_ids, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """Set selected joint position and velocity targets to their configured defaults."""
    asset = env.scene[asset_cfg.name]

    iter_env_ids = env_ids[:, None] if asset_cfg.joint_ids != slice(None) else env_ids
    joint_pos = asset.data.default_joint_pos.torch[iter_env_ids, asset_cfg.joint_ids].clone()
    joint_vel = asset.data.default_joint_vel.torch[iter_env_ids, asset_cfg.joint_ids].clone()

    asset.set_joint_position_target_index(target=joint_pos, joint_ids=asset_cfg.joint_ids, env_ids=env_ids)
    asset.set_joint_velocity_target_index(target=joint_vel, joint_ids=asset_cfg.joint_ids, env_ids=env_ids)
