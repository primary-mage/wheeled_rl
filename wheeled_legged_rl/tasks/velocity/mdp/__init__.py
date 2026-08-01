"""Custom MDP terms plus the Isaac Lab standard terms used by this task."""

from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg, JointVelocityActionCfg
from isaaclab.envs.mdp.commands.commands_cfg import UniformVelocityCommandCfg
from isaaclab.envs.mdp.events import (
    push_by_setting_velocity,
    randomize_rigid_body_com,
    randomize_rigid_body_mass,
    randomize_rigid_body_material,
    reset_joints_by_offset,
    reset_root_state_uniform,
)
from isaaclab.envs.mdp.observations import (
    base_ang_vel,
    base_lin_vel,
    generated_commands,
    joint_pos_rel,
    joint_vel_rel,
    last_action,
    projected_gravity,
)
from isaaclab.envs.mdp.rewards import (
    action_rate_l2,
    ang_vel_xy_l2,
    joint_acc_l2,
    joint_pos_limits,
    joint_torques_l2,
    lin_vel_z_l2,
    track_ang_vel_z_exp,
    track_lin_vel_xy_exp,
)
from isaaclab.envs.mdp.terminations import illegal_contact, time_out

from .commands import HeightConditionedRollCommandCfg, SmoothScalarCommandCfg, SmoothVelocityCommandCfg, UniformScalarCommandCfg
from .events import StaticStanceWrenchPulse, set_joint_targets_to_default
from .observations import base_height, base_roll, constant_command
from .rewards import leg_joint_symmetry_l2, lateral_lin_vel_l2, pitch_l2, track_base_height_exp, track_base_roll_exp, wheel_forward_alignment_l2
from .terminations import root_height_out_of_bounds, root_orientation_out_of_bounds, wheel_forward_offset_too_large

__all__ = [
    "JointPositionActionCfg",
    "HeightConditionedRollCommandCfg",
    "SmoothScalarCommandCfg",
    "SmoothVelocityCommandCfg",
    "UniformScalarCommandCfg",
    "JointVelocityActionCfg",
    "UniformVelocityCommandCfg",
    "action_rate_l2",
    "ang_vel_xy_l2",
    "base_ang_vel",
    "base_height",
    "base_lin_vel",
    "base_roll",
    "constant_command",
    "generated_commands",
    "joint_acc_l2",
    "joint_pos_limits",
    "joint_pos_rel",
    "joint_torques_l2",
    "joint_vel_rel",
    "leg_joint_symmetry_l2",
    "illegal_contact",
    "last_action",
    "lateral_lin_vel_l2",
    "lin_vel_z_l2",
    "pitch_l2",
    "projected_gravity",
    "push_by_setting_velocity",
    "randomize_rigid_body_material",
    "randomize_rigid_body_com",
    "randomize_rigid_body_mass",
    "reset_joints_by_offset",
    "reset_root_state_uniform",
    "root_height_out_of_bounds",
    "root_orientation_out_of_bounds",
    "wheel_forward_offset_too_large",
    "set_joint_targets_to_default",
    "StaticStanceWrenchPulse",
    "time_out",
    "track_ang_vel_z_exp",
    "track_base_height_exp",
    "track_base_roll_exp",
    "track_lin_vel_xy_exp",
    "wheel_forward_alignment_l2",
]
