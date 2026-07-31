"""Custom reset and disturbance events for the wheeled-legged robot tasks."""

import torch

from isaaclab.managers import ManagerTermBase, SceneEntityCfg


def set_joint_targets_to_default(env, env_ids, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """Set selected joint position and velocity targets to their configured defaults."""
    asset = env.scene[asset_cfg.name]

    iter_env_ids = env_ids[:, None] if asset_cfg.joint_ids != slice(None) else env_ids
    joint_pos = asset.data.default_joint_pos.torch[iter_env_ids, asset_cfg.joint_ids].clone()
    joint_vel = asset.data.default_joint_vel.torch[iter_env_ids, asset_cfg.joint_ids].clone()

    asset.set_joint_position_target_index(target=joint_pos, joint_ids=asset_cfg.joint_ids, env_ids=env_ids)
    asset.set_joint_velocity_target_index(target=joint_vel, joint_ids=asset_cfg.joint_ids, env_ids=env_ids)


class StaticStanceWrenchPulse(ManagerTermBase):
    """Apply sparse, short horizontal force or yaw-torque pulses to the base body."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.asset_cfg = cfg.params["asset_cfg"]
        self.asset = env.scene[self.asset_cfg.name]
        self._pulse_remaining = torch.zeros(env.scene.num_envs, dtype=torch.long, device=env.device)
        self._wait_remaining = torch.full((env.scene.num_envs,), -1, dtype=torch.long, device=env.device)
        self._forces = torch.zeros(env.scene.num_envs, len(self.asset_cfg.body_ids), 3, device=env.device)
        self._torques = torch.zeros_like(self._forces)

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self._env.scene.num_envs, device=self._env.device, dtype=torch.long)
        else:
            env_ids = env_ids.to(device=self._env.device, dtype=torch.long)
        self._pulse_remaining[env_ids] = 0
        self._wait_remaining[env_ids] = -1
        self._forces[env_ids] = 0.0
        self._torques[env_ids] = 0.0
        self.asset.set_external_force_and_torque(
            forces=self._forces[env_ids],
            torques=self._torques[env_ids],
            body_ids=self.asset_cfg.body_ids,
            env_ids=env_ids,
        )
        return {}

    def __call__(
        self,
        env,
        env_ids,
        asset_cfg: SceneEntityCfg,
        force_range: tuple[float, float],
        yaw_torque_range: tuple[float, float],
        pulse_duration_s: tuple[float, float],
        wait_time_s: tuple[float, float],
    ):
        if env_ids is None:
            env_ids = torch.arange(env.scene.num_envs, device=env.device, dtype=torch.long)
        else:
            env_ids = env_ids.to(device=env.device, dtype=torch.long)

        pulse_steps = torch.ceil(
            torch.empty(len(env_ids), device=env.device).uniform_(*pulse_duration_s) / env.step_dt
        ).to(torch.long)
        wait_steps = torch.ceil(
            torch.empty(len(env_ids), device=env.device).uniform_(*wait_time_s) / env.step_dt
        ).to(torch.long)

        uninitialized = self._wait_remaining[env_ids] < 0
        self._wait_remaining[env_ids[uninitialized]] = wait_steps[uninitialized]
        waiting = self._pulse_remaining[env_ids] == 0
        self._wait_remaining[env_ids[waiting]] -= 1
        starts = waiting & (self._wait_remaining[env_ids] <= 0)
        start_ids = env_ids[starts]
        if len(start_ids) > 0:
            self._forces[start_ids] = 0.0
            self._torques[start_ids] = 0.0
            directions = torch.randint(3, (len(start_ids),), device=env.device)
            magnitudes = torch.empty(len(start_ids), device=env.device).uniform_(*force_range)
            yaw_torques = torch.empty(len(start_ids), device=env.device).uniform_(*yaw_torque_range)
            self._forces[start_ids[directions == 0], 0, 0] = magnitudes[directions == 0]
            self._forces[start_ids[directions == 1], 0, 1] = magnitudes[directions == 1]
            self._torques[start_ids[directions == 2], 0, 2] = yaw_torques[directions == 2]
            self._pulse_remaining[start_ids] = pulse_steps[starts]

        active = self._pulse_remaining[env_ids] > 0
        inactive_ids = env_ids[~active]
        self._forces[inactive_ids] = 0.0
        self._torques[inactive_ids] = 0.0
        self.asset.set_external_force_and_torque(
            forces=self._forces[env_ids],
            torques=self._torques[env_ids],
            body_ids=asset_cfg.body_ids,
            env_ids=env_ids,
        )
        self._pulse_remaining[env_ids[active]] -= 1
        ended_ids = env_ids[(self._pulse_remaining[env_ids] == 0) & active]
        if len(ended_ids) > 0:
            self._wait_remaining[ended_ids] = wait_steps[(self._pulse_remaining[env_ids] == 0) & active]
