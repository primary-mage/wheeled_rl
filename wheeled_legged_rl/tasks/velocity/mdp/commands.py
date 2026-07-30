"""Custom command generators for wheeled-legged velocity tasks."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp.commands.commands_cfg import UniformVelocityCommandCfg
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils.configclass import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


@configclass
class UniformScalarCommandCfg(CommandTermCfg):
    """Configuration for uniformly sampled scalar commands."""

    class_type: type["UniformScalarCommand"] | str = "{DIR}.commands:UniformScalarCommand"

    ranges: tuple[float, float] = MISSING
    """Uniform sampling range for the scalar command."""

    element_name: str = "target"
    """Name of the scalar command element for deployment metadata."""


class UniformScalarCommand(CommandTerm):
    """Uniformly resampled scalar command with shape ``(num_envs, 1)``."""

    cfg: UniformScalarCommandCfg

    def __init__(self, cfg: UniformScalarCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.scalar_command = torch.zeros(self.num_envs, 1, device=self.device)
        self.metrics["mean"] = torch.zeros(self.num_envs, device=self.device)

        self.cfg.cmd_kind = self.cfg.cmd_kind or "command/scalar"
        self.cfg.element_names = self.cfg.element_names or [self.cfg.element_name]

    @property
    def command(self) -> torch.Tensor:
        return self.scalar_command

    def __str__(self) -> str:
        msg = "UniformScalarCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        msg += f"\tRange: {self.cfg.ranges}"
        return msg

    def _update_metrics(self):
        self.metrics["mean"][:] = self.scalar_command[:, 0]

    def _resample_command(self, env_ids: Sequence[int]):
        self.scalar_command[env_ids, 0] = torch.empty(len(env_ids), device=self.device).uniform_(*self.cfg.ranges)

    def _update_command(self):
        pass


@configclass
class SmoothScalarCommandCfg(UniformScalarCommandCfg):
    """Configuration for uniformly sampled scalar targets with rate-limited command output."""

    class_type: type["SmoothScalarCommand"] | str = "{DIR}.commands:SmoothScalarCommand"

    rate_limit: float = MISSING
    """Maximum command change rate in command units per second."""


class SmoothScalarCommand(UniformScalarCommand):
    """Uniform scalar command whose output moves gradually toward resampled targets."""

    cfg: SmoothScalarCommandCfg

    def __init__(self, cfg: SmoothScalarCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.target_scalar_command = torch.zeros_like(self.scalar_command)
        self.metrics["target_mean"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        msg = "SmoothScalarCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        msg += f"\tRange: {self.cfg.ranges}\n"
        msg += f"\tRate limit: {self.cfg.rate_limit}"
        return msg

    def _update_metrics(self):
        self.metrics["mean"][:] = self.scalar_command[:, 0]
        self.metrics["target_mean"][:] = self.target_scalar_command[:, 0]

    def _resample_command(self, env_ids: Sequence[int]):
        self.target_scalar_command[env_ids, 0] = torch.empty(len(env_ids), device=self.device).uniform_(
            *self.cfg.ranges
        )
        first_resample = self.command_counter[env_ids] == 0
        if torch.any(first_resample):
            first_env_ids = env_ids[first_resample]
            self.scalar_command[first_env_ids, 0] = self.target_scalar_command[first_env_ids, 0]

    def _update_command(self):
        max_delta = self.cfg.rate_limit * self._env.step_dt
        delta = torch.clamp(self.target_scalar_command - self.scalar_command, -max_delta, max_delta)
        self.scalar_command[:] += delta


@configclass
class SmoothVelocityCommandCfg(UniformVelocityCommandCfg):
    """Configuration for velocity targets with rate-limited command output."""

    class_type: type["SmoothVelocityCommand"] | str = "{DIR}.commands:SmoothVelocityCommand"

    rate_limits: tuple[float, float, float] = MISSING
    """Maximum command change rates for x, y, and yaw velocity commands."""


class SmoothVelocityCommand(CommandTerm):
    """Uniform velocity command whose output moves gradually toward resampled targets."""

    cfg: SmoothVelocityCommandCfg

    def __init__(self, cfg: SmoothVelocityCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.robot = env.scene[cfg.asset_name]
        self.vel_command_b = torch.zeros(self.num_envs, 3, device=self.device)
        self.target_vel_command_b = torch.zeros_like(self.vel_command_b)
        self.is_standing_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.metrics["error_vel_xy"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_vel_yaw"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["success_rate"] = torch.zeros(self.num_envs, device=self.device)
        self._error_xy_sum = torch.zeros(self.num_envs, device=self.device)
        self._error_yaw_sum = torch.zeros(self.num_envs, device=self.device)
        self._step_count = torch.zeros(self.num_envs, device=self.device)

        self.cfg.cmd_kind = self.cfg.cmd_kind or "command/body/velocity"
        self.cfg.element_names = self.cfg.element_names or ["lin_vel_x", "lin_vel_y", "ang_vel_z"]

    @property
    def command(self) -> torch.Tensor:
        return self.vel_command_b

    def __str__(self) -> str:
        msg = "SmoothVelocityCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        msg += f"\tStanding probability: {self.cfg.rel_standing_envs}\n"
        msg += f"\tRate limits: {self.cfg.rate_limits}"
        return msg

    def _update_metrics(self):
        error_xy = torch.linalg.norm(self.vel_command_b[:, :2] - self.robot.data.root_lin_vel_b.torch[:, :2], dim=-1)
        error_yaw = torch.abs(self.vel_command_b[:, 2] - self.robot.data.root_ang_vel_b.torch[:, 2])
        self._error_xy_sum += error_xy
        self._error_yaw_sum += error_yaw
        self._step_count += 1.0

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        if env_ids is None:
            env_ids = slice(None)
        denom = self._step_count[env_ids].clamp_min(1.0)
        mean_error_xy = self._error_xy_sum[env_ids] / denom
        mean_error_yaw = self._error_yaw_sum[env_ids] / denom
        self.metrics["error_vel_xy"][env_ids] = mean_error_xy
        self.metrics["error_vel_yaw"][env_ids] = mean_error_yaw
        self.metrics["success_rate"][env_ids] = (
            (mean_error_xy < self.cfg.vel_xy_success_threshold) & (mean_error_yaw < self.cfg.vel_yaw_success_threshold)
        ).float()
        extras = super().reset(env_ids)
        self._env.extras.setdefault("log", {})["Metrics/success_rate"] = extras.pop("success_rate")
        self._error_xy_sum[env_ids] = 0.0
        self._error_yaw_sum[env_ids] = 0.0
        self._step_count[env_ids] = 0.0
        return extras

    def _resample_command(self, env_ids: Sequence[int]):
        r = torch.empty(len(env_ids), device=self.device)
        self.target_vel_command_b[env_ids, 0] = r.uniform_(*self.cfg.ranges.lin_vel_x)
        self.target_vel_command_b[env_ids, 1] = r.uniform_(*self.cfg.ranges.lin_vel_y)
        self.target_vel_command_b[env_ids, 2] = r.uniform_(*self.cfg.ranges.ang_vel_z)
        self.is_standing_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_standing_envs
        self.target_vel_command_b[env_ids, :] *= (~self.is_standing_env[env_ids]).unsqueeze(1)

        first_resample = self.command_counter[env_ids] == 0
        if torch.any(first_resample):
            first_env_ids = env_ids[first_resample]
            self.vel_command_b[first_env_ids, :] = self.target_vel_command_b[first_env_ids, :]

    def _update_command(self):
        rate_limits = torch.tensor(self.cfg.rate_limits, device=self.device).unsqueeze(0)
        max_delta = rate_limits * self._env.step_dt
        delta = torch.clamp(self.target_vel_command_b - self.vel_command_b, -max_delta, max_delta)
        self.vel_command_b[:] += delta


@configclass
class HeightConditionedRollCommandCfg(UniformScalarCommandCfg):
    """Configuration for roll commands whose sampled range depends on the current height command."""

    class_type: type["HeightConditionedRollCommand"] | str = "{DIR}.commands:HeightConditionedRollCommand"

    height_command_name: str = "base_height"
    """Name of the height command term that limits the reachable roll range."""

    height_range: tuple[float, float] = (0.18, 0.36)
    """Height range used to compute the roll envelope."""

    center_height: float = 0.27
    """Height where the largest roll command is allowed."""

    center_roll_limit: float = MISSING
    """Absolute roll limit at center height."""

    edge_roll_limit: float = MISSING
    """Absolute roll limit near the low/high height edges."""


class HeightConditionedRollCommand(UniformScalarCommand):
    """Sample roll targets inside a triangular height-dependent reachable envelope."""

    cfg: HeightConditionedRollCommandCfg

    def __str__(self) -> str:
        msg = "HeightConditionedRollCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        msg += f"\tHeight range: {self.cfg.height_range}\n"
        msg += f"\tCenter height: {self.cfg.center_height}\n"
        msg += f"\tRoll limits: edge={self.cfg.edge_roll_limit}, center={self.cfg.center_roll_limit}"
        return msg

    def _roll_limit(self, env_ids: Sequence[int] | None = None) -> torch.Tensor:
        height = self._env.command_manager.get_command(self.cfg.height_command_name)[:, 0]
        if env_ids is not None:
            height = height[env_ids]

        low, high = self.cfg.height_range
        half_range = max(self.cfg.center_height - low, high - self.cfg.center_height)
        center_ratio = 1.0 - torch.abs(height - self.cfg.center_height) / half_range
        center_ratio = torch.clamp(center_ratio, 0.0, 1.0)
        return self.cfg.edge_roll_limit + (self.cfg.center_roll_limit - self.cfg.edge_roll_limit) * center_ratio

    def _resample_command(self, env_ids: Sequence[int]):
        roll_limit = self._roll_limit(env_ids)
        self.scalar_command[env_ids, 0] = (2.0 * torch.rand(len(env_ids), device=self.device) - 1.0) * roll_limit

    def _update_command(self):
        roll_limit = self._roll_limit()
        self.scalar_command[:, 0] = torch.clamp(self.scalar_command[:, 0], -roll_limit, roll_limit)
