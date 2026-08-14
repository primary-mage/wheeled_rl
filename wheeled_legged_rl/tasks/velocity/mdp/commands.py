"""Custom command generators for wheeled-legged velocity tasks."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING
import os
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
class AdaptiveScalarCommandCfg(UniformScalarCommandCfg):
    """Scalar command with per-environment promotion and demotion."""

    class_type: type["AdaptiveScalarCommand"] | str = "{DIR}.commands:AdaptiveScalarCommand"

    level_ranges: tuple[tuple[float, float], ...] = MISSING
    initial_level: int = 0
    success_threshold: float = 0.015
    failure_threshold: float = 0.04
    promote_after: int = 16
    demote_after: int = 8
    min_episode_steps: int = 25
    state_path: str = ""


class AdaptiveScalarCommand(UniformScalarCommand):
    """Uniform scalar command whose range adapts independently for each environment."""

    cfg: AdaptiveScalarCommandCfg

    def __init__(self, cfg: AdaptiveScalarCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.robot = env.scene["robot"]
        self.level = torch.full(
            (self.num_envs,), cfg.initial_level, dtype=torch.long, device=self.device
        ).clamp(0, len(cfg.level_ranges) - 1)
        self._success_streak = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._failure_streak = torch.zeros_like(self._success_streak)
        self._error_sum = torch.zeros(self.num_envs, device=self.device)
        self._step_count = torch.zeros(self.num_envs, device=self.device)
        self.state_path = cfg.state_path or os.environ.get("WHEELED_RL_HEIGHT_CURRICULUM_STATE", "")
        self._load_state()
        self.metrics["level"] = self.level.float()
        self.metrics["max_level_fraction"] = torch.zeros(self.num_envs, device=self.device)

    def _load_state(self):
        if not self.state_path or not os.path.isfile(self.state_path):
            return
        try:
            state = torch.load(self.state_path, map_location=self.device)
            levels = state.get("level", state) if isinstance(state, dict) else state
            levels = torch.as_tensor(levels, device=self.device, dtype=torch.long).flatten()
            if levels.numel() == self.num_envs:
                self.level[:] = levels.clamp(0, len(self.cfg.level_ranges) - 1)
            elif levels.numel() > 0:
                self.level[:] = torch.round(levels.float().mean()).long().clamp(0, len(self.cfg.level_ranges) - 1)
        except (OSError, RuntimeError, ValueError, TypeError):
            return

    def _save_state(self):
        if not self.state_path:
            return
        os.makedirs(os.path.dirname(self.state_path) or ".", exist_ok=True)
        tmp_path = f"{self.state_path}.tmp"
        torch.save({"level": self.level.detach().cpu()}, tmp_path)
        os.replace(tmp_path, self.state_path)

    def _range_tensors(self) -> tuple[torch.Tensor, torch.Tensor]:
        ranges = torch.as_tensor(self.cfg.level_ranges, device=self.device)
        return ranges[:, 0], ranges[:, 1]

    def _update_metrics(self):
        self.metrics["mean"][:] = self.scalar_command[:, 0]
        self._error_sum += torch.abs(self.scalar_command[:, 0] - self._target_for_metrics())
        self._step_count += 1.0
        self.metrics["level"][:] = self.level.float()
        self.metrics["max_level_fraction"][:] = (self.level == len(self.cfg.level_ranges) - 1).float()

    def _target_for_metrics(self) -> torch.Tensor:
        return self.robot.data.root_pos_w.torch[:, 2]

    def _update_levels(self, env_ids: torch.Tensor):
        if env_ids.numel() == 0:
            return
        denom = self._step_count[env_ids].clamp_min(1.0)
        mean_error = self._error_sum[env_ids] / denom
        valid_episode = self._step_count[env_ids] > 0.0
        enough_steps = self._step_count[env_ids] >= self.cfg.min_episode_steps
        successes = enough_steps & (mean_error <= self.cfg.success_threshold)
        failures = valid_episode & ((~enough_steps) | (mean_error >= self.cfg.failure_threshold))

        self._success_streak[env_ids] = torch.where(
            successes, self._success_streak[env_ids] + 1, torch.zeros_like(self._success_streak[env_ids])
        )
        self._failure_streak[env_ids] = torch.where(
            failures, self._failure_streak[env_ids] + 1, torch.zeros_like(self._failure_streak[env_ids])
        )

        promote = self._success_streak[env_ids] >= self.cfg.promote_after
        demote = self._failure_streak[env_ids] >= self.cfg.demote_after
        current = self.level[env_ids]
        self.level[env_ids] = torch.where(promote, current + 1, current).clamp(0, len(self.cfg.level_ranges) - 1)
        self.level[env_ids] = torch.where(demote, self.level[env_ids] - 1, self.level[env_ids]).clamp(
            0, len(self.cfg.level_ranges) - 1
        )
        changed = promote | demote
        if torch.any(changed):
            changed_ids = env_ids[changed]
            self._success_streak[changed_ids] = 0
            self._failure_streak[changed_ids] = 0
            self._save_state()

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        if env_ids is None or isinstance(env_ids, slice):
            ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        else:
            ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        self._update_levels(ids)
        extras = super().reset(env_ids)
        self._error_sum[ids] = 0.0
        self._step_count[ids] = 0.0
        self._env.extras.setdefault("log", {})["Curriculum/height_level"] = self.level.float().mean()
        self._env.extras["log"]["Curriculum/height_max_fraction"] = (
            (self.level == len(self.cfg.level_ranges) - 1).float().mean()
        )
        return extras

    def _resample_command(self, env_ids: Sequence[int]):
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        low, high = self._range_tensors()
        self.scalar_command[ids, 0] = low[self.level[ids]] + torch.rand(len(ids), device=self.device) * (
            high[self.level[ids]] - low[self.level[ids]]
        )

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
class ManeuverVelocityCommandCfg(SmoothVelocityCommandCfg):
    """Rate-limited velocity profiles matched to the deployment command shaper.

    Unlike a uniformly resampled command, this generator deliberately mixes
    stationary holds, cruises, braking segments, and reversals.  A reversal is
    still rate-limited, so its commanded velocity always passes through zero.
    """

    class_type: type["ManeuverVelocityCommand"] | str = "{DIR}.commands:ManeuverVelocityCommand"

    # (stationary, cruise, brake-to-zero, reverse-through-zero)
    mode_weights: tuple[float, float, float, float] = (0.25, 0.35, 0.25, 0.15)
    # Each bin is (weight, min_abs_vx, max_abs_vx).  A sign is sampled evenly.
    speed_bins: tuple[tuple[float, float, float], ...] = ((1.0, 0.05, 0.25),)
    max_lateral_accel: float = 0.35
    """Clamp yaw by ``abs(vx * yaw_rate)`` to avoid infeasible high-speed turns."""


class ManeuverVelocityCommand(SmoothVelocityCommand):
    """Generate stratified, rate-limited stationary and longitudinal maneuvers."""

    cfg: ManeuverVelocityCommandCfg

    STATIONARY = 0
    CRUISE = 1
    BRAKE = 2
    REVERSE = 3

    def __init__(self, cfg: ManeuverVelocityCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        if len(cfg.mode_weights) != 4 or any(weight < 0.0 for weight in cfg.mode_weights):
            raise ValueError("mode_weights must contain four non-negative values")
        if sum(cfg.mode_weights) <= 0.0:
            raise ValueError("mode_weights must contain at least one positive value")
        if not cfg.speed_bins or any(weight < 0.0 or low < 0.0 or high < low for weight, low, high in cfg.speed_bins):
            raise ValueError("speed_bins must contain (non-negative weight, low, high) tuples")
        if sum(weight for weight, _, _ in cfg.speed_bins) <= 0.0:
            raise ValueError("speed_bins must contain at least one positive weight")
        if cfg.max_lateral_accel <= 0.0:
            raise ValueError("max_lateral_accel must be positive")
        self._mode = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.metrics["stationary_fraction"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["braking_fraction"] = torch.zeros(self.num_envs, device=self.device)

    def _sample_speed(self, count: int) -> torch.Tensor:
        bins = torch.as_tensor(self.cfg.speed_bins, device=self.device, dtype=torch.float32)
        bin_ids = torch.multinomial(bins[:, 0], count, replacement=True)
        magnitudes = bins[bin_ids, 1] + torch.rand(count, device=self.device) * (bins[bin_ids, 2] - bins[bin_ids, 1])
        signs = torch.where(torch.rand(count, device=self.device) < 0.5, -1.0, 1.0)
        return magnitudes * signs

    def _sample_yaw(self, vx: torch.Tensor) -> torch.Tensor:
        yaw = torch.empty(len(vx), device=self.device).uniform_(*self.cfg.ranges.ang_vel_z)
        yaw_limit = self.cfg.max_lateral_accel / vx.abs().clamp_min(0.05)
        return torch.clamp(yaw, -yaw_limit, yaw_limit)

    def _resample_command(self, env_ids: Sequence[int]):
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if len(ids) == 0:
            return
        weights = torch.as_tensor(self.cfg.mode_weights, device=self.device, dtype=torch.float32)
        modes = torch.multinomial(weights, len(ids), replacement=True)
        target = torch.zeros((len(ids), 3), device=self.device)
        moving = modes == self.CRUISE
        reversing = modes == self.REVERSE

        if torch.any(moving):
            moving_ids = torch.nonzero(moving, as_tuple=False).squeeze(-1)
            target[moving_ids, 0] = self._sample_speed(len(moving_ids))
            target[moving_ids, 2] = self._sample_yaw(target[moving_ids, 0])
        if torch.any(reversing):
            reversing_ids = torch.nonzero(reversing, as_tuple=False).squeeze(-1)
            speed = self._sample_speed(len(reversing_ids))
            previous_sign = torch.sign(self.target_vel_command_b[ids[reversing_ids], 0])
            has_direction = previous_sign != 0.0
            speed[has_direction] = -previous_sign[has_direction] * speed[has_direction].abs()
            target[reversing_ids, 0] = speed
            target[reversing_ids, 2] = self._sample_yaw(speed)

        # STATIONARY and BRAKE both target zero.  Their separate labels make
        # the mixture observable in logs and prevent stationary behavior from
        # becoming an accidental low-probability outcome.
        self._mode[ids] = modes
        self.is_standing_env[ids] = (modes == self.STATIONARY) | (modes == self.BRAKE)
        self.target_vel_command_b[ids] = target

    def _update_metrics(self):
        super()._update_metrics()
        self.metrics["stationary_fraction"][:] = self.is_standing_env.float()
        self.metrics["braking_fraction"][:] = (self._mode == self.BRAKE).float()

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        if env_ids is None or isinstance(env_ids, slice):
            ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        else:
            ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        extras = super().reset(env_ids)
        # Each episode begins with a real zero-speed hold.  The following
        # profile therefore trains takeoff from rest instead of teleporting the
        # command to a cruise target on the reset frame.
        self.vel_command_b[ids] = 0.0
        self.target_vel_command_b[ids] = 0.0
        self._mode[ids] = self.STATIONARY
        self.is_standing_env[ids] = True
        return extras


@configclass
class AdaptiveSmoothVelocityCommandCfg(SmoothVelocityCommandCfg):
    """Smooth velocity command with per-environment yaw difficulty levels."""

    class_type: type["AdaptiveSmoothVelocityCommand"] | str = "{DIR}.commands:AdaptiveSmoothVelocityCommand"

    level_ranges: tuple[tuple[float, float], ...] = MISSING
    initial_level: int = 0
    success_after: int = 16
    failure_after: int = 8
    min_episode_steps: int = 25
    failure_xy_threshold: float = 0.30
    failure_yaw_threshold: float = 0.60
    state_path: str = ""


class AdaptiveSmoothVelocityCommand(SmoothVelocityCommand):
    """Smooth velocity command whose yaw range adapts independently per environment."""

    cfg: AdaptiveSmoothVelocityCommandCfg

    def __init__(self, cfg: AdaptiveSmoothVelocityCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.level = torch.full(
            (self.num_envs,), cfg.initial_level, dtype=torch.long, device=self.device
        ).clamp(0, len(cfg.level_ranges) - 1)
        self._success_streak = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._failure_streak = torch.zeros_like(self._success_streak)
        self.state_path = cfg.state_path or os.environ.get("WHEELED_RL_YAW_CURRICULUM_STATE", "")
        self._load_state()
        self.metrics["level"] = self.level.float()
        self.metrics["max_level_fraction"] = torch.zeros(self.num_envs, device=self.device)

    def _load_state(self):
        if not self.state_path or not os.path.isfile(self.state_path):
            return
        try:
            state = torch.load(self.state_path, map_location=self.device)
            levels = state.get("level", state) if isinstance(state, dict) else state
            levels = torch.as_tensor(levels, device=self.device, dtype=torch.long).flatten()
            if levels.numel() == self.num_envs:
                self.level[:] = levels.clamp(0, len(self.cfg.level_ranges) - 1)
            elif levels.numel() > 0:
                self.level[:] = torch.round(levels.float().mean()).long().clamp(0, len(self.cfg.level_ranges) - 1)
        except (OSError, RuntimeError, ValueError, TypeError):
            return

    def _save_state(self):
        if not self.state_path:
            return
        os.makedirs(os.path.dirname(self.state_path) or ".", exist_ok=True)
        tmp_path = f"{self.state_path}.tmp"
        torch.save({"level": self.level.detach().cpu()}, tmp_path)
        os.replace(tmp_path, self.state_path)

    def _update_levels(self, env_ids: torch.Tensor):
        if env_ids.numel() == 0:
            return
        denom = self._step_count[env_ids].clamp_min(1.0)
        mean_xy = self._error_xy_sum[env_ids] / denom
        mean_yaw = self._error_yaw_sum[env_ids] / denom
        valid_episode = self._step_count[env_ids] > 0.0
        enough_steps = self._step_count[env_ids] >= self.cfg.min_episode_steps
        successes = enough_steps & (mean_xy < self.cfg.vel_xy_success_threshold) & (
            mean_yaw < self.cfg.vel_yaw_success_threshold
        )
        failures = valid_episode & ((~enough_steps) | (mean_xy > self.cfg.failure_xy_threshold) | (
            mean_yaw > self.cfg.failure_yaw_threshold
        ))
        self._success_streak[env_ids] = torch.where(
            successes, self._success_streak[env_ids] + 1, torch.zeros_like(self._success_streak[env_ids])
        )
        self._failure_streak[env_ids] = torch.where(
            failures, self._failure_streak[env_ids] + 1, torch.zeros_like(self._failure_streak[env_ids])
        )
        promote = self._success_streak[env_ids] >= self.cfg.success_after
        demote = self._failure_streak[env_ids] >= self.cfg.failure_after
        current = self.level[env_ids]
        self.level[env_ids] = torch.where(promote, current + 1, current).clamp(0, len(self.cfg.level_ranges) - 1)
        self.level[env_ids] = torch.where(demote, self.level[env_ids] - 1, self.level[env_ids]).clamp(
            0, len(self.cfg.level_ranges) - 1
        )
        changed = promote | demote
        if torch.any(changed):
            changed_ids = env_ids[changed]
            self._success_streak[changed_ids] = 0
            self._failure_streak[changed_ids] = 0
            self._save_state()

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        if env_ids is None or isinstance(env_ids, slice):
            ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        else:
            ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        self._update_levels(ids)
        extras = super().reset(env_ids)
        self._env.extras.setdefault("log", {})["Curriculum/yaw_level"] = self.level.float().mean()
        self._env.extras["log"]["Curriculum/yaw_max_fraction"] = (
            (self.level == len(self.cfg.level_ranges) - 1).float().mean()
        )
        return extras

    def _resample_command(self, env_ids: Sequence[int]):
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        r = torch.empty(len(ids), device=self.device)
        self.target_vel_command_b[ids, 0] = r.uniform_(*self.cfg.ranges.lin_vel_x)
        self.target_vel_command_b[ids, 1] = r.uniform_(*self.cfg.ranges.lin_vel_y)
        yaw_ranges = torch.as_tensor(self.cfg.level_ranges, device=self.device)
        yaw_low = yaw_ranges[self.level[ids], 0]
        yaw_high = yaw_ranges[self.level[ids], 1]
        self.target_vel_command_b[ids, 2] = yaw_low + torch.rand(len(ids), device=self.device) * (yaw_high - yaw_low)
        self.is_standing_env[ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_standing_envs
        self.target_vel_command_b[ids, :] *= (~self.is_standing_env[ids]).unsqueeze(1)
        first_resample = self.command_counter[ids] == 0
        if torch.any(first_resample):
            first_ids = ids[first_resample]
            self.vel_command_b[first_ids, :] = self.target_vel_command_b[first_ids, :]
