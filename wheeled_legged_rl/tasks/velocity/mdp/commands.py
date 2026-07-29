"""Custom command generators for wheeled-legged velocity tasks."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

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
