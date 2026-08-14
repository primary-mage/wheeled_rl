"""P1-P6 deployment curriculum tasks for the wheeled-legged robot."""

import gymnasium as gym

from . import agents


def _register(task_id: str, env_cfg: str, runner_cfg: str) -> None:
    gym.register(
        id=task_id,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.wheeled_legged_env_cfg:{env_cfg}",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:{runner_cfg}",
        },
    )


_register("Isaac-WheeledLegged-P1-v0", "WheeledLeggedP1EnvCfg", "WheeledLeggedP1PPORunnerCfg")
_register("Isaac-WheeledLegged-P2-v0", "WheeledLeggedP2EnvCfg", "WheeledLeggedP2PPORunnerCfg")
_register("Isaac-WheeledLegged-P3-v0", "WheeledLeggedP3EnvCfg", "WheeledLeggedP3PPORunnerCfg")
_register("Isaac-WheeledLegged-P4A-v0", "WheeledLeggedP4AEnvCfg", "WheeledLeggedP4APPORunnerCfg")
_register("Isaac-WheeledLegged-P4B-v0", "WheeledLeggedP4BEnvCfg", "WheeledLeggedP4BPPORunnerCfg")
_register("Isaac-WheeledLegged-P4C-v0", "WheeledLeggedP4CEnvCfg", "WheeledLeggedP4CPPORunnerCfg")
_register("Isaac-WheeledLegged-P5-v0", "WheeledLeggedP5EnvCfg", "WheeledLeggedP5PPORunnerCfg")
_register("Isaac-WheeledLegged-P6-v0", "WheeledLeggedP6EnvCfg", "WheeledLeggedP6PPORunnerCfg")
_register("Isaac-WheeledLegged-P6-Play-v0", "WheeledLeggedP6EnvCfg_PLAY", "WheeledLeggedP6PPORunnerCfg")
