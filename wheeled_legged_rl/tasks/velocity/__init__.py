"""Velocity, height, and roll tracking tasks for the wheeled-legged robot."""

import gymnasium as gym

from . import agents


gym.register(
    id="Isaac-WheeledLegged-Stage1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.wheeled_legged_env_cfg:WheeledLeggedStage1EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WheeledLeggedStage1PPORunnerCfg",
    },
)

gym.register(
    id="Isaac-WheeledLegged-Stage1-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.wheeled_legged_env_cfg:WheeledLeggedStage1EnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WheeledLeggedStage1PPORunnerCfg",
    },
)

gym.register(
    id="Isaac-WheeledLegged-Stage2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.wheeled_legged_env_cfg:WheeledLeggedStage2EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WheeledLeggedStage2PPORunnerCfg",
    },
)

gym.register(
    id="Isaac-WheeledLegged-Stage3-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.wheeled_legged_env_cfg:WheeledLeggedStage3EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WheeledLeggedStage3PPORunnerCfg",
    },
)

gym.register(
    id="Isaac-WheeledLegged-Stage4-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.wheeled_legged_env_cfg:WheeledLeggedStage4EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WheeledLeggedStage4PPORunnerCfg",
    },
)

