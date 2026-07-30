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
    id="Isaac-WheeledLegged-Stage3a-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.wheeled_legged_env_cfg:WheeledLeggedStage3aEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WheeledLeggedStage3aPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-WheeledLegged-Stage3b-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.wheeled_legged_env_cfg:WheeledLeggedStage3bEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WheeledLeggedStage3bPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-WheeledLegged-Stage3c-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.wheeled_legged_env_cfg:WheeledLeggedStage3cEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WheeledLeggedStage3cPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-WheeledLegged-Stage3d-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.wheeled_legged_env_cfg:WheeledLeggedStage3dEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WheeledLeggedStage3dPPORunnerCfg",
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

gym.register(
    id="Isaac-WheeledLegged-Stage4a-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.wheeled_legged_env_cfg:WheeledLeggedStage4aEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WheeledLeggedStage4aPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-WheeledLegged-Stage4b-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.wheeled_legged_env_cfg:WheeledLeggedStage4bEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WheeledLeggedStage4bPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-WheeledLegged-Stage4c-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.wheeled_legged_env_cfg:WheeledLeggedStage4cEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WheeledLeggedStage4cPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-WheeledLegged-Stage5-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.wheeled_legged_env_cfg:WheeledLeggedStage5EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WheeledLeggedStage5PPORunnerCfg",
    },
)

gym.register(
    id="Isaac-WheeledLegged-Stage5a-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.wheeled_legged_env_cfg:WheeledLeggedStage5aEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WheeledLeggedStage5aPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-WheeledLegged-Stage5b-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.wheeled_legged_env_cfg:WheeledLeggedStage5bEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WheeledLeggedStage5bPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-WheeledLegged-Stage5c-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.wheeled_legged_env_cfg:WheeledLeggedStage5cEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WheeledLeggedStage5cPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-WheeledLegged-Stage6-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.wheeled_legged_env_cfg:WheeledLeggedStage6EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WheeledLeggedStage6PPORunnerCfg",
    },
)

gym.register(
    id="Isaac-WheeledLegged-Stage6a-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.wheeled_legged_env_cfg:WheeledLeggedStage6aEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WheeledLeggedStage6aPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-WheeledLegged-Stage6b-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.wheeled_legged_env_cfg:WheeledLeggedStage6bEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WheeledLeggedStage6bPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-WheeledLegged-Stage6c-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.wheeled_legged_env_cfg:WheeledLeggedStage6cEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WheeledLeggedStage6cPPORunnerCfg",
    },
)
