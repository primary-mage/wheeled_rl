"""RSL-RL PPO configs for the wheeled-legged robot stages."""

from isaaclab.utils.configclass import configclass
from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


@configclass
class WheeledLeggedStage1PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 500
    save_interval = 50
    experiment_name = "wheeled_legged_stage1"
    actor = RslRlMLPModelCfg(
        hidden_dims=[256, 128, 64],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.8),
    )
    critic = RslRlMLPModelCfg(hidden_dims=[256, 128, 64], activation="elu", obs_normalization=False)
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class WheeledLeggedStage2PPORunnerCfg(WheeledLeggedStage1PPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 800
        self.experiment_name = "wheeled_legged_stage2"

@configclass
class WheeledLeggedStage3PPORunnerCfg(WheeledLeggedStage2PPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 400
        self.experiment_name = "wheeled_legged_stage3_stance"


@configclass
class WheeledLeggedStage4PPORunnerCfg(WheeledLeggedStage3PPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 1200
        self.experiment_name = "wheeled_legged_stage4"


@configclass
class WheeledLeggedStage5PPORunnerCfg(WheeledLeggedStage4PPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 1200
        self.experiment_name = "wheeled_legged_stage5"


@configclass
class WheeledLeggedStage6PPORunnerCfg(WheeledLeggedStage5PPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 1200
        self.experiment_name = "wheeled_legged_stage6"
