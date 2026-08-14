"""RSL-RL PPO settings for the P1-P6 deployment curriculum."""

from isaaclab.utils.configclass import configclass
from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


@configclass
class WheeledLeggedP1PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Shared architecture and optimizer for all sequential curriculum phases."""

    num_steps_per_env = 24
    max_iterations = 1200
    save_interval = 50
    experiment_name = "wheeled_legged_p1_stationary"
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
class WheeledLeggedP2PPORunnerCfg(WheeledLeggedP1PPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 1200
        self.experiment_name = "wheeled_legged_p2_low_speed"


@configclass
class WheeledLeggedP3PPORunnerCfg(WheeledLeggedP2PPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 1600
        self.experiment_name = "wheeled_legged_p3_transients"


@configclass
class WheeledLeggedP4APPORunnerCfg(WheeledLeggedP3PPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 1400
        self.experiment_name = "wheeled_legged_p4a_0p6mps"


@configclass
class WheeledLeggedP4BPPORunnerCfg(WheeledLeggedP4APPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 1400
        self.experiment_name = "wheeled_legged_p4b_0p8mps"


@configclass
class WheeledLeggedP4CPPORunnerCfg(WheeledLeggedP4BPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 1800
        self.experiment_name = "wheeled_legged_p4c_1p0mps"


@configclass
class WheeledLeggedP5PPORunnerCfg(WheeledLeggedP4CPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 1800
        self.experiment_name = "wheeled_legged_p5_yaw"


@configclass
class WheeledLeggedP6PPORunnerCfg(WheeledLeggedP5PPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 2400
        self.experiment_name = "wheeled_legged_p6_robust"
