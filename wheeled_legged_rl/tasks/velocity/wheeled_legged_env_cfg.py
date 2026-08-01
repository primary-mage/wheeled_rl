"""Manager-based velocity/height/roll training configs for the wheeled-legged robot."""

import math
from pathlib import Path

from isaaclab_physx.physics import PhysxCfg

import isaaclab.sim as sim_utils
from isaaclab.actuators import DelayedPDActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

import wheeled_legged_rl.tasks.velocity.mdp as mdp


PROJECT_ROOT = Path(__file__).resolve().parents[3]
USD_PATH = str(PROJECT_ROOT / "asset" / "usd" / "wheeled_robot" / "wheeled_robot.usda")
NOMINAL_HEIGHT = 0.270
NOMINAL_ROLL = 0.0
STAGE3A_HEIGHT_COMMAND_RANGE = (0.24, 0.30)
STAGE3B_HEIGHT_COMMAND_RANGE = (0.22, 0.32)
STAGE3C_HEIGHT_COMMAND_RANGE = (0.20, 0.34)
STAGE3D_HEIGHT_COMMAND_RANGE = (0.18, 0.36)
HEIGHT_COMMAND_RANGE = STAGE3D_HEIGHT_COMMAND_RANGE
STAGE4A_ROLL_COMMAND_RANGE = (-math.radians(10.0), math.radians(10.0))
STAGE4B_ROLL_COMMAND_RANGE = (-math.radians(20.0), math.radians(20.0))
STAGE4C_ROLL_COMMAND_RANGE = (-math.radians(30.0), math.radians(30.0))
STAGE4C_EDGE_ROLL_LIMIT = math.radians(8.0)
STAGE5A_YAW_RATE_RANGE = (-0.4, 0.4)
STAGE5B_YAW_RATE_RANGE = (-0.8, 0.8)
STAGE5C_YAW_RATE_RANGE = (-1.2, 1.2)
STAGE6_COMMAND_RESAMPLING_TIME = (2.0, 4.0)
STAGE6_VELOCITY_RATE_LIMITS = (0.7, 0.7, 1.2)
STAGE6_HEIGHT_RATE_LIMIT = 0.06
STAGE6_ROLL_RATE_LIMIT = math.radians(20.0)
STAGE6_ROLL_DEMO_RANGE = (-math.radians(10.0), math.radians(10.0))
ROLL_COMMAND_RANGE = (-math.radians(30.0), math.radians(30.0))
STAGE3_SMOOTH_LEG_ACTION_SCALE = {
    "servo2": 1.60,
    "servo1": 1.25,
    "servo4": 1.60,
    "servo3": 1.25,
}


WHEELED_LEGGED_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=USD_PATH,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=100.0,
            max_angular_velocity=100.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
        activate_contact_sensors=True,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, NOMINAL_HEIGHT),
        joint_pos={
            "servo2": 0.9,
            "servo1": -1.9,
            "servo4": 0.9,
            "servo3": -1.9,
            "wheel1": 0.0,
            "wheel2": 0.0,
        },
    ),
    actuators={
        "leg_position": DelayedPDActuatorCfg(
            joint_names_expr=["servo.*"],
            effort_limit_sim=2.0,
            velocity_limit_sim=10.0,
            stiffness=10.0,
            damping=2.0,
            armature=0.01,
            min_delay=0,
            max_delay=4,
        ),
        "wheel_velocity": DelayedPDActuatorCfg(
            joint_names_expr=["wheel.*"],
            effort_limit=2.0,
            effort_limit_sim=2.0,
            velocity_limit_sim=20.0,
            stiffness=0.0,
            damping=2.0,
            armature=0.01,
            min_delay=0,
            max_delay=4,
        ),
    },
)


@configclass
class WheeledLeggedSceneCfg(InteractiveSceneCfg):
    """Flat training scene for the wheeled-legged robot."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=0.8,
        ),
        debug_vis=False,
    )
    robot: ArticulationCfg = WHEELED_LEGGED_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    leg_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Geometry/base_link/.*",
        update_period=0.0,
        history_length=1,
    )
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=1000.0),
    )


@configclass
class CommandsCfg:
    """Command ranges for base velocity, height, and roll targets."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(8.0, 12.0),
        rel_standing_envs=0.05,
        rel_heading_envs=0.0,
        heading_command=False,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.3, 0.3),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(0.0, 0.0),
            heading=(0.0, 0.0),
        ),
    )
    base_height = mdp.UniformScalarCommandCfg(
        resampling_time_range=(8.0, 12.0),
        ranges=(NOMINAL_HEIGHT, NOMINAL_HEIGHT),
        element_name="height",
    )
    base_roll = mdp.UniformScalarCommandCfg(
        resampling_time_range=(8.0, 12.0),
        ranges=(NOMINAL_ROLL, NOMINAL_ROLL),
        element_name="roll",
    )


@configclass
class FixedLegWheelActionsCfg:
    """Stage 1/2 actions: keep a fixed 6D interface while only wheel actions affect motion."""

    leg_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["servo2", "servo1", "servo4", "servo3"],
        scale=0.0,
        use_default_offset=True,
        preserve_order=True,
    )

    wheel_vel = mdp.JointVelocityActionCfg(
        asset_name="robot",
        joint_names=["wheel1", "wheel2"],
        scale=8.0,
        use_default_offset=True,
        preserve_order=True,
    )


@configclass
class LegWheelActionsCfg:
    """Stage 3/4 actions: four leg position targets and two wheel velocity targets."""

    leg_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["servo2", "servo1", "servo4", "servo3"],
        scale={
            "servo2": 2.47,
            "servo1": 1.90,
            "servo4": 2.47,
            "servo3": 1.90,
        },
        clip={
            "servo2": (-1.57, 1.57),
            "servo1": (-3.14, 0.0),
            "servo4": (-1.57, 1.57),
            "servo3": (-3.14, 0.0),
        },
        use_default_offset=True,
        preserve_order=True,
    )
    wheel_vel = mdp.JointVelocityActionCfg(
        asset_name="robot",
        joint_names=["wheel1", "wheel2"],
        scale=8.0,
        use_default_offset=True,
        preserve_order=True,
    )


@configclass
class ObservationsCfg:
    """Policy observations."""

    @configclass
    class PolicyCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.05, n_max=0.05))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.03, n_max=0.03))
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        height_target = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_height"})
        roll_target = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_roll"})
        base_height = ObsTerm(func=mdp.base_height)
        base_roll = ObsTerm(func=mdp.base_roll)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.5, n_max=0.5))
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventsCfg:
    """Randomization and reset terms."""

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["left_wheel", "right_wheel"]),
            "static_friction_range": (0.8, 1.2),
            "dynamic_friction_range": (0.6, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 32,
        },
    )
    randomize_body_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "mass_distribution_params": (0.9, 1.1),
            "operation": "scale",
            "distribution": "uniform",
            "recompute_inertia": True,
        },
    )
    randomize_base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["base_link"]),
            "com_range": {"x": (-0.01, 0.01), "y": (-0.01, 0.01), "z": (-0.01, 0.01)},
        },
    )
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.2, 0.2), "y": (-0.2, 0.2), "yaw": (-0.2, 0.2)},
            "velocity_range": {
                "x": (-0.05, 0.05),
                "y": (-0.05, 0.05),
                "z": (0.0, 0.0),
                "roll": (-0.05, 0.05),
                "pitch": (-0.05, 0.05),
                "yaw": (-0.05, 0.05),
            },
        },
    )
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
        },
    )
    hold_leg_joint_targets = EventTerm(
        func=mdp.set_joint_targets_to_default,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=["servo2", "servo1", "servo4", "servo3"],
                preserve_order=True,
            ),
        },
    )
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(8.0, 12.0),
        params={"velocity_range": {"x": (-0.2, 0.2), "y": (-0.1, 0.1), "yaw": (-0.2, 0.2)}},
    )
    stance_wrench_pulse: EventTerm | None = None


@configclass
class RewardsCfg:
    """Reward terms for staged training."""

    # Task rewards.
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=0.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    track_height_exp = RewTerm(
        func=mdp.track_base_height_exp,
        weight=0.5,
        params={"command_name": "base_height", "std": 0.04},
    )
    track_roll_exp = RewTerm(
        func=mdp.track_base_roll_exp,
        weight=0.25,
        params={"command_name": "base_roll", "std": 0.12},
    )

    # Stability and smoothness penalties.
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.5)
    lateral_lin_vel_l2 = RewTerm(func=mdp.lateral_lin_vel_l2, weight=-0.2)
    wheel_forward_alignment_l2 = RewTerm(
        func=mdp.wheel_forward_alignment_l2,
        weight=0.0,
        params={
            "forward_axis": 0,
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=["left_wheel", "right_wheel"],
                preserve_order=True,
            ),
        },
    )
    leg_joint_symmetry_l2 = RewTerm(
        func=mdp.leg_joint_symmetry_l2,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=["servo2", "servo1", "servo4", "servo3"],
                preserve_order=True,
            ),
        },
    )
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    pitch_l2 = RewTerm(func=mdp.pitch_l2, weight=-0.5)
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-2.5e-5)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-1.0e-7)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.02)
    joint_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-0.1)


@configclass
class TerminationsCfg:
    """Episode termination conditions."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    root_height = DoneTerm(func=mdp.root_height_out_of_bounds, params={"bounds": (0.20, 0.60)})
    root_orientation = DoneTerm(
        func=mdp.root_orientation_out_of_bounds,
        params={"roll_limit": 0.75, "pitch_limit": 0.75},
    )
    leg_or_foot_contact: DoneTerm | None = None
    wheel_scissor: DoneTerm | None = None


@configclass
class CurriculumCfg:
    """Placeholder curriculum container."""

    terrain_levels: CurrTerm | None = None


@configclass
class WheeledLeggedStage1EnvCfg(ManagerBasedRLEnvCfg):
    """Stage 1: fixed-height velocity tracking with symmetric leg control."""

    sim: SimulationCfg = SimulationCfg(physics=PhysxCfg(gpu_max_rigid_patch_count=10 * 2**15))
    scene: WheeledLeggedSceneCfg = WheeledLeggedSceneCfg(num_envs=2048, env_spacing=2.0, clone_in_fabric=True)
    observations: ObservationsCfg = ObservationsCfg()
    actions: LegWheelActionsCfg = LegWheelActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.viewer.eye = (2.5, -2.5, 1.5)
        self.viewer.lookat = (0.0, 0.0, 0.3)
        self.actions.leg_pos.scale = STAGE3_SMOOTH_LEG_ACTION_SCALE
        self.rewards.track_height_exp.weight = 1.0
        self.rewards.wheel_forward_alignment_l2.weight = -12.0
        self.rewards.leg_joint_symmetry_l2.weight = -4.0
        self.terminations.wheel_scissor = DoneTerm(
            func=mdp.wheel_forward_offset_too_large,
            params={
                "max_offset": 0.06,
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    body_names=["left_wheel", "right_wheel"],
                    preserve_order=True,
                ),
            },
        )


class WheeledLeggedStage1EnvCfg_PLAY(WheeledLeggedStage1EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.observations.policy.enable_corruption = False
        self.events.physics_material = None
        self.events.push_robot = None


@configclass
class WheeledLeggedStage2EnvCfg(WheeledLeggedStage1EnvCfg):
    """Stage 2: add yaw-rate tracking while retaining symmetric leg control."""

    actions: LegWheelActionsCfg = LegWheelActionsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.actions.leg_pos.scale = STAGE3_SMOOTH_LEG_ACTION_SCALE
        self.commands.base_velocity.ranges.lin_vel_x = (-0.6, 0.6)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.2, 1.2)
        self.rewards.track_ang_vel_z_exp.weight = 0.5
        self.rewards.wheel_forward_alignment_l2.weight = -8.0
        self.rewards.leg_joint_symmetry_l2.weight = -2.0
        self.events.push_robot.params["velocity_range"] = {"x": (-0.3, 0.3), "y": (-0.15, 0.15), "yaw": (-0.3, 0.3)}


@configclass
class WheeledLeggedStage3BaseEnvCfg(WheeledLeggedStage2EnvCfg):
    """Base Stage 3 settings: fixed-height static stance recovery under force pulses."""

    actions: LegWheelActionsCfg = LegWheelActionsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.actions.leg_pos.scale = STAGE3_SMOOTH_LEG_ACTION_SCALE
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.rel_standing_envs = 1.0
        self.commands.base_height.ranges = (NOMINAL_HEIGHT, NOMINAL_HEIGHT)
        self.commands.base_roll.ranges = (NOMINAL_ROLL, NOMINAL_ROLL)
        self.rewards.track_lin_vel_xy_exp.params["std"] = 0.25
        self.rewards.track_ang_vel_z_exp.params["std"] = 0.25
        self.rewards.track_ang_vel_z_exp.weight = 0.75
        self.rewards.track_height_exp.weight = 1.0
        self.rewards.track_roll_exp.weight = 0.5
        self.rewards.wheel_forward_alignment_l2.weight = -12.0
        self.rewards.leg_joint_symmetry_l2.weight = -4.0
        self.rewards.lin_vel_z_l2.weight = -0.35
        self.rewards.dof_acc_l2.weight = -5.0e-7
        self.rewards.action_rate_l2.weight = -0.075
        self.terminations.root_height.params["bounds"] = (0.12, 0.60)
        self.terminations.leg_or_foot_contact = DoneTerm(
            func=mdp.illegal_contact,
            params={
                "threshold": 1.0,
                "sensor_cfg": SceneEntityCfg(
                    "leg_contact",
                    body_names=["left_leg", "left_foot", "right_leg", "right_foot"],
                    preserve_order=True,
                ),
            },
        )
        self.events.push_robot = None
        self.events.stance_wrench_pulse = EventTerm(
            func=mdp.StaticStanceWrenchPulse,
            mode="interval",
            interval_range_s=(0.02, 0.02),
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=["base_link"], preserve_order=True),
                "force_range": (-6.0, 6.0),
                "yaw_torque_range": (-0.4, 0.4),
                "pulse_duration_s": (0.10, 0.20),
                "wait_time_s": (4.0, 7.0),
            },
        )


@configclass
class WheeledLeggedStage3aEnvCfg(WheeledLeggedStage3BaseEnvCfg):
    """Stage 3a: alias for the fixed-height static stance task."""

    pass


@configclass
class WheeledLeggedStage3bEnvCfg(WheeledLeggedStage3BaseEnvCfg):
    """Stage 3b: alias for the fixed-height static stance task."""

    pass


@configclass
class WheeledLeggedStage3cEnvCfg(WheeledLeggedStage3BaseEnvCfg):
    """Stage 3c: alias for the fixed-height static stance task."""

    pass


@configclass
class WheeledLeggedStage3dEnvCfg(WheeledLeggedStage3BaseEnvCfg):
    """Stage 3d: alias for the fixed-height static stance task."""

    pass


@configclass
class WheeledLeggedStage3EnvCfg(WheeledLeggedStage3dEnvCfg):
    """Stage 3 final alias: fixed-height static stance recovery."""

    pass


@configclass
class WheeledLeggedStage3EnvCfg_PLAY(WheeledLeggedStage3EnvCfg):
    """Deterministic Stage 3 playback configuration for cross-simulator logging."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.observations.policy.enable_corruption = False
        self.events.physics_material = None
        self.events.randomize_body_mass = None
        self.events.randomize_base_com = None
        # Keep the training reset path for playback initialization.  Make it
        # deterministic so this test isolates the root-height initialization.
        self.events.reset_base.params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}
        self.events.reset_base.params["velocity_range"] = {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        }
        self.events.reset_robot_joints = None
        self.events.hold_leg_joint_targets = None
        self.events.push_robot = None
        self.events.stance_wrench_pulse = None


@configclass
class WheeledLeggedStage4BaseEnvCfg(WheeledLeggedStage3EnvCfg):
    """Base Stage 4 settings: train roll tracking while holding target yaw rate at zero."""

    def __post_init__(self):
        super().__post_init__()
        self.terminations.leg_or_foot_contact = None
        self.commands.base_velocity.ranges.lin_vel_x = (-0.6, 0.6)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.rel_standing_envs = 0.05
        self.commands.base_height.ranges = STAGE3D_HEIGHT_COMMAND_RANGE
        self.rewards.track_lin_vel_xy_exp.params["std"] = 0.5
        self.rewards.track_ang_vel_z_exp.params["std"] = 0.5
        self.rewards.track_ang_vel_z_exp.weight = 0.2
        self.rewards.track_height_exp.weight = 0.8
        self.rewards.ang_vel_xy_l2.weight = -0.03
        self.rewards.action_rate_l2.weight = -0.05
        self.rewards.wheel_forward_alignment_l2.weight = -8.0
        self.rewards.leg_joint_symmetry_l2.weight = 0.0
        self.events.stance_wrench_pulse = None
        self.events.push_robot = EventTerm(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(8.0, 12.0),
            params={"velocity_range": {"x": (-0.3, 0.3), "y": (-0.15, 0.15), "yaw": (-0.3, 0.3)}},
        )

    def _set_roll_curriculum(self, roll_range: tuple[float, float], roll_weight: float):
        self.commands.base_roll.ranges = roll_range
        self.rewards.track_roll_exp.weight = roll_weight


@configclass
class WheeledLeggedStage4aEnvCfg(WheeledLeggedStage4BaseEnvCfg):
    """Stage 4a: small roll targets with zero target yaw rate."""

    def __post_init__(self):
        super().__post_init__()
        self._set_roll_curriculum(STAGE4A_ROLL_COMMAND_RANGE, 0.4)


@configclass
class WheeledLeggedStage4bEnvCfg(WheeledLeggedStage4BaseEnvCfg):
    """Stage 4b: medium roll targets with zero target yaw rate."""

    def __post_init__(self):
        super().__post_init__()
        self._set_roll_curriculum(STAGE4B_ROLL_COMMAND_RANGE, 0.6)


@configclass
class WheeledLeggedStage4cEnvCfg(WheeledLeggedStage4BaseEnvCfg):
    """Stage 4c: height-conditioned roll targets with zero target yaw rate."""

    def __post_init__(self):
        super().__post_init__()
        self.commands.base_roll = mdp.HeightConditionedRollCommandCfg(
            resampling_time_range=(8.0, 12.0),
            ranges=STAGE4C_ROLL_COMMAND_RANGE,
            element_name="roll",
            height_command_name="base_height",
            height_range=STAGE3D_HEIGHT_COMMAND_RANGE,
            center_height=NOMINAL_HEIGHT,
            center_roll_limit=STAGE4C_ROLL_COMMAND_RANGE[1],
            edge_roll_limit=STAGE4C_EDGE_ROLL_LIMIT,
        )
        self.rewards.track_roll_exp.weight = 0.8


@configclass
class WheeledLeggedStage4EnvCfg(WheeledLeggedStage4cEnvCfg):
    """Stage 4 final alias: full roll range with zero target yaw rate."""

    pass


@configclass
class WheeledLeggedStage5BaseEnvCfg(WheeledLeggedStage4cEnvCfg):
    """Base Stage 5 settings: reintroduce yaw-rate tracking under height-conditioned roll targets."""

    def _set_yaw_curriculum(self, yaw_rate_range: tuple[float, float], yaw_weight: float):
        self.commands.base_velocity.ranges.ang_vel_z = yaw_rate_range
        self.rewards.track_ang_vel_z_exp.weight = yaw_weight


@configclass
class WheeledLeggedStage5aEnvCfg(WheeledLeggedStage5BaseEnvCfg):
    """Stage 5a: small yaw-rate targets with full height and height-conditioned roll."""

    def __post_init__(self):
        super().__post_init__()
        self._set_yaw_curriculum(STAGE5A_YAW_RATE_RANGE, 0.3)


@configclass
class WheeledLeggedStage5bEnvCfg(WheeledLeggedStage5BaseEnvCfg):
    """Stage 5b: medium yaw-rate targets with full height and height-conditioned roll."""

    def __post_init__(self):
        super().__post_init__()
        self._set_yaw_curriculum(STAGE5B_YAW_RATE_RANGE, 0.4)


@configclass
class WheeledLeggedStage5cEnvCfg(WheeledLeggedStage5BaseEnvCfg):
    """Stage 5c: full yaw-rate targets with full height and height-conditioned roll."""

    def __post_init__(self):
        super().__post_init__()
        self._set_yaw_curriculum(STAGE5C_YAW_RATE_RANGE, 0.5)


@configclass
class WheeledLeggedStage5EnvCfg(WheeledLeggedStage5cEnvCfg):
    """Stage 5 final alias: full yaw-rate range with height-conditioned roll targets."""

    pass


@configclass
class WheeledLeggedStage6BaseEnvCfg(WheeledLeggedStage3dEnvCfg):
    """Base Stage 6 settings: rate-limited commands for sim-to-real control adaptation."""

    def __post_init__(self):
        super().__post_init__()
        self.terminations.leg_or_foot_contact = None
        self.events.stance_wrench_pulse = None
        self.events.push_robot = EventTerm(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(8.0, 12.0),
            params={"velocity_range": {"x": (-0.3, 0.3), "y": (-0.15, 0.15), "yaw": (-0.3, 0.3)}},
        )
        self.rewards.track_lin_vel_xy_exp.params["std"] = 0.5
        self.rewards.track_ang_vel_z_exp.params["std"] = 0.5
        self.rewards.action_rate_l2.weight = -0.05
        self.rewards.wheel_forward_alignment_l2.weight = -8.0
        self.rewards.leg_joint_symmetry_l2.weight = 0.0
        self.commands.base_height = mdp.SmoothScalarCommandCfg(
            resampling_time_range=STAGE6_COMMAND_RESAMPLING_TIME,
            ranges=STAGE3D_HEIGHT_COMMAND_RANGE,
            element_name="height",
            rate_limit=STAGE6_HEIGHT_RATE_LIMIT,
        )
        self.commands.base_roll = mdp.SmoothScalarCommandCfg(
            resampling_time_range=STAGE6_COMMAND_RESAMPLING_TIME,
            ranges=(0.0, 0.0),
            element_name="roll",
            rate_limit=STAGE6_ROLL_RATE_LIMIT,
        )
        self.rewards.track_height_exp.weight = 1.0
        self.rewards.track_roll_exp.weight = 0.25
        self.rewards.lin_vel_z_l2.weight = -0.45
        self.rewards.track_ang_vel_z_exp.weight = 0.2

    def _set_smooth_velocity_command(self, yaw_rate_range: tuple[float, float], yaw_weight: float):
        self.commands.base_velocity = mdp.SmoothVelocityCommandCfg(
            asset_name="robot",
            resampling_time_range=STAGE6_COMMAND_RESAMPLING_TIME,
            rel_standing_envs=0.05,
            rel_heading_envs=0.0,
            heading_command=False,
            debug_vis=True,
            ranges=mdp.SmoothVelocityCommandCfg.Ranges(
                lin_vel_x=(-0.6, 0.6),
                lin_vel_y=(0.0, 0.0),
                ang_vel_z=yaw_rate_range,
                heading=(0.0, 0.0),
            ),
            rate_limits=STAGE6_VELOCITY_RATE_LIMITS,
        )
        self.rewards.track_ang_vel_z_exp.weight = yaw_weight


@configclass
class WheeledLeggedStage6aEnvCfg(WheeledLeggedStage6BaseEnvCfg):
    """Stage 6a: smooth forward velocity and height commands with zero target yaw and roll."""

    def __post_init__(self):
        super().__post_init__()
        self._set_smooth_velocity_command((0.0, 0.0), 0.2)


@configclass
class WheeledLeggedStage6bEnvCfg(WheeledLeggedStage6BaseEnvCfg):
    """Stage 6b: smooth forward velocity, yaw-rate, and height commands with zero roll."""

    def __post_init__(self):
        super().__post_init__()
        self._set_smooth_velocity_command(STAGE5C_YAW_RATE_RANGE, 0.5)


@configclass
class WheeledLeggedStage6cEnvCfg(WheeledLeggedStage6bEnvCfg):
    """Stage 6c: optional small smooth roll display commands on top of Stage 6b."""

    def __post_init__(self):
        super().__post_init__()
        self.commands.base_roll.ranges = STAGE6_ROLL_DEMO_RANGE
        self.rewards.track_roll_exp.weight = 0.35


@configclass
class WheeledLeggedStage6EnvCfg(WheeledLeggedStage6bEnvCfg):
    """Stage 6 final alias: real-control adaptation without random roll display commands."""

    pass
