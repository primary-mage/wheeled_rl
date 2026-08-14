"""P1-P6 training configurations for the wheeled-legged robot.

The curriculum is deliberately global: every phase has one fixed command
distribution and is promoted only by the external evaluation gate.  This keeps
the optimization target explicit and avoids mixing easy and hard command
distributions inside a single PPO run.
"""

import math
from pathlib import Path

from isaaclab_physx.physics import PhysxCfg

import isaaclab.sim as sim_utils
from isaaclab.actuators import DelayedPDActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
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
POLICY_DT = 0.020
ACTUATOR_TORQUE_LIMIT_NM = 2.0
LEG_ACTION_SCALE = {
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
        # Required by the leg/foot contact termination sensors below.
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
            effort_limit_sim=ACTUATOR_TORQUE_LIMIT_NM,
            velocity_limit_sim=10.0,
            stiffness=10.0,
            damping=2.0,
            armature=0.01,
            min_delay=0,
            max_delay=4,
        ),
        "wheel_velocity": DelayedPDActuatorCfg(
            joint_names_expr=["wheel.*"],
            effort_limit=ACTUATOR_TORQUE_LIMIT_NM,
            effort_limit_sim=ACTUATOR_TORQUE_LIMIT_NM,
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
    """Flat scene plus contact reporters used by every P1-P6 phase."""

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
    left_leg_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Geometry/base_link/left_leg",
        update_period=0.0,
        history_length=1,
    )
    left_foot_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Geometry/base_link/left_leg/left_foot",
        update_period=0.0,
        history_length=1,
    )
    right_leg_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Geometry/base_link/right_leg",
        update_period=0.0,
        history_length=1,
    )
    right_foot_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Geometry/base_link/right_leg/right_foot",
        update_period=0.0,
        history_length=1,
    )
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=1000.0),
    )


def maneuver_command(
    *,
    speed_bins: tuple[tuple[float, float, float], ...],
    yaw_range: tuple[float, float],
    mode_weights: tuple[float, float, float, float],
    resampling_time_range: tuple[float, float],
    rate_limits: tuple[float, float, float],
    max_lateral_accel: float = 0.35,
) -> mdp.ManeuverVelocityCommandCfg:
    """Build the command source shared by training and deployment semantics."""

    return mdp.ManeuverVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=resampling_time_range,
        rel_standing_envs=0.0,
        rel_heading_envs=0.0,
        heading_command=False,
        debug_vis=True,
        ranges=mdp.ManeuverVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=yaw_range,
            heading=(0.0, 0.0),
        ),
        rate_limits=rate_limits,
        mode_weights=mode_weights,
        speed_bins=speed_bins,
        max_lateral_accel=max_lateral_accel,
    )


@configclass
class CommandsCfg:
    """Command terms; target height and roll stay fixed in all training phases."""

    base_velocity = maneuver_command(
        speed_bins=((1.0, 0.0, 0.0),),
        yaw_range=(0.0, 0.0),
        mode_weights=(1.0, 0.0, 0.0, 0.0),
        resampling_time_range=(2.0, 4.0),
        rate_limits=(0.40, 0.0, 0.80),
    )
    base_height = mdp.UniformScalarCommandCfg(
        resampling_time_range=(20.0, 20.0),
        ranges=(NOMINAL_HEIGHT, NOMINAL_HEIGHT),
        element_name="height",
    )
    base_roll = mdp.UniformScalarCommandCfg(
        resampling_time_range=(20.0, 20.0),
        ranges=(NOMINAL_ROLL, NOMINAL_ROLL),
        element_name="roll",
    )


@configclass
class LegWheelActionsCfg:
    """Four leg position targets followed by two wheel velocity targets."""

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
    """The unchanged 34-D policy observation used by MuJoCo and hardware."""

    @configclass
    class PolicyCfg(ObsGroup):
        # [0:3], [3:6], [6:9]
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.05, n_max=0.05))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.03, n_max=0.03))
        # [9:12], [12], [13], [14], [15]
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        height_target = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_height"})
        roll_target = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_roll"})
        base_height = ObsTerm(func=mdp.base_height)
        base_roll = ObsTerm(func=mdp.base_roll)
        # [16:22], [22:28], [28:34]
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.5, n_max=0.5))
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventsCfg:
    """Reset terms and the full P6 domain-randomization set."""

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["left_wheel", "right_wheel"]),
            "static_friction_range": (0.70, 1.25),
            "dynamic_friction_range": (0.55, 1.05),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 32,
        },
    )
    randomize_body_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "mass_distribution_params": (0.90, 1.10),
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
            "pose_range": {"x": (-0.2, 0.2), "y": (-0.2, 0.2), "yaw": (-0.15, 0.15)},
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
        params={"position_range": (-0.03, 0.03), "velocity_range": (-0.10, 0.10)},
    )
    hold_leg_joint_targets = EventTerm(
        func=mdp.set_joint_targets_to_default,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=["servo2", "servo1", "servo4", "servo3"],
                preserve_order=True,
            )
        },
    )
    push_robot: EventTerm | None = None


@configclass
class RewardsCfg:
    """Common reward terms; each phase adjusts only its relevant weights."""

    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=1.50,
        params={"command_name": "base_velocity", "std": 0.25},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=0.0,
        params={"command_name": "base_velocity", "std": 0.25},
    )
    track_height_exp = RewTerm(
        func=mdp.track_base_height_exp,
        weight=0.80,
        params={"command_name": "base_height", "std": 0.035},
    )
    track_roll_exp = RewTerm(
        func=mdp.track_base_roll_exp,
        weight=0.80,
        params={"command_name": "base_roll", "std": 0.10},
    )
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.50)
    lateral_lin_vel_l2 = RewTerm(func=mdp.lateral_lin_vel_l2, weight=-0.25)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.08)
    pitch_l2 = RewTerm(func=mdp.pitch_l2, weight=-0.80)
    stationary_wheel_vel_l2 = RewTerm(
        func=mdp.wheel_velocity_when_stationary_l2,
        weight=-0.002,
        params={
            "command_name": "base_velocity",
            "stationary_threshold": 0.05,
            "asset_cfg": SceneEntityCfg("robot", joint_names=["wheel1", "wheel2"], preserve_order=True),
        },
    )
    wheel_forward_alignment_l2 = RewTerm(
        func=mdp.wheel_forward_alignment_l2,
        weight=-10.0,
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
        weight=-3.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=["servo2", "servo1", "servo4", "servo3"],
                preserve_order=True,
            ),
        },
    )
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-2.5e-5)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-5.0e-7)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.06)
    joint_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-0.10)


@configclass
class TerminationsCfg:
    """Safety terminations shared by all phases, including P6."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    root_height = DoneTerm(func=mdp.root_height_out_of_bounds, params={"bounds": (0.15, 0.45)})
    root_orientation = DoneTerm(
        func=mdp.root_orientation_out_of_bounds,
        params={"roll_limit": math.radians(35.0), "pitch_limit": math.radians(35.0)},
    )
    leg_or_foot_contact = DoneTerm(
        func=mdp.leg_or_foot_contact,
        params={
            "force_threshold": 1.0,
            "sensor_names": (
                "left_leg_contact",
                "left_foot_contact",
                "right_leg_contact",
                "right_foot_contact",
            ),
        },
    )
    wheel_scissor = DoneTerm(
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


@configclass
class WheeledLeggedP1EnvCfg(ManagerBasedRLEnvCfg):
    """P1: fixed-pose stationary lock and recovery from small velocity pushes."""

    sim: SimulationCfg = SimulationCfg(physics=PhysxCfg(gpu_max_rigid_patch_count=10 * 2**15))
    scene: WheeledLeggedSceneCfg = WheeledLeggedSceneCfg(num_envs=2048, env_spacing=2.0, clone_in_fabric=True)
    observations: ObservationsCfg = ObservationsCfg()
    actions: LegWheelActionsCfg = LegWheelActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.viewer.eye = (2.5, -2.5, 1.5)
        self.viewer.lookat = (0.0, 0.0, 0.3)
        self.actions.leg_pos.scale = LEG_ACTION_SCALE

        # The first five phases train nominal dynamics only.  P6 explicitly
        # re-enables the complete randomization set.
        self.events.physics_material = None
        self.events.randomize_body_mass = None
        self.events.randomize_base_com = None
        self.events.push_robot = EventTerm(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(3.0, 6.0),
            params={"velocity_range": {"x": (-0.08, 0.08), "y": (-0.05, 0.05), "yaw": (-0.10, 0.10)}},
        )


@configclass
class WheeledLeggedP2EnvCfg(WheeledLeggedP1EnvCfg):
    """P2: low-speed longitudinal tracking while retaining frequent zero holds."""

    def __post_init__(self):
        super().__post_init__()
        self.commands.base_velocity = maneuver_command(
            speed_bins=((1.0, 0.05, 0.25),),
            yaw_range=(0.0, 0.0),
            mode_weights=(0.45, 0.35, 0.20, 0.0),
            resampling_time_range=(1.5, 3.0),
            rate_limits=(0.30, 0.0, 0.80),
        )
        self.events.push_robot = None
        self.rewards.track_lin_vel_xy_exp.params["std"] = 0.18
        self.rewards.action_rate_l2.weight = -0.05


@configclass
class WheeledLeggedP3EnvCfg(WheeledLeggedP2EnvCfg):
    """P3: abrupt braking and reversal-through-zero at moderate speed."""

    def __post_init__(self):
        super().__post_init__()
        self.commands.base_velocity = maneuver_command(
            speed_bins=((0.55, 0.05, 0.25), (0.45, 0.25, 0.50)),
            yaw_range=(0.0, 0.0),
            mode_weights=(0.32, 0.25, 0.27, 0.16),
            resampling_time_range=(0.75, 1.75),
            rate_limits=(1.00, 0.0, 0.80),
        )
        self.rewards.track_lin_vel_xy_exp.params["std"] = 0.25
        self.rewards.action_rate_l2.weight = -0.035


@configclass
class WheeledLeggedP4AEnvCfg(WheeledLeggedP3EnvCfg):
    """P4a: extend the validated maneuver set to 0.6 m/s."""

    def __post_init__(self):
        super().__post_init__()
        self.commands.base_velocity = maneuver_command(
            speed_bins=((0.45, 0.05, 0.30), (0.55, 0.30, 0.60)),
            yaw_range=(0.0, 0.0),
            mode_weights=(0.28, 0.26, 0.27, 0.19),
            resampling_time_range=(0.75, 1.75),
            rate_limits=(1.10, 0.0, 0.80),
        )
        self.rewards.track_lin_vel_xy_exp.params["std"] = 0.32


@configclass
class WheeledLeggedP4BEnvCfg(WheeledLeggedP4AEnvCfg):
    """P4b: extend the validated maneuver set to 0.8 m/s."""

    def __post_init__(self):
        super().__post_init__()
        self.commands.base_velocity = maneuver_command(
            speed_bins=((0.30, 0.05, 0.30), (0.35, 0.30, 0.60), (0.35, 0.60, 0.80)),
            yaw_range=(0.0, 0.0),
            mode_weights=(0.28, 0.26, 0.27, 0.19),
            resampling_time_range=(0.75, 1.75),
            rate_limits=(1.20, 0.0, 0.80),
        )
        self.rewards.track_lin_vel_xy_exp.params["std"] = 0.40


@configclass
class WheeledLeggedP4CEnvCfg(WheeledLeggedP4BEnvCfg):
    """P4c: final straight-line high-speed maneuver training to 1.0 m/s."""

    def __post_init__(self):
        super().__post_init__()
        self.commands.base_velocity = maneuver_command(
            speed_bins=((0.25, 0.05, 0.30), (0.30, 0.30, 0.60), (0.20, 0.60, 0.80), (0.25, 0.80, 1.00)),
            yaw_range=(0.0, 0.0),
            mode_weights=(0.28, 0.26, 0.27, 0.19),
            resampling_time_range=(0.75, 1.75),
            rate_limits=(1.25, 0.0, 0.80),
        )
        self.rewards.track_lin_vel_xy_exp.params["std"] = 0.50


@configclass
class WheeledLeggedP5EnvCfg(WheeledLeggedP4CEnvCfg):
    """P5: coupled longitudinal/yaw tracking, constrained by lateral acceleration."""

    def __post_init__(self):
        super().__post_init__()
        self.commands.base_velocity = maneuver_command(
            speed_bins=((0.30, 0.05, 0.30), (0.35, 0.30, 0.60), (0.20, 0.60, 0.80), (0.15, 0.80, 1.00)),
            yaw_range=(-0.50, 0.50),
            mode_weights=(0.28, 0.26, 0.27, 0.19),
            resampling_time_range=(0.75, 1.75),
            rate_limits=(1.25, 0.0, 1.00),
            max_lateral_accel=0.35,
        )
        self.rewards.track_ang_vel_z_exp.weight = 0.70
        self.rewards.track_ang_vel_z_exp.params["std"] = 0.35
        self.events.push_robot = EventTerm(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(5.0, 8.0),
            params={"velocity_range": {"x": (-0.10, 0.10), "y": (-0.06, 0.06), "yaw": (-0.12, 0.12)}},
        )


@configclass
class WheeledLeggedP6EnvCfg(WheeledLeggedP5EnvCfg):
    """P6: full operating envelope with randomization for sim-to-real transfer."""

    def __post_init__(self):
        super().__post_init__()
        # Restore the domain randomization defined in EventsCfg.
        self.events.physics_material = EventTerm(
            func=mdp.randomize_rigid_body_material,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=["left_wheel", "right_wheel"]),
                "static_friction_range": (0.70, 1.25),
                "dynamic_friction_range": (0.55, 1.05),
                "restitution_range": (0.0, 0.0),
                "num_buckets": 32,
            },
        )
        self.events.randomize_body_mass = EventTerm(
            func=mdp.randomize_rigid_body_mass,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
                "mass_distribution_params": (0.90, 1.10),
                "operation": "scale",
                "distribution": "uniform",
                "recompute_inertia": True,
            },
        )
        self.events.randomize_base_com = EventTerm(
            func=mdp.randomize_rigid_body_com,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=["base_link"]),
                "com_range": {"x": (-0.01, 0.01), "y": (-0.01, 0.01), "z": (-0.01, 0.01)},
            },
        )
        self.commands.base_velocity = maneuver_command(
            speed_bins=((0.30, 0.05, 0.30), (0.35, 0.30, 0.60), (0.20, 0.60, 0.80), (0.15, 0.80, 1.00)),
            yaw_range=(-0.50, 0.50),
            mode_weights=(0.30, 0.25, 0.25, 0.20),
            resampling_time_range=(0.75, 1.75),
            rate_limits=(1.25, 0.0, 1.00),
            max_lateral_accel=0.35,
        )
        self.events.push_robot = EventTerm(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(4.0, 7.0),
            params={"velocity_range": {"x": (-0.18, 0.18), "y": (-0.10, 0.10), "yaw": (-0.20, 0.20)}},
        )
        self.rewards.track_ang_vel_z_exp.weight = 0.70
        self.rewards.track_ang_vel_z_exp.params["std"] = 0.35
        self.rewards.action_rate_l2.weight = -0.05
        self.rewards.dof_acc_l2.weight = -7.5e-7


@configclass
class WheeledLeggedP6EnvCfg_PLAY(WheeledLeggedP6EnvCfg):
    """Deterministic P6 playback for Isaac/MuJoCo/hardware comparison logs."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.observations.policy.enable_corruption = False
        self.events.physics_material = None
        self.events.randomize_body_mass = None
        self.events.randomize_base_com = None
        self.events.push_robot = None
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
