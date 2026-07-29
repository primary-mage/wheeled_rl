"""Minimal Isaac Lab control test for the wheeled-legged robot USD."""

import argparse

from isaaclab.app import AppLauncher


DEFAULT_USD = "/home/mage/projects/wheeled_legged_rl/asset/usd/wheeled_robot/wheeled_robot.usda"
NOMINAL_HEIGHT = 0.270
NOMINAL_JOINT_POS = {
    "servo2": 0.9,
    "servo1": -1.9,
    "servo4": 0.9,
    "servo3": -1.9,
    "wheel1": 0.0,
    "wheel2": 0.0,
}

parser = argparse.ArgumentParser(description="Minimal control test for wheeled_robot.usda.")
parser.add_argument("--usd_path", default=DEFAULT_USD, help="Path to the converted robot USD/USDA.")
parser.add_argument("--wheel_speed", type=float, default=3.0, help="Wheel velocity target in rad/s.")
parser.add_argument("--leg_amp", type=float, default=0.25, help="Leg position oscillation amplitude in rad.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import math

import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg


ROBOT_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=args_cli.usd_path,
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
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, NOMINAL_HEIGHT),
        joint_pos=NOMINAL_JOINT_POS,
    ),
    actuators={
        "leg_position": ImplicitActuatorCfg(
            joint_names_expr=["servo.*"],
            effort_limit_sim=5.0,
            velocity_limit_sim=10.0,
            stiffness=10.0,
            damping=2.0,
        ),
        "wheel_velocity": ImplicitActuatorCfg(
            joint_names_expr=["wheel.*"],
            effort_limit_sim=5.0,
            velocity_limit_sim=20.0,
            stiffness=0.0,
            damping=2.0,
        ),
    },
)


class MinimalControlSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )
    robot = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    robot = scene["robot"]
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0

    print("[INFO]: Joint names:")
    for index, name in enumerate(robot.joint_names):
        print(f"  {index}: {name}")

    joint_name_to_id = {name: index for index, name in enumerate(robot.joint_names)}
    leg_ids = [joint_name_to_id[name] for name in ["servo2", "servo1", "servo4", "servo3"]]
    wheel_ids = [joint_name_to_id[name] for name in ["wheel1", "wheel2"]]

    while simulation_app.is_running():
        if count % 600 == 0:
            count = 0
            root_pose = robot.data.default_root_pose.torch.clone()
            root_pose[:, :3] += scene.env_origins
            root_vel = robot.data.default_root_vel.torch.clone()
            joint_pos = robot.data.default_joint_pos.torch.clone()
            joint_vel = robot.data.default_joint_vel.torch.clone()

            robot.write_root_pose_to_sim_index(root_pose=root_pose)
            robot.write_root_velocity_to_sim_index(root_velocity=root_vel)
            robot.write_joint_position_to_sim_index(position=joint_pos)
            robot.write_joint_velocity_to_sim_index(velocity=joint_vel)
            scene.reset()
            print("[INFO]: Reset robot state.")

        leg_target = robot.data.default_joint_pos.torch.clone()
        wave = args_cli.leg_amp * math.sin(2.0 * math.pi * 0.5 * sim_time)
        leg_target[:, leg_ids[0]] += wave
        leg_target[:, leg_ids[1]] += 0.5 * wave
        leg_target[:, leg_ids[2]] += wave
        leg_target[:, leg_ids[3]] += 0.5 * wave

        wheel_target = torch.zeros((scene.num_envs, len(wheel_ids)), device=sim.device)
        wheel_target[:, :] = args_cli.wheel_speed

        robot.set_joint_position_target_index(target=leg_target)
        robot.set_joint_velocity_target_index(target=wheel_target, joint_ids=wheel_ids)

        scene.write_data_to_sim()
        sim.step()
        sim_time += sim_dt
        count += 1
        scene.update(sim_dt)


def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([2.0, -2.0, 1.4], [0.0, 0.0, 0.25])

    scene_cfg = MinimalControlSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)

    sim.reset()
    print("[INFO]: Setup complete.")
    run_simulator(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()
