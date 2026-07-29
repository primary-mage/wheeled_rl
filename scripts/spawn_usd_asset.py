"""Spawn a USD asset in Isaac Lab for visual inspection."""

import argparse

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Spawn a USD asset in Isaac Lab.")
parser.add_argument("--usd_path", required=True, help="Absolute path to the USD/USDA asset.")
parser.add_argument("--prim_path", default="/World/Robot", help="Prim path where the asset is spawned.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils


def main():
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([2.0, -2.0, 1.4], [0.0, 0.0, 0.35])

    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/defaultGroundPlane", ground_cfg)

    light_cfg = sim_utils.DistantLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/lightDistant", light_cfg, translation=(1.0, -1.0, 3.0))

    asset_cfg = sim_utils.UsdFileCfg(usd_path=args_cli.usd_path)
    asset_cfg.func(args_cli.prim_path, asset_cfg, translation=(0.0, 0.0, 0.5))

    sim.reset()
    print(f"[INFO]: Spawned USD asset at {args_cli.prim_path}")
    print(f"[INFO]: USD path: {args_cli.usd_path}")

    while simulation_app.is_running():
        sim.step()


if __name__ == "__main__":
    main()
    simulation_app.close()
