"""Train wheeled-legged tasks with Isaac Lab's RSL-RL entry point."""

import runpy
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ISAACLAB_TRAIN = Path("/home/mage/IsaacLab/scripts/reinforcement_learning/rsl_rl/train.py")
ISAACLAB_RSL_RL_DIR = ISAACLAB_TRAIN.parent

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(ISAACLAB_RSL_RL_DIR))
import wheeled_legged_rl.tasks.velocity  # noqa: F401, E402

runpy.run_path(str(ISAACLAB_TRAIN), run_name="__main__")
