# Wheeled-Legged Velocity Tasks

This package registers staged Isaac Lab ManagerBasedRLEnv tasks for the wheeled-legged robot.

The robot base frame follows the Isaac/ROS-style convention: `+X` forward, `+Y` left, `+Z` up.

## Tasks

- `Isaac-WheeledLegged-Stage1-v0`: low-speed forward/backward velocity tracking, fixed height and roll.
- `Isaac-WheeledLegged-Stage2-v0`: adds yaw-rate tracking.
- `Isaac-WheeledLegged-Stage3-v0`: enables leg control and samples height targets.
- `Isaac-WheeledLegged-Stage4-v0`: adds sampled roll targets.

## Nominal Pose

```text
servo2 =  0.9
servo1 = -1.9
servo4 =  0.9
servo3 = -1.9
wheel1 =  0.0
wheel2 =  0.0
root z =  0.270
```

## Actions

All stages use the same 6D policy action interface:

```text
leg_pos:   servo2, servo1, servo4, servo3 -> joint position targets
wheel_vel: wheel1, wheel2                 -> joint velocity targets
```

Stage 1 and Stage 2 keep `leg_pos.scale = 0.0`, so the policy still outputs leg actions but the
servo targets stay at the nominal pose. Stage 3 and Stage 4 use default-offset joint position
targets with joint-limit clipping, so raw zero servo actions still map to the nominal pose while
nonzero actions can reach the height and roll command ranges.

```text
Stage 1/2: servo actions masked by zero scale; wheel actions active
Stage 3/4: servo actions active; wheel actions active
```

## Height And Roll Commands

Stage 1 and Stage 2 keep height and roll fixed:

```text
height target = 0.270 m
roll target   = 0.0 rad
```

Stage 3 samples height targets:

```text
height target range = [0.18, 0.36] m
```

Stage 4 inherits the height target range and samples roll targets:

```text
roll target range = [-0.5236, 0.5236] rad  # +/-30 deg
```

## Run

From the Isaac Lab root:

```bash
PYTHONPATH=/home/mage/projects/wheeled_legged_rl \
./isaaclab.sh train --rl_library rsl_rl \
  --task Isaac-WheeledLegged-Stage1-v0 \
  --external_callback wheeled_legged_rl.tasks.register.register_tasks \
  --viz kit
```

On this machine, `--viz none` currently crashes during Isaac Sim startup even for built-in tasks. Use `--viz kit`
until the headless runtime issue is fixed.

For visual playback:

```bash
PYTHONPATH=/home/mage/projects/wheeled_legged_rl \
./isaaclab.sh play --rl_library rsl_rl \
  --task Isaac-WheeledLegged-Stage1-Play-v0 \
  --external_callback wheeled_legged_rl.tasks.register.register_tasks \
  --viz kit
```
