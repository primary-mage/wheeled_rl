# Wheeled-Legged Velocity Tasks

This package registers staged Isaac Lab ManagerBasedRLEnv tasks for the wheeled-legged robot.

## Tasks

- `Isaac-WheeledLegged-Stage1-v0`: low-speed forward/backward velocity tracking, fixed height and roll.
- `Isaac-WheeledLegged-Stage2-v0`: adds yaw-rate tracking.
- `Isaac-WheeledLegged-Stage3-v0`: scaffolds height tracking with a lower fixed target.
- `Isaac-WheeledLegged-Stage4-v0`: scaffolds nonzero roll tracking.

## Nominal Pose

```text
servo2 =  0.9
servo1 = -1.9
servo4 =  0.9
servo3 = -1.9
wheel1 =  0.0
wheel2 =  0.0
root z =  0.38
```

## Actions

Stage 1 and Stage 2 use wheel-only actions while the leg servos stay at the nominal pose:

```text
wheel_vel: wheel1, wheel2 -> joint velocity targets
```

Stage 3 and Stage 4 open the leg servos:

```text
leg_pos:   servo2, servo1, servo4, servo3 -> joint position targets
wheel_vel: wheel1, wheel2                 -> joint velocity targets
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
