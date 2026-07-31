# Wheeled-Legged Velocity Tasks

This package registers staged Isaac Lab ManagerBasedRLEnv tasks for the wheeled-legged robot.

The robot base frame follows the Isaac/ROS-style convention: `+X` forward, `+Y` left, `+Z` up.

## Tasks

- `Isaac-WheeledLegged-Stage1-v0`: low-speed forward/backward velocity tracking, fixed height and roll.
- `Isaac-WheeledLegged-Stage2-v0`: adds yaw-rate tracking.
- `Isaac-WheeledLegged-Stage3a-v0` through `Stage3d-v0`: fixed-height static stance recovery under intermittent wrench pulses.
- `Isaac-WheeledLegged-Stage3-v0`: alias for the static stance task.
- `Isaac-WheeledLegged-Stage4a-v0`: adds small roll targets while holding yaw-rate targets at zero.
- `Isaac-WheeledLegged-Stage4b-v0`: expands roll targets while holding yaw-rate targets at zero.
- `Isaac-WheeledLegged-Stage4c-v0`: trains height-conditioned roll targets while holding yaw-rate targets at zero.
- `Isaac-WheeledLegged-Stage4-v0`: alias for Stage 4c.
- `Isaac-WheeledLegged-Stage5a-v0`: reintroduces small yaw-rate targets under height-conditioned roll.
- `Isaac-WheeledLegged-Stage5b-v0`: expands yaw-rate targets under height-conditioned roll.
- `Isaac-WheeledLegged-Stage5c-v0`: trains the full yaw-rate range under height-conditioned roll.
- `Isaac-WheeledLegged-Stage5-v0`: alias for Stage 5c.
- `Isaac-WheeledLegged-Stage6a-v0`: trains smooth forward velocity and height commands with zero yaw and roll.
- `Isaac-WheeledLegged-Stage6b-v0`: trains smooth forward velocity, yaw-rate, and height commands with zero roll.
- `Isaac-WheeledLegged-Stage6c-v0`: optional small smooth roll display commands on top of Stage 6b.
- `Isaac-WheeledLegged-Stage6-v0`: alias for Stage 6b, the real-control adaptation task.

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
nonzero actions can stabilize the body or reach the downstream height and roll command ranges.

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

Stage 3 holds a fixed static stance:

```text
height target = 0.270 m, roll target = 0.0 rad
linear and yaw-rate targets = 0.0
horizontal-force or yaw-torque pulse every 4-7 s
pulse duration = 0.10-0.20 s
force range = [-6.0, 6.0] N, yaw-torque range = [-0.4, 0.4] N m
wheel fore-aft alignment penalty = -12.0 * (left_wheel_x - right_wheel_x)^2
leg symmetry penalty = -4.0 * ((servo2 - servo4)^2 + (servo1 - servo3)^2)
Stage 3/4 smooth leg action scale = {servo2: 1.60, servo1: 1.25, servo4: 1.60, servo3: 1.25}
```

Stage 4 inherits the full Stage 3d height target range, holds the target yaw rate at zero, and
uses a roll curriculum:

```text
Stage 4a: roll target range = [-0.1745, 0.1745] rad  # +/-10 deg, roll weight = 0.4
Stage 4b: roll target range = [-0.3491, 0.3491] rad  # +/-20 deg, roll weight = 0.6
Stage 4c: roll target range is conditioned on height, roll weight = 0.8
  height near 0.27 m: roll limit = +/-30 deg
  height near 0.18/0.36 m: roll limit = +/-8 deg
yaw-rate target range = [0.0, 0.0] rad/s, yaw-rate tracking weight = 0.2
```

Stage 5 inherits Stage 4c and gradually reintroduces yaw-rate targets:

```text
Stage 5a: yaw-rate target range = [-0.4, 0.4] rad/s, yaw weight = 0.3
Stage 5b: yaw-rate target range = [-0.8, 0.8] rad/s, yaw weight = 0.4
Stage 5c: yaw-rate target range = [-1.2, 1.2] rad/s, yaw weight = 0.5
```

Stage 6 uses rate-limited commands to match the real control interface. Roll is fixed to zero for
the main real-control task and only reintroduced as a small optional display command in Stage 6c.

```text
Command resampling time = [2.0, 4.0] s
Velocity command rate limits = [0.7, 0.7, 1.2] for vx, vy, yaw_rate
Height command rate limit = 0.06 m/s

Stage 6a: vx [-0.6, 0.6] m/s, yaw_rate = 0, height [0.18, 0.36] m, roll = 0
Stage 6b: vx [-0.6, 0.6] m/s, yaw_rate [-1.2, 1.2] rad/s, height [0.18, 0.36] m, roll = 0
Stage 6c: Stage 6b plus small smooth roll display commands in [-10, 10] deg
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
