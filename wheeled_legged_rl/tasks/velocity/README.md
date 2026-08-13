# Wheeled-Legged Velocity Tasks

This package registers staged Isaac Lab ManagerBasedRLEnv tasks for the wheeled-legged robot.
The training chain is `Stage 1 -> Stage 2 -> Stage 3 -> Stage 5 -> Stage 6`; each later group
loads the checkpoint produced by the preceding group.

To resume Stage 5 and Stage 6 from a Stage 3 checkpoint, use
`scripts/train_stage56_auto.sh`. It trains in chunks and moves to the next group only after at least
90% of parallel environments remain at the highest yaw curriculum level for two consecutive checks.

```bash
ISAACLAB_ROOT=/path/to/IsaacLab \
./scripts/train_stage56_auto.sh \
  logs/stage3_stance/model_399_stage3_contact_2nm.pt
```

The script is restartable. It stores the latest run/checkpoint in
`logs/stage56_from_stage3_contact_2nm_auto/resume.env` and curriculum levels in the same directory.
Set `RESET_CURRICULUM_STATE=1` to start a fresh automatic session. `CHUNK_ITERATIONS`,
`MAX_CHUNKS`, `MIN_MAX_LEVEL_FRACTION`, and `REQUIRED_READY_CHECKS` override the defaults.

The robot base frame follows the Isaac/ROS-style convention: `+X` forward, `+Y` left, `+Z` up.

## Tasks

- `Isaac-WheeledLegged-Stage1-v0`: low-speed forward/backward velocity tracking, fixed height and roll.
- `Isaac-WheeledLegged-Stage2-v0`: adds yaw-rate tracking.
- `Isaac-WheeledLegged-Stage3-v0`: per-environment adaptive height-range static stance recovery under intermittent wrench pulses.
- `Isaac-WheeledLegged-Stage3-Play-v0`: deterministic Stage 3 playback with randomization and external disturbances disabled.
- `Isaac-WheeledLegged-Stage5-v0`: per-environment adaptive yaw-rate tracking with fixed zero roll.
- `Isaac-WheeledLegged-Stage6-v0`: per-environment adaptive yaw-rate tracking with smooth velocity and height commands for real-control adaptation.

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

All stages train the leg and wheel actions jointly. Leg targets use a default offset and joint-limit
clipping, so raw zero servo actions map to the nominal pose while nonzero actions can stabilize the
body or reach the downstream height and roll command ranges.

```text
Stage 1-6: servo actions active; wheel actions active
```

## Height And Roll Commands

Stage 1 and Stage 2 keep height and roll fixed:

```text
height target = 0.270 m
roll target   = 0.0 rad
```

Stage 3 assigns each parallel environment an independent height level. An environment advances after
four successful near-full episodes and drops one level after two failed episodes. The active target
is already included in the policy observation.

```text
level 0: height target = [0.24, 0.30] m
level 1: height target = [0.22, 0.32] m
level 2: height target = [0.20, 0.34] m
level 3: height target = [0.18, 0.36] m
roll target = 0.0 rad
linear and yaw-rate targets = 0.0
horizontal-force or yaw-torque pulse every 4-7 s
pulse duration = 0.10-0.20 s
force range = [-6.0, 6.0] N, yaw-torque range = [-0.4, 0.4] N m
wheel fore-aft alignment penalty = -12.0 * (left_wheel_x - right_wheel_x)^2
leg symmetry penalty = -4.0 * ((servo2 - servo4)^2 + (servo1 - servo3)^2)
leg action scale = {servo2: 1.60, servo1: 1.25, servo4: 1.60, servo3: 1.25}
```

Stage 5 removes the roll task. Roll target remains zero while each parallel environment adapts its
yaw-rate range independently with the same four-success/two-failure promotion and demotion logic:

```text
level 0: yaw-rate target = 0.0 rad/s
level 1: yaw-rate target = [-0.4, 0.4] rad/s
level 2: yaw-rate target = [-0.8, 0.8] rad/s
level 3: yaw-rate target = [-1.2, 1.2] rad/s
```

Stage 6 uses the same per-environment yaw levels with rate-limited commands to match the real
control interface. Roll remains fixed at zero.

```text
Command resampling time = [2.0, 4.0] s
Velocity command rate limits = [0.7, 0.7, 1.2] for vx, vy, yaw_rate
Height command rate limit = 0.06 m/s

vx [-0.6, 0.6] m/s, height [0.18, 0.36] m, roll = 0
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
