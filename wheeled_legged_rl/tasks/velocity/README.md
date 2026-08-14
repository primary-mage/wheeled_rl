# Wheeled-Legged P1-P6 Curriculum

This is the only active training curriculum for this repository. The former `Stage1` through `Stage6` tasks, adaptive height/yaw levels, and their training scripts have been removed. Do not initialize a new run from historical `model_5691`, `model_1998`, `model_399`, or Stage 5/6 checkpoints.

P0 is a model and actuator-alignment check, not an RL task. It verifies the USD/MJCF model, joint order, 34-D observation construction, 6-D action mapping, and 2 Nm actuator limits before P1 begins from random initialization.

The base frame is `+X` forward, `+Y` left, `+Z` up. Every task uses a 5 ms physics step, decimation 4, and therefore a 20 ms / 50 Hz policy step.

## Curriculum

| Phase | Task | Focus | Longitudinal range | Yaw range |
| --- | --- | --- | --- | --- |
| P1 | `Isaac-WheeledLegged-P1-v0` | Stationary lock and recovery from small pushes | 0 | 0 |
| P2 | `Isaac-WheeledLegged-P2-v0` | Low-speed forward/backward motion plus frequent holds | 0.05-0.25 m/s | 0 |
| P3 | `Isaac-WheeledLegged-P3-v0` | Start, brake, and reverse through zero | up to 0.50 m/s | 0 |
| P4a | `Isaac-WheeledLegged-P4A-v0` | High-speed straight-line stability | up to 0.60 m/s | 0 |
| P4b | `Isaac-WheeledLegged-P4B-v0` | High-speed straight-line stability | up to 0.80 m/s | 0 |
| P4c | `Isaac-WheeledLegged-P4C-v0` | Full straight-line envelope | up to 1.00 m/s | 0 |
| P5 | `Isaac-WheeledLegged-P5-v0` | Coupled velocity and yaw tracking | up to 1.00 m/s | +/-0.50 rad/s |
| P6 | `Isaac-WheeledLegged-P6-v0` | Sim-to-real robustness across the complete operating envelope | up to 1.00 m/s | +/-0.50 rad/s |

All moving phases intentionally mix four maneuver types: stationary hold, cruise, braking to zero, and reversal through zero. Commands are rate limited, so a reversal cannot jump directly from positive to negative speed. P5/P6 also enforce `abs(vx * yaw_rate) <= 0.35 m/s^2`; at high speed the sampled yaw-rate is reduced to a physically reasonable value.

Height and roll are no longer task commands. Their observation slots remain present for deployment compatibility, but their targets are always `0.270 m` and `0 rad`. The policy learns to regulate them as balance variables, rather than attempting a separate posture-command task.

P1-P5 use nominal dynamics. P6 enables wheel friction randomization, whole-body mass scaling, base center-of-mass offsets, reset perturbations, and periodic base-velocity pushes. Every phase terminates on excessive body height/orientation error, lower-leg or foot contact, or a wheel scissor offset. The lower-leg/foot contact condition is specifically retained to exclude the previous kneeling strategy.

## Interface Contract

The model interface is unchanged across P1-P6 and must also remain unchanged in MuJoCo and the ROS 2 deployment workspace.

```text
observation[ 0: 3]  base linear velocity in base frame       [vx, vy, vz]
observation[ 3: 6]  base angular velocity in base frame      [wx, wy, wz]
observation[ 6: 9]  projected gravity in base frame
observation[ 9:12]  command                                [vx, vy=0, yaw_rate]
observation[12]     fixed height target                     0.270 m
observation[13]     fixed roll target                       0 rad
observation[14]     measured base height
observation[15]     measured base roll
observation[16:22]  joint position relative to default      [servo2, servo4, servo1, servo3, wheel1, wheel2]
observation[22:28]  joint velocity in the same order
observation[28:34]  previous raw policy action in action order
```

```text
raw_action[0] -> servo2: clip( 0.9 + 1.60 * a0, -1.57,  1.57)
raw_action[1] -> servo1: clip(-1.9 + 1.25 * a1, -3.14,  0.00)
raw_action[2] -> servo4: clip( 0.9 + 1.60 * a2, -1.57,  1.57)
raw_action[3] -> servo3: clip(-1.9 + 1.25 * a3, -3.14,  0.00)
raw_action[4] -> wheel1 velocity target: 8.0 * a4 rad/s
raw_action[5] -> wheel2 velocity target: 8.0 * a5 rad/s
```

The Isaac and MJCF actuator effort limits are both `+/-2 Nm`. The policy is evaluated at 50 Hz. The training model randomizes an actuator delay of 0-4 physics steps; deployment/MuJoCo evaluation uses four 5 ms steps (20 ms).

## Automatic Global Promotion

Run the complete sequence from the Isaac Lab root with:

```bash
cd /path/to/wheeled_rl
ISAACLAB_ROOT=/path/to/IsaacLab \
EVAL_PYTHON=/path/to/mujoco-python \
./scripts/train_deployment_curriculum.sh
```

`EVAL_PYTHON` must have `mujoco` and `torch` installed. The script trains each phase in `CHUNK_ITERATIONS` blocks, writes the most recent resume point to `logs/p1_p6_deployment_auto/resume.env`, evaluates it with `scripts/evaluate_mujoco_policy.py`, and promotes it only after `REQUIRED_PASS_CHECKS` consecutive gate passes. It is restartable; set `RESET_SESSION=1` only when deliberately starting a new P1 run.

For a cloud setup that keeps Isaac Lab training and the CPU-only MuJoCo evaluator separate, follow [the cloud MuJoCo acceptance guide](../../../docs/cloud_mujoco_acceptance.md).

Useful overrides:

```bash
NUM_ENVS=10240 CHUNK_ITERATIONS=400 MAX_CHUNKS_PER_PHASE=16 VIZ=kit \
ISAACLAB_ROOT=/path/to/IsaacLab EVAL_PYTHON=python \
./scripts/train_deployment_curriculum.sh
```

The automated MuJoCo gate rejects falls, low base height, excessive roll/pitch, uncontrolled stationary creep, and phase-specific velocity/yaw tracking error. It is a promotion filter, not a substitute for final Isaac visual inspection or hardware safety testing.

To train one phase manually, use its task ID and seed it only from the preceding P-phase checkpoint:

```bash
cd /path/to/IsaacLab
PYTHONPATH=/path/to/wheeled_rl \
./isaaclab.sh train --rl_library rsl_rl \
  --task Isaac-WheeledLegged-P3-v0 \
  --external_callback wheeled_legged_rl.tasks.register.register_tasks \
  --resume --load_run <previous_run> --checkpoint <previous_model.pt> \
  --num_envs 4096 --viz kit
```

## Playback

`Isaac-WheeledLegged-P6-Play-v0` is deterministic: observation noise, domain randomization, pushes, and reset perturbations are disabled. It is the task to use for Isaac/MuJoCo log alignment with a P6 checkpoint.

```bash
cd /path/to/IsaacLab
PYTHONPATH=/path/to/wheeled_rl \
./isaaclab.sh play --rl_library rsl_rl \
  --task Isaac-WheeledLegged-P6-Play-v0 \
  --external_callback wheeled_legged_rl.tasks.register.register_tasks \
  --checkpoint /absolute/path/to/model.pt --num_envs 1 --viz kit
```

For MuJoCo playback, pass an explicit P1-P6 checkpoint. The CLI exposes only the trained command interface: forward speed and yaw-rate. Height, roll, and lateral velocity remain fixed by design.

```bash
python scripts/play_mujoco.py \
  --model asset/wheeled_robot.xml \
  --checkpoint /absolute/path/to/model.pt \
  --vx 0.0 --yaw-rate 0.0 \
  --action-delay-steps 4 --realtime \
  --log-csv logs/mujoco_p6.csv
```

Terminal controls are `W/S` for `+/-0.1 m/s` increments up to `+/-1.0 m/s`, `A/D` for `+/-0.1 rad/s` increments up to `+/-0.5 rad/s`, and Space to clear both commands.
