# 云端 MuJoCo 验收对接

本文说明如何在 Isaac Lab 云端训练机上运行 P1-P6 的 MuJoCo 验收门槛。验收是自动课程切换的一部分，不是可选的离线回放。

## 1. 运行拓扑

训练与验收必须使用不同的 Python 环境：

```text
Isaac Lab Python / CUDA
    |
    |-- 训练一个 PPO chunk，写入 model_*.pt
    |
独立 MuJoCo CPU Python (EVAL_PYTHON)
    |
    |-- 加载相同的 MJCF、34D observation builder 和 checkpoint
    |-- 以 5 ms physics / 50 Hz policy / 4 physics-step delay 运行固定验收轨迹
    |-- 写入 JSON 指标，并由 gate 判定是否进入下一阶段
```

MuJoCo 验收不启动 viewer、不需要 X11、Wayland、GPU 或 `DISPLAY`。它只使用一个 CPU 核心进行策略推理和物理仿真。不要把 `mujoco` 安装到 Isaac Sim 内置 Python；使用独立环境可避免 Isaac Sim、PyTorch 和系统 Python 的二进制依赖冲突。

自动脚本是 `scripts/train_deployment_curriculum.sh`。每完成一个 `CHUNK_ITERATIONS` 训练块，它依次调用：

1. `scripts/evaluate_mujoco_policy.py`：产生验收 JSON。
2. `scripts/check_policy_gate.py`：比较 JSON 与当前 P 阶段门槛。
3. 只有连续 `REQUIRED_PASS_CHECKS` 次通过，才会用该 checkpoint 启动下一阶段。

验收使用 `asset/wheeled_robot.xml`，包括全部六个 policy actuator 的 `+/-2 Nm` 限制。它同时检查部署用的 34 维观测顺序、六维动作映射、20 ms policy 周期和四个 5 ms physics-step 动作延迟。

## 2. 前置条件

以下示例假设云端路径为：

```bash
WORKSPACE="$HOME/gpufree-data/wheeled_rl"
ISAACLAB_ROOT="$HOME/gpufree-data/IsaacLab"
EVAL_VENV="$WORKSPACE/.venv-mujoco-eval"
```

要求：

- 64 位 Linux 和可用的 `python3.11`。仓库当前的 CPU PyTorch 固定为 `2.3.1+cpu`，建议不要用 Python 3.12 创建该验收环境。
- 工作树中包含当前 P1-P6 代码、`asset/wheeled_robot.xml` 和 `requirements-mujoco-inference.txt`。
- 云端训练和本机部署必须处于相同 Git 提交，且 `asset/wheeled_robot.xml` 的 SHA256 相同。
- 评估环境必须同时有 `mujoco==3.11.0` 与 `torch==2.3.1+cpu`。MuJoCo 会随 pip wheel 安装，通常不需要安装系统 OpenGL 或图形桌面包。

先核对路径和 Python：

```bash
test -x "$ISAACLAB_ROOT/isaaclab.sh"
python3.11 --version
cd "$WORKSPACE"
git rev-parse HEAD
sha256sum asset/wheeled_robot.xml
```

如果系统没有 `python3.11`，先在云端安装或使用已有的 Python 3.11；不要在尚未确认有对应 CPU PyTorch wheel 的 Python 版本上继续安装。

## 3. 创建独立验收环境

在云端执行一次：

```bash
cd "$WORKSPACE"
python3.11 -m venv "$EVAL_VENV"
"$EVAL_VENV/bin/python" -m pip install --upgrade pip
"$EVAL_VENV/bin/python" -m pip install -r requirements-mujoco-inference.txt
"$EVAL_VENV/bin/python" -m pip install \
  --index-url https://download.pytorch.org/whl/cpu \
  torch==2.3.1+cpu
```

不要在这一步使用 `isaaclab.sh -p`。MuJoCo evaluator 只需要 CPU PyTorch；Isaac Lab 训练仍由 `isaaclab.sh train` 选择其自身的解释器和 CUDA 运行时。

验证依赖、MJCF 和力矩范围：

```bash
cd "$WORKSPACE"
"$EVAL_VENV/bin/python" - <<'PY'
import mujoco
import torch

model = mujoco.MjModel.from_xml_path("asset/wheeled_robot.xml")
names = ("pos_servo2", "pos_servo1", "pos_servo4", "pos_servo3", "vel_wheel1", "vel_wheel2")
print("mujoco:", mujoco.__version__)
print("torch:", torch.__version__)
for name in names:
    actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
    print(name, tuple(model.actuator_forcerange[actuator_id]))
PY
```

预期六行 actuator 输出均为 `(-2.0, 2.0)`。

## 4. 首次验收冒烟测试

在开始长训练前，先用任意可加载的历史 checkpoint 验证环境。历史 checkpoint 不会通过新课程门槛，这是正常的；本步骤只验证 MuJoCo、PyTorch、MJCF 和模型加载链路。

```bash
cd "$WORKSPACE"
mkdir -p /tmp/wheeled_rl_mujoco_smoke

"$EVAL_VENV/bin/python" scripts/evaluate_mujoco_policy.py \
  --checkpoint logs/stage3_stance/model_399_stage3_contact_2nm.pt \
  --phase p1 \
  --output /tmp/wheeled_rl_mujoco_smoke/p1.json

cat /tmp/wheeled_rl_mujoco_smoke/p1.json
```

该命令成功时会输出 JSON，字段至少包括：

```text
phase, checkpoint, policy_steps, falls,
vx_rmse, yaw_rmse, stationary_speed_rms,
min_base_height, max_abs_roll, max_abs_pitch
```

不要用历史 checkpoint 执行 `check_policy_gate.py` 并期待成功。旧 Stage 权重是刻意排除的新课程起点，通常会被新的姿态或跟踪门槛拒绝。

## 5. 启动自动 P1-P6 训练

确认冒烟测试后，使用固定的环境变量启动：

```bash
cd "$WORKSPACE"

ISAACLAB_ROOT="$ISAACLAB_ROOT" \
EVAL_PYTHON="$EVAL_VENV/bin/python" \
SESSION_NAME="p1_p6_cloud_$(date +%Y%m%d_%H%M%S)" \
NUM_ENVS=10240 \
CHUNK_ITERATIONS=400 \
MAX_CHUNKS_PER_PHASE=16 \
REQUIRED_PASS_CHECKS=2 \
VIZ=kit \
./scripts/train_deployment_curriculum.sh
```

说明：

- `EVAL_PYTHON` 必须是绝对路径，不能依赖当前 shell 是否激活虚拟环境。
- `VIZ=kit` 是当前已验证的 Isaac Lab 启动方式；不要因为 MuJoCo 验收无界面，就擅自改变 Isaac Lab 的可视化配置。
- 每次训练块结束后，MuJoCo evaluator 会在 CPU 上运行数秒到数十秒。它不占用训练 GPU。
- P1 从随机权重开始。P2-P6 只从前一阶段通过验收的 checkpoint 续训，绝不从 `model_5691`、`model_1998`、历史 Stage 3 或 Stage 5/6 checkpoint 初始化。

## 6. 输出、恢复与重启

训练状态位于两个位置：

| 位置 | 内容 |
| --- | --- |
| `$ISAACLAB_ROOT/logs/rsl_rl/wheeled_legged_deployment/` | Isaac Lab training run、TensorBoard event 与 `model_*.pt` |
| `$WORKSPACE/logs/<SESSION_NAME>_auto/` | `resume.env`、每个 chunk 的 `p*_gate_*.json`、最终 `completed` 标记 |

`resume.env` 只在一个 training chunk 正常退出且 checkpoint 被找到后更新。若 SSH 中断、终端关闭或进程被停止，使用完全相同的环境变量再次执行启动命令即可从最近一次记录的 checkpoint 继续。不要手工编辑 `resume.env`。

仅在明确放弃当前 P1-P6 会话、准备重新从随机 P1 开始时，才使用：

```bash
RESET_SESSION=1 \
ISAACLAB_ROOT="$ISAACLAB_ROOT" \
EVAL_PYTHON="$EVAL_VENV/bin/python" \
SESSION_NAME="p1_p6_cloud_new" \
./scripts/train_deployment_curriculum.sh
```

`RESET_SESSION=1` 删除该 session 的 manifest、完成标志和 gate JSON，但不会删除 Isaac Lab 已有 run 目录。不要对仍需要恢复的 session 使用它。

## 7. 验收门槛与结果解释

每个 JSON 都经过以下通用限制：

```text
falls                = 0
min_base_height      >= 0.18 m
max_abs_roll         <= 0.45 rad
max_abs_pitch        <= 0.45 rad
```

阶段特定的最大误差如下：

| Phase | stationary_speed_rms (m/s) | vx_rmse (m/s) | yaw_rmse (rad/s) |
| --- | ---: | ---: | ---: |
| P1 | 0.10 | 0.12 | 0.20 |
| P2 | 0.12 | 0.20 | 0.25 |
| P3 | 0.14 | 0.28 | 0.30 |
| P4a | 0.15 | 0.35 | 0.32 |
| P4b | 0.16 | 0.42 | 0.34 |
| P4c | 0.18 | 0.50 | 0.36 |
| P5 | 0.18 | 0.52 | 0.35 |
| P6 | 0.20 | 0.55 | 0.40 |

门槛失败不会切换阶段。脚本会保留同一 P 阶段、从最新 checkpoint 再训练一个 chunk；连续通过计数会在一次失败后清零。某阶段达到 `MAX_CHUNKS_PER_PHASE` 仍无法连续通过时，脚本以非零状态退出，保留全部 checkpoint 和 JSON 供分析。

手工复查某次结果：

```bash
cd "$WORKSPACE"
"$EVAL_VENV/bin/python" scripts/check_policy_gate.py \
  logs/p1_p6_cloud_example_auto/p3_gate_4.json --phase p3
```

## 8. 常见故障

| 现象 | 检查与处理 |
| --- | --- |
| `EVAL_PYTHON must provide both mujoco and torch` | 运行 `"$EVAL_VENV/bin/python" -c 'import mujoco, torch'`；确认启动命令中的 `EVAL_PYTHON` 是该绝对路径。 |
| `ModuleNotFoundError: mujoco` | 在独立环境中重新执行 `pip install -r requirements-mujoco-inference.txt`，不要安装到 Isaac Sim Python。 |
| `ModuleNotFoundError: torch` 或 checkpoint 无法加载 | 为同一个 `EVAL_VENV` 安装 CPU PyTorch `2.3.1+cpu`；不要混用系统 `python` 和 venv 的 pip。 |
| `Unable to find ... model_*.pt` | 检查 `ISAACLAB_ROOT` 是否指向实际训练目录，以及该 chunk 是否已完成。训练脚本只会验收成功生成的 checkpoint。 |
| gate 显示 `falls`、高度或姿态失败 | 这是策略性能失败，不是 MuJoCo 安装失败。查看同名 `p*_gate_*.json`，保持当前 phase 继续训练或调整课程/奖励后重新开新 session。 |
| gate 显示跟踪误差失败 | 先确认 checkpoint、MJCF 和 Git SHA 未混用；之后检查该 phase 的 TensorBoard，再决定是否增加该 phase 的训练块。 |
| 图形相关报错、`DISPLAY` 缺失 | evaluator 不使用 viewer。确认运行的是 `evaluate_mujoco_policy.py`，不要对它附加 `--realtime` 或 MuJoCo GUI 参数。 |

## 9. 云端与本机一致性检查

每次准备回传候选 checkpoint 前，在两端分别执行：

```bash
cd /path/to/wheeled_rl
git rev-parse HEAD
sha256sum asset/wheeled_robot.xml
```

两项输出必须相同。若在一个 session 中修改了 MJCF、观测构造、动作映射、2 Nm 限幅或验收门槛，应停止该 session，并用新的 `SESSION_NAME` 和 `RESET_SESSION=1` 重新开始；旧 JSON 不可与新物理模型混合用于阶段晋级。
