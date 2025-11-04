# Unitree G1 + Dex3: Direct SDK2 Interface

This note describes how to control Unitree G1 (29 DoF) with the Dex3 end-effector from HIL-SERL by reusing the official Unitree control stack. Actions are converted to joint torques through Unitree's IK solver and the Dex3 commands flow through the dedicated shared-memory interface used by `Dex3_1_Controller`.

## Environment setup

1. **Create/activate Conda environment**
   ```bash
   conda create -n hilserl python=3.10
   conda activate hilserl
   ```

2. **Install JAX (CUDA 12 build)**
   ```bash
   pip install --upgrade "jax[cuda12_pip]==0.4.35" -i https://pypi.tuna.tsinghua.edu.cn/simple
   ```

3. **Install SERL Launcher**
   ```bash
   cd serl_launcher
   pip install -e .
   pip install -r requirements.txt
   cd ..
   ```

4. **Install Unitree SDK2 Python bindings**
   ```bash
   cd unitree_sdk2_python
   pip install -e .
   cd ..
   ```

5. **Install remaining Python dependencies**
   ```bash
   pip install flask
   ```

6. **Install Unitree LeRobot fork (Dex3 support)**
   Follow the instructions at https://github.com/yiheng-wang-nv/unitree_IL_lerobot/tree/3-camera-eval and install it in editable mode inside the same environment.

7. **Install the shared infrastructure package**
   ```bash
   cd serl_robot_infra
   pip install -e .
   cd ..
   ```

### Optional: install Unitree XR teleoperation in the same environment

If you plan to drive the robot with Unitree's XR controllers and stream actions into HIL‑SERL, install `xr_teleoperate` inside the *same* Conda env. The official instructions boil down to:

```bash
conda install -c conda-forge pinocchio=3.1.0 numpy=1.26.4
git clone https://github.com/unitreerobotics/xr_teleoperate.git
cd xr_teleoperate
git submodule update --init --depth 1

cd teleop/televuer
pip install -e .
openssl req -x509 -nodes -days 365 -newkey rsa:2048 -keyout key.pem -out cert.pem

cd ../robot_control/dex-retargeting
pip install -e .

cd ../../../
pip install -r requirements.txt
cd ..
```

With both repositories installed in editable mode, the teleop scripts can call `publish_xr_action(...)`, and HIL‑SERL can pick up the commands without additional path tweaks.
## Prerequisites

Start the Unitree simulator or connect to a real robot so that the SDK2 DDS topics are active.  Wait for the log line `DDS communication initialized`.

## Quick start

```python
import numpy as np

from serl_robot_infra.unitree_env import UnitreeG1DirectEnv

env = UnitreeG1DirectEnv(
    arm="G1_29",
    ee="dex3",
    simulation=True,      # set False when running on real hardware
    action_dt=0.02,
)

obs, info = env.reset()
print("observation shape:", obs.shape)

# Action layout: [14 arm joints | 7 left-hand joints | 7 right-hand joints]
action = np.zeros(28, dtype=np.float32)  # replace with your own command vector
obs, reward, terminated, truncated, info = env.step(action)

env.close()  # automatically returns to the zero joint configuration before releasing control

# If you only need the immediate robot state without issuing a new command:
state = env.observe()

# If you need safety guarding without cameras:
from serl_robot_infra.unitree_env import UnitreeSafetyWrapper
safe_env = UnitreeSafetyWrapper(env)
obs, info = safe_env.reset()
```

### Attaching camera streams

If you need RGB observations (for example to finetune a VLA policy on top of the Unitree deployment stack) wrap the base environment with the new vision helper:

```python
from serl_robot_infra.unitree_env import UnitreeVisionWrapper

env = UnitreeG1DirectEnv(arm="G1_29", ee="dex3", simulation=True)
vision_env = UnitreeVisionWrapper(env)  # safety is enabled by default

obs, info = vision_env.reset()
print(obs.keys())
# dict_keys(['robot_state', 'video.room_view', 'video.room_view_left',
#            'video.room_view_right', 'video.left_wrist_view', 'video.right_wrist_view'])

action = np.zeros(28, dtype=np.float32)
obs, reward, terminated, truncated, info = vision_env.step(action)

vision_env.close()
```

`UnitreeVisionWrapper` reuses `unitree_lerobot`'s `setup_image_client`, so the frames match the official teleoperation and evaluation pipelines.  The wrapper also enables the safety layer by default; pass `enable_safety=False` if you deliberately want to bypass it. Closing the wrapper releases the shared-memory buffers used by the camera client.

### XR teleoperation bridge

To feed actions from `xr_teleoperate` into HIL-SERL, import the helper once per control loop and publish the 28-D joint vector after you compute it:

```python
from serl_robot_infra.unitree_env import publish_xr_action

publish_xr_action(joint_targets)  # numpy array shaped (28,)
```

`UnitreeXRIntervention` (enabled automatically in the Unitree assemble experiment) picks up the latest command, replaces the action passed to `env.step`, and records it in `info["intervene_action"]`.  `unitree_record_demos.py` then captures those actions alongside observations, keeping teleop and demonstrations in sync.

（确保运行遥操作脚本的 Python 环境已经 `pip install -e hil-serl` 或把仓库根目录加入 `PYTHONPATH`，这样 `serl_robot_infra` 才能被正确导入。）

### Replaying a recorded episode

If you have demonstrations stored with LeRobot, use the helper script to reproduce them on the simulator or a real robot:

```bash
python unitree_examples/unitree_replay.py \
    --repo-id i4h/install_trocar \
    --episode 0 \
    --simulation \
    --frequency 50
```

The script loads the specified episode, aligns the robot with the recorded initial pose, then streams the stored joint targets at the requested control frequency.  Pass `--vision` if you want the replay loop to expose camera observations (useful for sanity checks) and `--no-safety` if you need to bypass the safety wrapper for debugging.

### Joint ordering

| Segment | Indices | Description |
| ------- | ------- | ----------- |
| Arm – left | 0–6 | Shoulder pitch/roll/yaw, elbow, wrist roll/pitch/yaw (`G1_29_JointIndex` arm subset) |
| Arm – right | 7–13 | Shoulder pitch/roll/yaw, elbow, wrist roll/pitch/yaw |
| Dex3 – left | 14–20 | Thumb joints 0–2, middle joints 0–1, index joints 0–1 (`Dex3_1_Left_JointIndex`) |
| Dex3 – right | 21–27 | Thumb joints 0–2, index joints 0–1, middle joints 0–1 (`Dex3_1_Right_JointIndex`) |

Every joint is clipped to the limits exported from
`unitree_lerobot/eval_robot/assets/g1/g1_body29_hand14.urdf`. The numerical
values (and joint names) live in `serl_robot_infra/unitree_env/joint_limits.py`.
If Unitree updates the URDF, regenerate that table and keep the ordering
aligned with the action vector documented above.

### Observations

`UnitreeG1DirectEnv` concatenates:
```
[arm joint positions, hand joint positions,
 arm joint velocities, zeros for hand velocity]
```
Dex3 velocities are not exposed by the shared memory and are therefore
reported as zeros.  If you only need positions you can slice the first 28
entries.

## How it works

`UnitreeG1DirectEnv` clamps actions to the URDF-sourced limits, so RL agents are
kept within the safe ranges defined by Unitree. The control loop sleeps for
`action_dt` seconds between steps; the default `0.02` matches Unitree's 50 Hz
examples (`examples/low_level/lowlevel_control.py`).  When you need a
Franka-style reset experience, wrap the environment with
`UnitreeSafetyWrapper` so every `reset()` performs a go-home, optional settle,
and gradual speed unlock before returning control to the agent. The wrapper also
tracks DDS heartbeats, controller ownership, and motor fault codes; if an
anomaly is detected it attempts to release the current mode, return to the safe
pose, re-select the control mode, and restart the soft-start sequence before
continuing.

Internally the environment calls:

```python
from unitree_lerobot.eval_robot.make_robot import setup_robot_interface
robot_if = setup_robot_interface(
    SimpleNamespace(arm="G1_29", ee="dex3", motion=False, sim=True)
)
```

which instantiates:

* `G1_29_ArmController` and `G1_29_ArmIK` for the arms;
* `Dex3_1_Controller` for the hand (with shared-memory shuttles for actions/states).

During each step:

1. The first 14 action values run through Unitree's `clip_arm_q_target`
   (velocity-limited interpolation) before solving IK torques.
2. The clipped targets and torques are sent to `ctrl_dual_arm`, matching the SDK2 workflow.
2. The remaining values are written to the Dex3 shared memory, exactly like
   the official evaluation scripts (`ee_shared_mem["left"]` / `["right"]`).
3. Observations are built from `arm_ctrl.get_current_dual_arm_q/dq()` and the
   current Dex3 joint positions cached in shared memory.

This means the environment behaves the same way as the deployment stack (e.g.
`unitree_lerobot/eval_robot/eval_g1_gr00t.py`), so policies trained in HIL-SERL
will see the same control semantics as at evaluation time.

## Next steps

* Integrate the new environment into the `examples/experiments` pipeline by
  supplying a `TrainConfig` that exposes the 28-D action/observation keys.
* Extend reset/safety logic (e.g. “go to home pose”, collision limits).
* Add camera/reward wrappers if needed (the `setup_image_client` helper from
  Unitree can be reused in a similar fashion).

## Optional HTTP wrapper

For compatibility with existing Franka-style tooling, a lightweight HTTP server is available:
```bash
python -m serl_robot_infra.robot_servers.unitree_g1_server --simulation
```
It exposes:
- `/set_action` (POST): JSON `{"action": [...]} ` with 28 values `[14 arm | 7 left dex3 | 7 right dex3]`.
- `/get_state` (GET): returns the latest observation.
- `/go_home` (POST): optionally `{"steps": 200}` to hold the zero joint configuration for the requested number of control cycles (the server also calls this automatically when shutting down).

Internally the server delegates to `UnitreeG1DirectEnv`, so control semantics remain identical.  
The server keeps streaming the most recent action at 50 Hz, so the
simulation/robot continues to receive commands even when you are not sending new HTTP requests.

Example request sequence (assuming the server runs on `localhost:6000`):

```bash
# Query the latest observation
curl -s http://127.0.0.1:6000/get_state | jq

# Send a small joint command (here: +0.05 rad on both elbows)
curl -s -X POST http://127.0.0.1:6000/set_action \
     -H "Content-Type: application/json" \
     -d '{"action": [0,0,0,0.05,0,0,0,  0,0,0,0.05,0,0,0,  0,0,0,0,0,0,0,  0,0,0,0,0,0,0]}'

# Drive the robot back to the initial pose (hold for ~4 s)
curl -s -X POST http://127.0.0.1:6000/go_home -H "Content-Type: application/json" -d '{"steps": 200}'
```

The server replies with `{"status": "ok"}` when the payload is accepted. Values outside the joint limits are clipped using the URDF-derived bounds documented above.

### End-to-end verification workflow

1. **Launch the Unitree simulator** (`unitree_sim_isaaclab`) with the DDS pipeline (e.g. `Isaac-Simple-Wave-G129-Dex3-Joint`). Wait for the console message: `DDS communication initialized`.
2. **Start the HTTP bridge**:
   ```bash
   (hilserl) python -m serl_robot_infra.robot_servers.unitree_g1_server --simulation --port 6000
   ```
   You should see the controller thread logging from Isaac and the Flask server banner.
3. **Check the current state**:
   ```bash
   curl -s http://127.0.0.1:6000/get_state | jq '.arm_joint_positions'
   ```
   The 14 returned numbers are the arm joint angles (radians). They should match the pose shown in Isaac.
4. **Send a test command**:
   ```bash
   # Add +0.2 rad to the left elbow (index 3)
   curl -s -X POST http://127.0.0.1:6000/set_action \
        -H "Content-Type: application/json" \
        -d '{"action": [0,0,0,0.2,0,0,0,  0,0,0,0,0,0,0,  0,0,0,0,0,0,0,  0,0,0,0,0,0,0]}'
   ```
   Within a second you should see the left elbow bend forward in simulation. A subsequent `get_state`
   call will show the elbow angle close to `0.2` rad, confirming the two-way connection.
5. **Hold or reset**: send another action (e.g. all zeros or the initial joint vector from `get_state`) to keep the arms steady, or call `/go_home` to move back to the all-zero joint vector (the HTTP server will invoke this automatically on exit).

## Suggested Development Roadmap

1. **Core Environment reset & safety** – wrap `UnitreeG1DirectEnv` in a task-specific Gym env that handles go-to-home, randomisation, safe joint limits, and error recovery.
2. **Observation wrappers** – reimplement Franka-style wrappers (RelativeFrame, Chunking, SpacemouseIntervention) and integrate camera streams.
3. **Reward / termination** – define task-specific rewards, success criteria, and episode termination.
4. **Task configs & scripts** – add Unitree entries under `examples/experiments/` with `TrainConfig`, `run_actor.sh`, `run_learner.sh`, etc.
5. **Data collection tools** – update `record_demos.py`, `record_success_fail.py`, and dataset loaders for Unitree observations/actions.
6. **Evaluation & logging** – integrate video logging, metrics, and optional HTTP control (already provided).

Tackle these steps in order to reach feature parity with the existing Franka pipeline.

## Maintaining joint limits

`serl_robot_infra/unitree_env/joint_limits.py` mirrors the limits encoded in
`unitree_lerobot/eval_robot/assets/g1/g1_body29_hand14.urdf`. When Unitree
releases a new URDF, re-run the extraction script (or update the table manually)
so the environment clips actions to the official ranges for both the arms and
the Dex3 hands.

## Leveraging Unitree safety features

`UnitreeG1DirectEnv` now proxies the velocity-limited interpolation
(`clip_arm_q_target`), gradual speed ramp (`speed_gradual_max`), instant unlock
(`speed_instant_max`), and SDK-provided go-home sequence
(`ctrl_dual_arm_go_home`). When the controller exposes these helpers they are
used automatically (e.g., during `step` and `close`), and you can call the
wrapper methods directly for custom scripts:

```python
env.speed_gradual_max(duration=5.0)  # soft-start arm velocity limit
env.speed_instant_max()              # jump to max velocity when needed
```
