# Unitree G1 + Dex3: Direct SDK2 Interface

This note describes how to control Unitree G1 (29 DoF) with the Dex3 end-effector from HIL-SERL by reusing the official Unitree control stack. Actions are converted to joint torques through Unitree's IK solver and the Dex3 commands flow through the dedicated shared-memory interface used by `Dex3_1_Controller`.
## Prerequisites

1. Install Unitree's SDK2 in the same environment that runs HIL-SERL:
   ```bash
   cd /localhome/local-vennw/code/unitree_sdk2_python
   pip install -e .
   ```

2. install `unitree_IL_lerobot` in the environment:

   refer to: https://github.com/yiheng-wang-nv/unitree_IL_lerobot/tree/3-camera-eval

3. Start the Unitree simulator or connect to a real robot so that the SDK2 DDS
   topics are active.  Wait for the log line `DDS communication initialized`.

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
action = np.zeros(28, dtype=np.float32)
obs, reward, terminated, truncated, info = env.step(action)

# Return to the zero joint configuration before shutting down (optional but recommended)
env.go_home(steps=200)

# Small non-zero commands move the arms and hands via the official SDK2 stack.
env.close()
```

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
examples (`examples/low_level/lowlevel_control.py`).

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

1. The first 14 action values are passed through `solve_tau` and fed to
   `ctrl_dual_arm`, matching the SDK2 torque workflow.
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
- `/go_home` (POST): optionally `{"steps": 200}` to hold the zero joint configuration for the requested number of control cycles.

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
5. **Hold or reset**: send another action (e.g. all zeros or the initial joint vector from `get_state`) to keep the arms steady, or call `/go_home` to move back to the all-zero joint vector.

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
