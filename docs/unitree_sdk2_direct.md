# Unitree G1 + Dex3: Direct SDK2 Interface

This note describes how to control Unitree G1 (29 DoF) with the Dex3 end-effector from HIL-SERL by reusing the official Unitree control stack. Actions are converted to joint torques through Unitree's IK solver and the Dex3 commands flow through the dedicated shared-memory interface used by `Dex3_1_Controller`.
## Prerequisites

1. Install Unitree's SDK2 in the same environment that runs HIL-SERL:
   ```bash
   cd /localhome/local-vennw/code/unitree_sdk2_python
   pip install -e .
   ```

2. Clone the Unitree evaluation repository (already present at
   `/localhome/local-vennw/code/unitree_IL_lerobot`) and ensure it is visible
   on the Python path.  The environment automatically checks the environment
   variable `UNITREE_LEROBOT_ROOT`, so you can simply set:
   ```bash
   export UNITREE_LEROBOT_ROOT=/localhome/local-vennw/code/unitree_IL_lerobot
   ```
  
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

Arm angles are interpreted in radians. By default the action space clamps arm
targets to ±2.7 rad and Dex3 targets to the range `[0, 1]` (open → closed), but
you can override this by passing `arm_position_limit`, `hand_min_position`, and
`hand_max_position` to the environment.

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

Internally `UnitreeG1DirectEnv` calls:

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
