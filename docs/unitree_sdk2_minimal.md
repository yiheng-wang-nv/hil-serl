# Minimal Unitree G1 SDK2 Bridge (Preview)

This note explains the minimal bridge that was added to help validate the
`unitree_sdk2_python` integration path without migrating the entire HIL-SERL
stack.  The goal is to provide a quick end-to-end smoke test:

1. run the HTTP server that forwards REST calls to Unitree DDS topics;
2. interact with the server via a tiny Gym environment;
3. confirm that the arm and Dex3 joints respond correctly.

## Implemented components

| File | Purpose | Reference code |
| ---- | ------- | -------------- |
| `serl_robot_infra/robot_servers/unitree_g1_server.py` | Flask server exposing `/joint_position`, `/open_gripper`, `/close_gripper`, `/getstate` endpoints. It bridges both the 14 arm joints and 14 Dex3 finger joints via the Unitree SDK2 DDS topics (arm code mirrors `G1_29_ArmController`, hand code mirrors `Dex3_1_Controller`). | Franka server structure from `serl_robot_infra/robot_servers/franka_server.py` |
| `serl_robot_infra/unitree_env/unitree_arm_env.py` | Minimal Gym env that sends 28-dimensional joint targets (arm + Dex3) to the server and reads back joint positions/velocities. Designed only for smoke testing. | Observation/action wiring pattern from `serl_robot_infra/franka_env/envs/franka_env.py` (step/reset methods) |
| `serl_robot_infra/unitree_env/__init__.py` | Package export for the new environment. | Conventional module init pattern |
| `docs/unitree_sdk2_minimal.md` | This guide. | — |

## Prerequisites

1. **Dependencies in the same Conda environment**  
   Activate `hilserl` (or whichever environment you use for HIL-SERL) and run:
   ```bash
   cd unitree_sdk2_python
   pip install -e .
   pip install flask
   ```
   The Unitree bridge imports SDK2 directly; installing it in a different environment will lead to `ModuleNotFoundError`.

## How to review the code

1. **Server logic**  
   - `G1ArmBridge` handles the torso/arm LowCmd topic (mirrors `G1_29_ArmController`).  
   - `Dex3Bridge` handles the hand topics (`rt/dex3/*`) similar to `Dex3_1_Controller`.  
   - Endpoints in `create_app()` split/merge the 28-D vectors and forward them to the two bridges.

2. **Environment behaviour**  
   - `UnitreeG1ArmEnv` clamps the 28-D action, posts it via `/joint_position`, and concatenates the returned arm/Dex3 states.  
   - Observations are limited to joint position/velocity; reward is always zero so that training loops can boot without extra logic.

3. **Safety / TODOs**  
   - No collision checks or soft limits are enforced yet.  
   - Reset logic should be extended once the kinematic home pose is known.  
   - Camera streams, reward signals, and human intervention wrappers still need to be ported from the Franka stack.

## Quick validation workflow


1. Launch the Unitree simulator. Wait for the log line `DDS communication initialized`.  
   *(Follow the official simulation instructions; for example, an Isaac-based workflow might look like `python sim_main.py --task Isaac-Simple-Wave-G129-Dex3-Joint --enable_dex3_dds --robot_type g129`.)*

2. Start the bridge server in a new terminal:
   ```bash
   conda activate hilserl
   python -m serl_robot_infra.robot_servers.unitree_g1_server --simulation --port 6000
   ```

3. Run the minimal environment test (28-D action for arm + Dex3):
   ```python
from serl_robot_infra.unitree_env import UnitreeG1ArmEnv
import numpy as np

env = UnitreeG1ArmEnv(server_url="http://127.0.0.1:6000/")
obs, info = env.reset()
action = np.zeros(28, dtype=np.float32)  # [14 arm | 7 left | 7 right]
obs, reward, terminated, truncated, info = env.step(action)
env.close()
```

4. Watch the server console output and the simulator:
   - If DDS is flowing, `/getstate` will return live joint data (verify with `curl -X POST http://127.0.0.1:6000/getstate`).
   - Try small non-zero entries in `action` to check that the simulated arm and Dex3 respond.

## Troubleshooting

- **`ModuleNotFoundError: No module named 'flask'`**  
  Install Flask inside the same environment (`pip install flask`).

- **`channel factory init error` or `failed to enumerate interfaces`**  
  Supply a proper `CYCLONEDDS_URI` (see prerequisites). This occurs when the selected interface does not exist or the machine is offline.

- **`[Reader] take sample error` spam**  
  The subscriber did not receive any `rt/lowstate` messages. Make sure the simulation/robot is running and the DDS topics are available.

- **HTTP 500 when calling `/joint_position` or `/getstate`**  
  Check the server log—DDS initialisation failures often bubble up as 500 responses. Resolve the DDS issue first, then retry.

## Next steps

- Add reset/randomisation logic and collision-aware safety checks.
- Expose camera feeds and reward classifiers similar to the Franka pipeline.
- Integrate the environment into the `examples/experiments` pipeline once the
  long-term observation/action structure is agreed upon.
