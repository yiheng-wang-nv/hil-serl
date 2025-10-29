# Minimal Unitree G1 SDK2 Bridge (Preview)

This note explains the minimal bridge that was added to help validate the
`unitree_sdk2_python` integration path without migrating the entire HIL-SERL
stack.  The goal is to provide a quick end-to-end smoke test:

1. run the HTTP server that forwards REST calls to Unitree DDS topics;
2. interact with the server via a tiny Gym environment;
3. confirm that joints and the placeholder gripper command respond correctly.

## Implemented components

| File | Purpose | Reference code |
| ---- | ------- | -------------- |
| `serl_robot_infra/robot_servers/unitree_g1_server.py` | Flask server exposing `/joint_position`, `/open_gripper`, `/close_gripper`, `/getstate` endpoints. It internally wraps a DDS helper that mirrors the logic of Unitree's `G1_29_ArmController` (see `xr_teleoperate/teleop/robot_control/robot_arm.py` lines ~300+). | Franka server structure from `serl_robot_infra/robot_servers/franka_server.py` |
| `serl_robot_infra/unitree_env/unitree_arm_env.py` | Minimal Gym env that sends absolute joint targets to the server and reads back joint positions/velocities. Designed only for smoke testing. | Observation/action wiring pattern from `serl_robot_infra/franka_env/envs/franka_env.py` (step/reset methods) |
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
   - Start with `G1ArmBridge` inside `unitree_g1_server.py`.  It sets up DDS publishers/subscribers, keeps the latest low state snapshot, and streams the command message at 250 Hz.  
   - Endpoints in `create_app()` simply forward HTTP payloads to `G1ArmBridge` methods.  
   - Key differences vs Franka: only position control is exposed, and the Dex3 command is currently modelled as a simple wrist yaw placeholder.

2. **Environment behaviour**  
   - `UnitreeG1ArmEnv` clamps actions to a configurable joint limit, posts them via `/joint_position`, and pulls observations from `/getstate`.  
   - Observations are limited to joint position/velocity; reward is always zero so that training loops can boot without extra logic.

3. **Safety / TODOs**  
   - No collision checks or soft limits are enforced yet.  
   - Proper Dex3 finger control needs a dedicated DDS publisher.  
   - Reset logic should be extended once the kinematic home pose is known.

## Quick validation workflow


1. Launch the Unitree simulator. Wait for the log line `DDS communication initialized`.  
   *(Follow the official simulation instructions; for example, an Isaac-based workflow might look like `python sim_main.py --task Isaac-Simple-Wave-G129-Dex3-Joint --enable_dex3_dds --robot_type g129`.)*

2. Start the bridge server in a new terminal:
   ```bash
   conda activate hilserl
   python -m serl_robot_infra.robot_servers.unitree_g1_server --simulation --port 6000
   ```

3. Run the minimal environment test:
   ```python
   from serl_robot_infra.unitree_env import UnitreeG1ArmEnv
   import numpy as np

   env = UnitreeG1ArmEnv(server_url="http://127.0.0.1:6000/")
   obs, info = env.reset()
   action = np.zeros(14, dtype=np.float32)
   obs, reward, terminated, truncated, info = env.step(action)
   env.close()
   ```

5. Watch the server console output and the simulator:
   - If DDS is flowing, `/getstate` will return live joint data (you can verify with `curl -X POST http://127.0.0.1:6000/getstate`).
   - Try small non-zero entries in `action` to check that the simulated arm responds.

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

- Map the REST interface to the exact Dex3 finger state.
- Port reset/randomisation logic from the Franka environment.
- Integrate the environment into the `examples/experiments` pipeline once the
  long-term observation/action structure is agreed upon.
