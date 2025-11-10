# Unitree G1 + Dex3: Direct SDK2 Interface

Instructions for running Unitree control inside the HIL-SERL environment.

## Environment Setup

```bash
conda create -n hilserl python=3.10
conda activate hilserl
conda install -n hilserl -c conda-forge pinocchio=3.1.0 numpy=1.26.4

pip install --upgrade "jax[cuda12_pip]==0.4.35" jax jaxlib flax optax orbax-checkpoint -i https://pypi.tuna.tsinghua.edu.cn/simple

cd serl_launcher && pip install -e . && pip install -r requirements.txt && cd ..

# install unitree sdk2 python
cd ..
git clone https://github.com/unitreerobotics/unitree_sdk2_python.git
cd unitree_sdk2_python  && pip install -e .
cd ..
pip install flask
cd hil-serl
cd serl_robot_infra && pip install -e . && cd ..
# install unitree_IL_lerobot
cd ..
git clone --recurse-submodules git@github.com:yiheng-wang-nv/unitree_IL_lerobot.git
cd unitree_IL_lerobot
git submodule update --init --recursive
conda install pinocchio -c conda-forge
cd unitree_lerobot/lerobot && pip install -e .
cd ../../ && pip install -e .

# install XR teleoperation in the same env
cd ..
git clone https://github.com/unitreerobotics/xr_teleoperate.git
cd xr_teleoperate
git submodule update --init --depth 1
cd teleop/televuer && pip install -e . && openssl req -x509 -nodes -days 365 -newkey rsa:2048 -keyout key.pem -out cert.pem && cd ..
cd robot_control/dex-retargeting && pip install -e . && cd ../../..
pip install -r requirements.txt
cd ../hil-serl/
```

## Quick Start Example

```python
import numpy as np

from serl_robot_infra.unitree_env import UnitreeG1DirectEnv

env = UnitreeG1DirectEnv(arm="G1_29", ee="dex3", simulation=True, action_dt=0.02)
obs, info = env.reset()
print(obs.shape)
action = np.zeros(28, dtype=np.float32)
env.step(action)
env.close()
```

## Vision and Safety

```python
from serl_robot_infra.unitree_env import UnitreeVisionWrapper

env = UnitreeVisionWrapper(UnitreeG1DirectEnv(arm="G1_29", ee="dex3", simulation=True))
obs, info = env.reset()
print(obs.keys())
env.close()
```

## Teleop Bridge

```python
from serl_robot_infra.unitree_env import publish_xr_action

publish_xr_action(joint_targets)  # numpy array shaped (28,)
```

This helper matches the legacy SERL wrapper API. Typical deployments now drive the robot entirely from `xr_teleoperate` without using a Gym wrapper.

(Ensure the teleop Python environment has `pip install -e serl_robot_infra` if you reuse the bridge utilities.)

## Data Logging

Use `xr_teleoperate` to teleoperate the G1 and record demonstrations. The XR controller handles joint commands, camera capture, and success/failure labelling.

After each session copy the saved files into the HIL-SERL workspace for offline processing or training. HIL-SERL keeps only offline utilities; recording now lives entirely in the XR teleoperate repository.
