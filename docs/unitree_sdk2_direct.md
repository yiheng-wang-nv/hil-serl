# Unitree G1 + Dex3: Direct SDK2 Interface

Instructions for running Unitree control inside the HIL-SERL environment.

## Environment Setup

```bash
conda create -n hilserl python=3.10
conda activate hilserl
conda install -n hilserl -c conda-forge pinocchio=3.1.0 numpy=1.26.4

pip install --upgrade "jax[cuda12_pip]==0.4.35" -i https://pypi.tuna.tsinghua.edu.cn/simple

cd serl_launcher && pip install -e . && pip install -r requirements.txt && cd ..
cd unitree_sdk2_python && pip install -e . && cd ..
pip install flask
# install unitree_IL_lerobot (editable) per https://github.com/yiheng-wang-nv/unitree_IL_lerobot/tree/3-camera-eval
cd serl_robot_infra && pip install -e . && cd ..

# optional: install XR teleoperation in the same env
cd ..
git clone https://github.com/unitreerobotics/xr_teleoperate.git
cd xr_teleoperate
git submodule update --init --depth 1
cd teleop/televuer && pip install -e . && openssl req -x509 -nodes -days 365 -newkey rsa:2048 -keyout key.pem -out cert.pem && cd ..
cd robot_control/dex-retargeting && pip install -e . && cd ../../..
pip install -r requirements.txt
pip install -e .
cd ..
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

`UnitreeXRIntervention` reads the latest teleop action, replaces `env.step` input, and records it in `info["intervene_action"]` so data-collection scripts capture the real command.

(Ensure the teleop Python environment has `pip install -e serl_robot_infra`.)
