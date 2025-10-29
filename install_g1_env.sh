## Installation
1. **Setup Conda Environment:**
    create an environment with
    ```bash
    conda create -n hilserl python=3.10
    ```

2. **Install Jax as follows:**

```
pip install --upgrade "jax[cuda12_pip]==0.4.35" -i https://pypi.tuna.tsinghua.edu.cn/simple
```

3. **Install the serl_launcher**
    ```bash
    cd serl_launcher
    pip install -e .
    pip install -r requirements.txt
    ```

4. **Install Unitree Dependencies**

```bash
cd unitree_sdk2_python
pip install -e .
```

5. **Install other dependencies**

```bash
pip install flask
```