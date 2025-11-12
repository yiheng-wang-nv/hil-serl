# Example: training with XR teleoperation dataset
# Set JAX_PLATFORMS=cpu to force CPU if GPU compilation has issues
# export JAX_PLATFORMS=cpu

# export XLA_FLAGS="--xla_gpu_autotune_level=0 --xla_gpu_enable_triton_gemm=false"
# export TF_CUDNN_DETERMINISTIC=1
# export XLA_PYTHON_CLIENT_PREALLOCATE=false
# export CUDA_LAUNCH_BLOCKING=1

# python - <<'PY'
# import os, platform, jax, jaxlib
# print(">>> env check")
# print("jax:", jax.__version__)
# print("jaxlib:", jaxlib.__version__)
# print("devices:", jax.devices())
# print("XLA_FLAGS:", os.environ.get("XLA_FLAGS"))
# print("TF_CUDNN_DETERMINISTIC:", os.environ.get("TF_CUDNN_DETERMINISTIC"))
# print("XLA_PYTHON_CLIENT_PREALLOCATE:", os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE"))
# print("CUDA_LAUNCH_BLOCKING:", os.environ.get("CUDA_LAUNCH_BLOCKING"))
# print("platform:", platform.platform())
# print("<<< env check")
# PY

python examples/unitree_train_bc.py --exp_name unitree_install_trocar \
--dataset_dir /home/nvidia/workspace/yiheng/xr_teleoperate/teleop/utils/data/install_trocar_from_tray/debug \
--bc_checkpoint_path checkpoints/unitree_train_bc \
--train_steps 20000 --save_video False --debug False
