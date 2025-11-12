# test direct environment with simulation
python test_unitree_direct_env.py --simulation
# test direct environment with real robot
python test_unitree_direct_env.py --steps 100
# test vision environment
export PYTHONPATH=/home/nvidia/workspace/yiheng/hil-serl:$PYTHONPATH
python test_unitree_vision_wrapper.py --steps 100