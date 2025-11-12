# test direct environment with simulation
python test_unitree_direct_env.py --simulation --action_dt 0.02
# test direct environment with real robot
python test_unitree_direct_env.py --steps 10 --action_dt 0.02
# test vision environment
export PYTHONPATH=/home/nvidia/workspace/yiheng/hil-serl:$PYTHONPATH
python test_unitree_vision_wrapper.py