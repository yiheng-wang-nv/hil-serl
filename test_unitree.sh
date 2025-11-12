# test direct environment with simulation
python test_unitree_direct_env.py --simulation
# test direct environment with real robot
python test_unitree_direct_env.py --steps 100
# test vision environment
export PYTHONPATH=/home/nvidia/workspace/yiheng/hil-serl:$PYTHONPATH
python test_unitree_vision_wrapper.py --steps 100
# test replay episode
python test_unitree_replay_episode.py \
--dataset_dir /home/nvidia/workspace/yiheng/xr_teleoperate/teleop/utils/data/install_trocar_from_tray/debug \
--episode 0 \
--steps -1

# test BC eval
python examples/unitree_train_bc.py \
--exp_name unitree_install_trocar \
--bc_checkpoint_path /home/nvidia/workspace/yiheng/hil-serl/checkpoints/unitree_train_bc \
--eval_n_trajs 5 \
--unitree_video_host 192.168.123.164 \
--unitree_video_port 5555