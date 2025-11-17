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


# train bc
python examples/unitree_train_bc.py --exp_name unitree_install_trocar \
--dataset_dir /home/nvidia/workspace/yiheng/xr_teleoperate/teleop/utils/data/install_trocar_from_tray/debug \
--bc_checkpoint_path checkpoints/unitree_train_bc \
--train_steps 20000 --save_video False --debug False

# test BC eval
python examples/unitree_train_bc.py \
--exp_name unitree_install_trocar \
--bc_checkpoint_path /home/nvidia/workspace/yiheng/hil-serl/checkpoints/unitree_train_bc \
--eval_n_trajs 1 \
--unitree_video_host 192.168.123.164 \
--unitree_video_port 5555

# preprocess clf images
python process_clf_images.py /localhome/local-vennw/code/datasets/install_trocar_from_tray

# train reward classifier
python examples/unitree_train_reward_classifier.py --exp_name unitree_install_trocar \
--data_dir /localhome/local-vennw/code/datasets/install_trocar_from_tray/labels \
--num_classes 6 \
--val_episodes 10 \
--num_epochs 100 \
--batch_size 256 \
--image_height 512 \
--image_width 512 \
--output_dir /localhome/local-vennw/code/hil-serl/checkpoints/unitree_train_reward_classifier