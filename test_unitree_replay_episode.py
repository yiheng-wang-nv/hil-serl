#!/usr/bin/env python3
"""
Replay a recorded Unitree XR episode directly on the robot.

Loads the actions from a dataset episode (data.json format) and sequentially
plays them back through UnitreeG1DirectEnv. Useful for verifying joint ordering
and scaling before running BC policies.
"""

import argparse
import json
import os
import time
from typing import List, Tuple

import numpy as np

from serl_robot_infra.unitree_env.unitree_robot_env import UnitreeG1DirectEnv


def _scan_episode_dirs(root: str) -> List[str]:
    if os.path.isfile(os.path.join(root, "data.json")):
        return [root]
    subdirs = [
        os.path.join(root, d)
        for d in sorted(os.listdir(root))
        if d.startswith("episode_") and os.path.isdir(os.path.join(root, d))
    ]
    if not subdirs:
        raise ValueError(f"No episode_* directories or data.json found under {root}")
    return subdirs


def load_episode_actions(root: str, episode_index: int) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    episode_dirs = _scan_episode_dirs(root)
    if episode_index < 0 or episode_index >= len(episode_dirs):
        raise IndexError(f"Episode index {episode_index} out of range (total {len(episode_dirs)})")

    episode_path = episode_dirs[episode_index]
    data_path = os.path.join(episode_path, "data.json")
    with open(data_path, "r") as f:
        payload = json.load(f)
    frames = payload.get("data", [])
    if len(frames) < 2:
        raise ValueError(f"Episode {episode_path} has fewer than 2 frames.")

    actions = []
    states = []
    for fr in frames:
        action = fr.get("actions", {})
        state = fr.get("states", {})
        def _get_qpos(tree, key):
            return np.array(tree.get(key, {}).get("qpos", []), dtype=np.float32)
        act_vec = np.concatenate(
            [
                _get_qpos(action, "left_arm"),
                _get_qpos(action, "right_arm"),
                _get_qpos(action, "left_ee"),
                _get_qpos(action, "right_ee"),
            ],
            axis=0,
        )
        state_vec = np.concatenate(
            [
                _get_qpos(state, "left_arm"),
                _get_qpos(state, "right_arm"),
                _get_qpos(state, "left_ee"),
                _get_qpos(state, "right_ee"),
            ],
            axis=0,
        )
        actions.append(act_vec)
        states.append(state_vec)

    return actions, states


def pad_action(action: np.ndarray, target_shape: Tuple[int, ...]) -> np.ndarray:
    target_dim = target_shape[0]
    if action.shape[0] == target_dim:
        return action.astype(np.float32, copy=True)
    padded = np.zeros(target_dim, dtype=np.float32)
    length = min(target_dim, action.shape[0])
    padded[:length] = action[:length]
    return padded


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_dir", required=True, help="Path containing episode_*/data.json")
    parser.add_argument("--episode", type=int, default=0, help="Episode index to replay.")
    parser.add_argument("--steps", type=int, default=-1, help="Max number of steps to replay (-1 = full episode).")
    parser.add_argument("--sleep", type=float, default=0.02, help="Delay between steps (seconds).")
    parser.add_argument("--simulation", action="store_true", help="Use simulator DDS channel.")
    parser.add_argument("--motion_mode", action="store_true", help="Enable Unitree motion blending.")
    parser.add_argument("--action_dt", type=float, default=0.02, help="Env control period.")
    args = parser.parse_args()

    actions, states = load_episode_actions(args.dataset_dir, args.episode)
    env = UnitreeG1DirectEnv(
        simulation=args.simulation,
        motion_mode=args.motion_mode,
        action_dt=args.action_dt,
    )

    try:
        obs, info = env.reset()
        print("Reset info:", info)
        max_steps = len(actions) if args.steps < 0 else min(args.steps, len(actions))
        for idx in range(max_steps):
            action = pad_action(actions[idx], env.action_space.shape)
            obs, reward, terminated, truncated, step_info = env.step(action)
            print(
                f"Step {idx + 1}/{max_steps}: reward={reward}, terminated={terminated}, "
                f"truncated={truncated}, obs[:5]={np.asarray(obs)[:5]}"
            )
            if args.sleep > 0:
                time.sleep(args.sleep)
            if terminated or truncated:
                print("Environment finished episode early; resetting...")
                obs, info = env.reset()
    finally:
        time.sleep(1.0)
        env.close()


if __name__ == "__main__":
    main()
