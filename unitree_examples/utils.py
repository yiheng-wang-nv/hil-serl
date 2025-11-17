from __future__ import annotations

from typing import Dict

import numpy as np


def vector_to_teleop_action(action: np.ndarray) -> Dict[str, Dict[str, list]]:
    action = np.asarray(action, dtype=np.float32)
    if action.shape[0] != 28:
        raise ValueError(f"Expected action vector of length 28, got {action.shape}")

    left_arm = action[:7].tolist()
    right_arm = action[7:14].tolist()
    left_hand = action[14:21].tolist()
    right_hand = action[21:28].tolist()

    return {
        "left_arm": {"qpos": left_arm, "qvel": [], "torque": []},
        "right_arm": {"qpos": right_arm, "qvel": [], "torque": []},
        "left_ee": {"qpos": left_hand, "qvel": [], "torque": []},
        "right_ee": {"qpos": right_hand, "qvel": [], "torque": []},
        "body": {"qpos": []},
    }


def observation_to_teleop(obs: Dict[str, np.ndarray]) -> Dict[str, Dict[str, np.ndarray]]:
    colors: Dict[str, np.ndarray] = {}
    depths: Dict[str, np.ndarray] = {}

    idx = 0
    if "video.room_view_left" in obs and "video.room_view_right" in obs:
        colors[f"color_{idx}"] = np.array(obs["video.room_view_left"], copy=True)
        idx += 1
        colors[f"color_{idx}"] = np.array(obs["video.room_view_right"], copy=True)
        idx += 1
    elif "video.room_view" in obs:
        colors[f"color_{idx}"] = np.array(obs["video.room_view"], copy=True)
        idx += 1

    if "video.left_wrist_view" in obs:
        colors[f"color_{idx}"] = np.array(obs["video.left_wrist_view"], copy=True)
        idx += 1
    if "video.right_wrist_view" in obs:
        colors[f"color_{idx}"] = np.array(obs["video.right_wrist_view"], copy=True)

    robot_state = np.asarray(obs.get("robot_state", []), dtype=np.float32)
    states = {}
    if robot_state.size >= 28:
        pos = robot_state[:28]
        left_arm = pos[:7].tolist()
        right_arm = pos[7:14].tolist()
        left_hand = pos[14:21].tolist()
        right_hand = pos[21:28].tolist()
        states = {
            "left_arm": {"qpos": left_arm, "qvel": [], "torque": []},
            "right_arm": {"qpos": right_arm, "qvel": [], "torque": []},
            "left_ee": {"qpos": left_hand, "qvel": [], "torque": []},
            "right_ee": {"qpos": right_hand, "qvel": [], "torque": []},
            "body": {"qpos": []},
        }

    return {"colors": colors, "depths": {}, "states": states}
