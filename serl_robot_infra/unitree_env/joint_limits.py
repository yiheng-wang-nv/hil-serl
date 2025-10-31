"""
Per-joint limits for Unitree G1 (arms + Dex3 hands).

The numbers are copied from the URDF shipped with the official Unitree control
stack: ``unitree_lerobot/eval_robot/assets/g1/g1_body29_hand14.urdf``.
Keep the ordering aligned with the action vector used by ``UnitreeG1DirectEnv``:

    [14 arm joints | 7 left-hand joints | 7 right-hand joints]
"""
from __future__ import annotations

import numpy as np

JOINT_ORDER = [
    # Arm – left
    ("left_shoulder_pitch_joint", -3.0892, 2.6704),  # [-176.998°, 153.003°]
    ("left_shoulder_roll_joint", -1.5882, 2.2515),  # [-90.997°, 129.001°]
    ("left_shoulder_yaw_joint", -2.6180, 2.6180),  # [-150.000°, 150.000°]
    ("left_elbow_joint", -1.0472, 2.0944),  # [-60.000°, 120.000°]
    ("left_wrist_roll_joint", -1.972222054, 1.972222054),  # [-113.000°, 113.000°]
    ("left_wrist_pitch_joint", -1.614429558, 1.614429558),  # [-92.500°, 92.500°]
    ("left_wrist_yaw_joint", -1.614429558, 1.614429558),  # [-92.500°, 92.500°]
    # Arm – right
    ("right_shoulder_pitch_joint", -3.0892, 2.6704),  # [-176.998°, 153.003°]
    ("right_shoulder_roll_joint", -2.2515, 1.5882),  # [-129.001°, 90.997°]
    ("right_shoulder_yaw_joint", -2.6180, 2.6180),  # [-150.000°, 150.000°]
    ("right_elbow_joint", -1.0472, 2.0944),  # [-60.000°, 120.000°]
    ("right_wrist_roll_joint", -1.972222054, 1.972222054),  # [-113.000°, 113.000°]
    ("right_wrist_pitch_joint", -1.614429558, 1.614429558),  # [-92.500°, 92.500°]
    ("right_wrist_yaw_joint", -1.614429558, 1.614429558),  # [-92.500°, 92.500°]
    # Dex3 – left
    ("left_hand_thumb_0_joint", -1.04719755, 1.04719755),  # [-60.000°, 60.000°]
    ("left_hand_thumb_1_joint", -0.72431163, 1.04719755),  # [-41.500°, 60.000°]
    ("left_hand_thumb_2_joint", 0.0, 1.74532925),  # [0.000°, 100.000°]
    ("left_hand_middle_0_joint", -1.57079632, 0.0),  # [-90.000°, 0.000°]
    ("left_hand_middle_1_joint", -1.74532925, 0.0),  # [-100.000°, 0.000°]
    ("left_hand_index_0_joint", -1.57079632, 0.0),  # [-90.000°, 0.000°]
    ("left_hand_index_1_joint", -1.74532925, 0.0),  # [-100.000°, 0.000°]
    # Dex3 – right
    ("right_hand_thumb_0_joint", -1.04719755, 1.04719755),  # [-60.000°, 60.000°]
    ("right_hand_thumb_1_joint", -1.04719755, 0.72431163),  # [-60.000°, 41.500°]
    ("right_hand_thumb_2_joint", -1.74532925, 0.0),  # [-100.000°, 0.000°]
    ("right_hand_index_0_joint", 0.0, 1.57079632),  # [0.000°, 90.000°]
    ("right_hand_index_1_joint", 0.0, 1.74532925),  # [0.000°, 100.000°]
    ("right_hand_middle_0_joint", 0.0, 1.57079632),  # [0.000°, 90.000°]
    ("right_hand_middle_1_joint", 0.0, 1.74532925),  # [0.000°, 100.000°]
]

JOINT_LIMITS = {
    name: (float(lower), float(upper))
    for name, lower, upper in JOINT_ORDER
}

JOINT_NAMES = [name for name, _, _ in JOINT_ORDER]
JOINT_LOWER_BOUNDS = np.array([lower for _, lower, _ in JOINT_ORDER], dtype=np.float32)
JOINT_UPPER_BOUNDS = np.array([upper for _, _, upper in JOINT_ORDER], dtype=np.float32)

__all__ = ["JOINT_ORDER", "JOINT_LIMITS", "JOINT_NAMES", "JOINT_LOWER_BOUNDS", "JOINT_UPPER_BOUNDS"]
