"""
Minimal Gym interface for Unitree G1 arm using the SDK2 HTTP bridge.

This environment mirrors the structure of FrankaEnv at a very small scale.
Its purpose is to test connectivity between HIL-SERL and the new
`unitree_g1_server.py` without depending on the full Franka stack.

Key characteristics:
    * action space: absolute joint targets for the 14 arm joints
    * observation: concatenated joint positions and velocities
    * reward: always zero (placeholder)

Before using this environment with real hardware, safety limits, reset logic,
and robust error handling must be added.
"""
from __future__ import annotations

from typing import Dict, Optional

import gymnasium as gym
import numpy as np
import requests


class UnitreeG1ArmEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        server_url: str = "http://127.0.0.1:6000/",
        joint_position_limit: float = 2.5,
        default_hold_pos: Optional[np.ndarray] = None,
    ):
        super().__init__()
        self.server_url = server_url.rstrip("/") + "/"

        self._num_joints = 14
        limit = float(joint_position_limit)
        high = np.ones(self._num_joints, dtype=np.float32) * limit
        self.action_space = gym.spaces.Box(-high, high, dtype=np.float32)

        obs_high = np.ones(self._num_joints * 2, dtype=np.float32) * np.inf
        self.observation_space = gym.spaces.Box(-obs_high, obs_high, dtype=np.float32)

        self._last_obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        self._hold_position = (
            np.array(default_hold_pos, dtype=np.float32)
            if default_hold_pos is not None
            else np.zeros(self._num_joints, dtype=np.float32)
        )

    # ------------------------------------------------------------------ utilities
    def _fetch_state(self) -> Dict[str, np.ndarray]:
        response = requests.post(self.server_url + "getstate", timeout=1.0)
        response.raise_for_status()
        data = response.json()
        positions = np.asarray(data.get("joint_positions", []), dtype=np.float32)
        velocities = np.asarray(data.get("joint_velocities", []), dtype=np.float32)
        if positions.size < self._num_joints or velocities.size < self._num_joints:
            raise RuntimeError("Unitree state message has insufficient length.")
        return {
            "joint_positions": positions[: self._num_joints],
            "joint_velocities": velocities[: self._num_joints],
        }

    def _send_joint_targets(self, targets: np.ndarray):
        payload = {"joint_positions": targets.astype(float).tolist()}
        response = requests.post(self.server_url + "joint_position", json=payload, timeout=1.0)
        response.raise_for_status()

    def _compose_obs(self, state: Dict[str, np.ndarray]) -> np.ndarray:
        obs = np.concatenate([state["joint_positions"], state["joint_velocities"]])
        self._last_obs = obs
        return obs

    # -------------------------------------------------------------------------- gym API
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        # Hold current position to avoid sudden jumps
        try:
            self._send_joint_targets(self._hold_position)
        except requests.RequestException:
            pass

        state = self._fetch_state()
        return self._compose_obs(state), {}

    def step(self, action: np.ndarray):
        clipped = np.clip(action, self.action_space.low, self.action_space.high)
        self._send_joint_targets(clipped)
        state = self._fetch_state()
        obs = self._compose_obs(state)
        reward = 0.0
        terminated = False
        truncated = False
        info: Dict[str, float] = {}
        return obs, reward, terminated, truncated, info

    def close(self):
        # Nothing to clean up for the HTTP client.
        return None

