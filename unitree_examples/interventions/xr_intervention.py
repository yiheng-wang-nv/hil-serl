"""Wrapper that injects XR teleop actions into Unitree environments."""

from __future__ import annotations

from typing import Optional

import gymnasium as gym
import numpy as np

from serl_robot_infra.unitree_env.xr_action_bridge import (
    close_bridge,
    fetch_latest_action,
)
from unitree_examples.utils import vector_to_teleop_action


class UnitreeXRIntervention(gym.Wrapper):
    """Replace actions with XR teleoperation inputs when available."""

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self._last_seq = 0.0
        self._speed_unlocked = False

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        obs, info = self.env.reset(seed=seed, options=options)
        info = dict(info)
        info.setdefault("intervene_action", None)
        return obs, info

    def step(self, action):
        candidate = np.asarray(action, dtype=np.float32)
        override, updated, self._last_seq = fetch_latest_action(self._last_seq)

        if updated:
            candidate = override
            if not self._speed_unlocked:
                controller = self.env.unwrapped
                speed_fn = getattr(controller, "speed_instant_max", None)
                if callable(speed_fn):
                    try:
                        speed_fn()
                    except Exception:
                        pass
                self._speed_unlocked = True

        teleop_action = vector_to_teleop_action(candidate)
        action_vector = np.asarray(candidate, dtype=np.float32).copy()

        candidate = np.clip(candidate, self.action_space.low, self.action_space.high)

        obs, reward, terminated, truncated, info = self.env.step(candidate)
        info = dict(info)
        info["intervene_action"] = teleop_action
        info["intervene_action_vector"] = action_vector
        info["intervene_override"] = updated
        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        try:
            close_bridge()
        finally:
            super().close()
