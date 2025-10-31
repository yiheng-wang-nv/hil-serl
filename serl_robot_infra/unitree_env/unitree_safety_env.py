"""
Safety wrapper for Unitree G1 + Dex3 environments.

This wrapper mirrors the reset/go-home behaviour we rely on in the Franka
pipeline: every reset first reinitialises the underlying env, then drives the
arms back to the zero joint configuration using Unitree's official
``ctrl_dual_arm_go_home`` helper, and optionally applies the gradual speed
unlock so the subsequent commands start from a safe velocity profile.
"""
from __future__ import annotations

import time
from typing import Any, Dict, Optional

import gymnasium as gym


class UnitreeSafetyWrapper(gym.Wrapper):
    """Wrap ``UnitreeG1DirectEnv`` with safety-aware reset and go-home logic."""

    def __init__(
        self,
        env: gym.Env,
        *,
        go_home_steps: int = 200,
        settle_time: float = 0.5,
        soft_start_duration: Optional[float] = 5.0,
    ) -> None:
        """
        Args:
            env: An instance of :class:`UnitreeG1DirectEnv` (or compatible API).
            go_home_steps: Number of control cycles to hold the zero pose if the
                SDK helper is unavailable (fallback path). Default ~4 s at 50 Hz.
            settle_time: Optional pause after go-home to let the robot stabilise
                before the next command.
            soft_start_duration: When provided, call
                ``env.speed_gradual_max(duration)`` on every reset so the arm
                velocity limit ramps up smoothly. Set to ``None`` to disable.
        """
        super().__init__(env)
        self._go_home_steps = go_home_steps
        self._settle_time = settle_time
        self._soft_start_duration = soft_start_duration

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ):
        obs, info = self.env.reset(seed=seed, options=options)

        # Drive both arms (and Dex3) to the all-zero joint configuration.
        obs = self.env.go_home(steps=self._go_home_steps)
        if self._settle_time > 0.0:
            time.sleep(self._settle_time)
            obs = self.env.observe()

        if self._soft_start_duration is not None:
            self.env.speed_gradual_max(duration=self._soft_start_duration)

        if info is None:
            info = {}
        info["safety_reset"] = True
        return obs, info

    def close(self) -> None:
        try:
            self.env.go_home(steps=self._go_home_steps)
        finally:
            super().close()
