"""
Safety wrapper for Unitree G1 + Dex3 environments.

This wrapper mirrors the reset/go-home behaviour used on Franka: every reset
first pushes the robot to a known safe pose (zero joint configuration) and
optionally performs a soft-start on the joint velocity limits. In addition, it
continuously monitors the Unitree SDK2 lowstate stream, motor fault flags, and
controller ownership. Whenever an anomaly is detected, the wrapper attempts to
recover automatically by releasing the current mode, returning to the safe pose,
re-selecting the control mode, and reapplying the soft-start helper provided by
SDK2.
"""
from __future__ import annotations

import time
from typing import Any, Dict, Optional

import gymnasium as gym

try:
    from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
except ImportError:  # pragma: no cover - optional in some environments
    MotionSwitcherClient = None


class UnitreeSafetyWrapper(gym.Wrapper):
    """Safety-aware wrapper around :class:`UnitreeG1DirectEnv`."""

    def __init__(
        self,
        env: gym.Env,
        *,
        go_home_steps: int = 200,
        settle_time: float = 0.5,
        soft_start_duration: Optional[float] = 5.0,
        health_timeout: float = 0.5,
    ) -> None:
        super().__init__(env)
        self._go_home_steps = go_home_steps
        self._settle_time = settle_time
        self._soft_start_duration = soft_start_duration
        self._health_timeout = health_timeout

        self._last_lowstate_stamp = time.monotonic()
        self._recovering = False
        self._motion_switcher = None
        self._monitor_control_mode = False

    # ------------------------------------------------------------------ gym API
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ):
        obs, info = self.env.reset(seed=seed, options=options)
        self._snapshot_lowstate()
        self._init_mode_monitor()

        obs = self._drive_home()
        self._apply_soft_start()

        if info is None:
            info = {}
        info["safety_reset"] = True
        return obs, info

    def step(self, action):
        self._ensure_runtime_health()
        result = self.env.step(action)
        self._ensure_runtime_health()
        return result

    def close(self) -> None:
        try:
            self._drive_home()
        finally:
            super().close()

    # ------------------------------------------------------------------ health monitoring
    def _ensure_runtime_health(self):
        ctrl = getattr(self.env, "_arm_ctrl", None)
        if ctrl is None:
            return

        now = time.monotonic()
        data = self._get_lowstate_data(ctrl)
        if data is not None:
            self._last_lowstate_stamp = now
            motor_states = getattr(data, "motor_state", None)
            if motor_states is not None:
                arm_dof = getattr(ctrl, "arm_dof", 14)
                for idx in range(min(arm_dof, len(motor_states))):
                    fault = getattr(motor_states[idx], "motorstate", 0)
                    if fault:
                        self._recover(f"motor {idx} fault code {fault}")
                        return
        elif now - self._last_lowstate_stamp > self._health_timeout:
            self._recover("lowstate heartbeat stalled")
            return

        mode_machine = None
        if hasattr(ctrl, "get_mode_machine"):
            try:
                mode_machine = ctrl.get_mode_machine()
            except Exception:
                mode_machine = None
        if self._monitor_control_mode and mode_machine in (None, 0):
            if self._get_motion_switcher() is not None:
                self._recover("control mode lost")

    # ------------------------------------------------------------------ recovery helpers
    def _recover(self, reason: str):
        if self._recovering:
            return
        self._recovering = True
        print(f"[UnitreeSafetyWrapper] Recovery triggered: {reason}")

        try:
            switcher = self._get_motion_switcher()
            if switcher is not None:
                try:
                    switcher.ReleaseMode()
                    time.sleep(0.2)
                except Exception:
                    pass

            try:
                self._drive_home()
            except Exception:
                pass

            if switcher is not None:
                try:
                    switcher.SelectMode("ai")
                    time.sleep(0.2)
                except Exception:
                    pass

            self._apply_soft_start()
            self._wait_for_lowstate(timeout=1.0)
            self._init_mode_monitor()
            print("[UnitreeSafetyWrapper] Recovery complete.")
        finally:
            self._recovering = False

    # ------------------------------------------------------------------ lowstate utilities
    def _snapshot_lowstate(self):
        ctrl = getattr(self.env, "_arm_ctrl", None)
        if ctrl is None:
            return
        data = self._get_lowstate_data(ctrl)
        if data is not None:
            self._last_lowstate_stamp = time.monotonic()

    def _get_lowstate_data(self, ctrl):
        buf = getattr(ctrl, "lowstate_buffer", None)
        return buf.GetData() if buf is not None else None

    def _wait_for_lowstate(self, timeout: float) -> bool:
        ctrl = getattr(self.env, "_arm_ctrl", None)
        if ctrl is None:
            return False
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            data = self._get_lowstate_data(ctrl)
            if data is not None:
                self._last_lowstate_stamp = time.monotonic()
                return True
            time.sleep(0.02)
        return False

    def _get_motion_switcher(self):
        if MotionSwitcherClient is None:
            return None
        if self._motion_switcher is False:
            return None
        if self._motion_switcher is None:
            try:
                client = MotionSwitcherClient()
                client.SetTimeout(5.0)
                client.Init()
                self._motion_switcher = client
            except Exception:
                self._motion_switcher = False
        return self._motion_switcher if self._motion_switcher is not False else None

    def _init_mode_monitor(self):
        ctrl = getattr(self.env, "_arm_ctrl", None)
        self._monitor_control_mode = False
        if ctrl is None:
            return
        switcher = self._get_motion_switcher()
        if switcher is None:
            return
        if hasattr(ctrl, "get_mode_machine"):
            try:
                mode_machine = ctrl.get_mode_machine()
            except Exception:
                mode_machine = None
            if mode_machine not in (None, 0):
                self._monitor_control_mode = True

    # ------------------------------------------------------------------ control helpers
    def _drive_home(self):
        obs = self.env.go_home(steps=self._go_home_steps)
        if self._settle_time > 0.0:
            time.sleep(self._settle_time)
            obs = self.env.observe()
        return obs

    def _apply_soft_start(self):
        if self._soft_start_duration is not None:
            try:
                self.env.speed_gradual_max(duration=self._soft_start_duration)
            except Exception:
                pass
