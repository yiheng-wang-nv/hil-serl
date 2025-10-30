"""
Direct Unitree G1 + Dex3 environment that reuses the official control stack.

This environment wraps ``unitree_lerobot.eval_robot.make_robot.setup_robot_interface``
so that actions and observations follow the same structure used during model
deployment:

    action = [14 joint targets for the arms,
              7 joint targets for the left Dex3 hand,
              7 joint targets for the right Dex3 hand]

Actions are interpreted as target joint positions. The arm targets are converted
to torques via the Unitree IK solver (solve_tau) before being sent to the SDK2
controller, while Dex3 joint targets are written to the shared memory that is
consumed by ``Dex3_1_Controller``.

Observations concatenate the current arm joint positions and velocities with the
Dex3 joint positions (Dex3 velocities are not provided by the shared memory and
are therefore reported as zeros).

Prerequisites:
    - ``unitree_sdk2_python`` installed in the active environment.
    - ``unitree_lerobot`` (already installed in the Python environment).
"""
from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Dict, Optional

import gymnasium as gym
import numpy as np


@dataclass
class UnitreeRobotState:
    arm_position: np.ndarray
    arm_velocity: np.ndarray
    hand_position: np.ndarray


class UnitreeG1DirectEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        arm: str = "G1_29",
        ee: str = "dex3",
        motion_mode: bool = False,
        simulation: bool = False,
        action_dt: float = 0.02,
        arm_position_limit: float = 2.7,
        hand_min_position: float = 0.0,
        hand_max_position: float = 1.0,
    ):
        super().__init__()

        # Expect unitree_lerobot to be importable; rely on environment setup.
        from unitree_lerobot.eval_robot.make_robot import setup_robot_interface

        args = SimpleNamespace(arm=arm, ee=ee, motion=motion_mode, sim=simulation)
        robot_if = setup_robot_interface(args)

        self._arm_ctrl = robot_if["arm_ctrl"]
        self._arm_ik = robot_if["arm_ik"]
        self._ee_shared_mem = robot_if.get("ee_shared_mem")
        self._arm_dof: int = int(robot_if["arm_dof"])
        self._ee_dof: int = int(robot_if.get("ee_dof", 0))
        self._has_dex3: bool = self._ee_dof > 0 and self._ee_shared_mem is not None

        self._action_dt = float(action_dt)

        arm_low = -np.ones(self._arm_dof, dtype=np.float32) * arm_position_limit
        arm_high = np.ones(self._arm_dof, dtype=np.float32) * arm_position_limit

        if self._has_dex3:
            hand_low = np.full(self._ee_dof * 2, hand_min_position, dtype=np.float32)
            hand_high = np.full(self._ee_dof * 2, hand_max_position, dtype=np.float32)
            action_low = np.concatenate([arm_low, hand_low])
            action_high = np.concatenate([arm_high, hand_high])
        else:
            action_low = arm_low
            action_high = arm_high

        self.action_space = gym.spaces.Box(action_low, action_high, dtype=np.float32)

        obs_dim = self._arm_dof + (self._ee_dof * 2 if self._has_dex3 else 0)
        self.observation_space = gym.spaces.Box(
            -np.inf, np.inf, shape=(obs_dim * 2,), dtype=np.float32
        )

        self._initial_state = self._read_robot_state(wait_for_hand=True)
        self._last_action = np.zeros_like(self._initial_state.arm_position, dtype=np.float32)

    # ------------------------------------------------------------------ helpers
    def _read_robot_state(self, wait_for_hand: bool = False) -> UnitreeRobotState:
        arm_pos = np.asarray(self._arm_ctrl.get_current_dual_arm_q(), dtype=np.float32)
        arm_vel = np.asarray(self._arm_ctrl.get_current_dual_arm_dq(), dtype=np.float32)

        if self._has_dex3:
            start = time.time()
            hand_pos = np.zeros(self._ee_dof * 2, dtype=np.float32)
            while True:
                with self._ee_shared_mem["lock"]:
                    shared = np.array(self._ee_shared_mem["state"][:], dtype=np.float32)
                if shared.size >= self._ee_dof * 2 and (not wait_for_hand or np.any(shared)):
                    hand_pos = shared[: self._ee_dof * 2]
                    break
                if not wait_for_hand or (time.time() - start) > 5.0:
                    break
                time.sleep(0.01)
        else:
            hand_pos = np.zeros(0, dtype=np.float32)

        return UnitreeRobotState(arm_position=arm_pos, arm_velocity=arm_vel, hand_position=hand_pos)

    def _apply_action(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        if action.shape[0] != self.action_space.shape[0]:
            raise ValueError(f"Expected action shape {(self.action_space.shape[0],)}, got {action.shape}.")

        arm_target = action[: self._arm_dof]
        tau = self._arm_ik.solve_tau(arm_target)
        self._arm_ctrl.ctrl_dual_arm(arm_target, tau)

        if self._has_dex3:
            left = action[self._arm_dof : self._arm_dof + self._ee_dof]
            right = action[self._arm_dof + self._ee_dof : self._arm_dof + 2 * self._ee_dof]
            with self._ee_shared_mem["lock"]:
                self._ee_shared_mem["left"][:] = left
                self._ee_shared_mem["right"][:] = right
                if "action" in self._ee_shared_mem:
                    self._ee_shared_mem["action"][: self._ee_dof] = left
                    self._ee_shared_mem["action"][self._ee_dof : 2 * self._ee_dof] = right

    def _compose_observation(self, state: UnitreeRobotState) -> np.ndarray:
        hand_velocity = (
            np.zeros_like(state.hand_position) if state.hand_position.size else np.zeros(0, dtype=np.float32)
        )
        obs = np.concatenate([state.arm_position, state.hand_position, state.arm_velocity, hand_velocity])
        return obs.astype(np.float32)

    # ------------------------------------------------------------------ gym API
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        super().reset(seed=seed)
        arm_home = self._initial_state.arm_position
        tau = self._arm_ik.solve_tau(arm_home)
        self._arm_ctrl.ctrl_dual_arm(arm_home, tau)

        if self._has_dex3 and self._initial_state.hand_position.size:
            left = self._initial_state.hand_position[: self._ee_dof]
            right = self._initial_state.hand_position[self._ee_dof :]
            with self._ee_shared_mem["lock"]:
                self._ee_shared_mem["left"][:] = left
                self._ee_shared_mem["right"][:] = right
                if "action" in self._ee_shared_mem:
                    self._ee_shared_mem["action"][: self._ee_dof] = left
                    self._ee_shared_mem["action"][self._ee_dof : 2 * self._ee_dof] = right

        time.sleep(self._action_dt)
        obs = self._compose_observation(self._read_robot_state())
        info: Dict[str, float] = {}
        return obs, info

    def step(self, action: np.ndarray):
        self._apply_action(action)
        time.sleep(self._action_dt)
        state = self._read_robot_state()
        obs = self._compose_observation(state)
        reward = 0.0
        terminated = False
        truncated = False
        info: Dict[str, float] = {}
        return obs, reward, terminated, truncated, info

    def render(self):
        return None

    def close(self):
        try:
            arm_home = self._initial_state.arm_position
            tau = self._arm_ik.solve_tau(arm_home)
            self._arm_ctrl.ctrl_dual_arm(arm_home, tau)
        except Exception:
            pass