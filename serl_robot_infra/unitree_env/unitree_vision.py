"""
Camera utilities for Unitree G1 + Dex3 environments.

This module provides a thin wrapper around the official Unitree image client so
that camera frames can be retrieved alongside joint observations during RL
rollouts.  It mirrors the workflow used by ``unitree_lerobot`` and reuses the
same shared-memory buffers to minimise copies.
"""
from __future__ import annotations

import contextlib
from types import SimpleNamespace
from typing import Any, Dict, Optional

import gymnasium as gym
import numpy as np


class UnitreeImageClient:
    """Manage Unitree's shared-memory image client.

    Parameters
    ----------
    simulation:
        When ``True`` the client connects to the simulator image stream
        (matching the ``--sim`` flag in Unitree's tooling).  Otherwise it
        expects the real-robot image server.
    image_args:
        Optional namespace passed directly to
        :func:`unitree_lerobot.eval_robot.make_robot.setup_image_client`.  When
        omitted a minimal namespace with the ``sim`` attribute is created.
    """

    def __init__(
        self,
        *,
        simulation: bool,
        image_args: Optional[SimpleNamespace] = None,
    ) -> None:
        from unitree_lerobot.eval_robot.make_robot import setup_image_client

        if image_args is None:
            image_args = SimpleNamespace(sim=simulation)
        elif not hasattr(image_args, "sim"):
            setattr(image_args, "sim", simulation)

        info = setup_image_client(image_args)

        self._tv_img_array = info.get("tv_img_array")
        self._wrist_img_array = info.get("wrist_img_array")
        self._tv_img_shape = info.get("tv_img_shape")
        self._wrist_img_shape = info.get("wrist_img_shape")
        self._is_binocular = bool(info.get("is_binocular", False))
        self._has_wrist_cam = bool(info.get("has_wrist_cam", False))
        self._shm_resources = list(info.get("shm_resources", []))

        if self._tv_img_array is None:
            raise RuntimeError("setup_image_client did not return a head camera array.")

    def get_latest_frames(self, *, copy: bool = True) -> Dict[str, np.ndarray]:
        """Return the most recent camera frames.

        Parameters
        ----------
        copy:
            Whether to return copies of the shared-memory buffers.  Leave as
            ``True`` when the data might outlive the next DDS update.
        """
        frames: Dict[str, np.ndarray] = {}

        head_image = np.array(self._tv_img_array, copy=copy)
        frames["video.room_view"] = head_image

        if self._is_binocular and self._tv_img_shape is not None:
            width = self._tv_img_shape[1]
            mid = width // 2
            frames["video.room_view_left"] = np.array(head_image[:, :mid], copy=False)
            frames["video.room_view_right"] = np.array(head_image[:, mid:], copy=False)

        if self._has_wrist_cam and self._wrist_img_array is not None and self._wrist_img_shape is not None:
            wrist_image = np.array(self._wrist_img_array, copy=copy)
            width = self._wrist_img_shape[1]
            mid = width // 2
            frames["video.left_wrist_view"] = wrist_image[:, :mid]
            frames["video.right_wrist_view"] = wrist_image[:, mid:]
        return frames

    def close(self) -> None:
        """Release shared-memory resources."""
        for shm in self._shm_resources:
            with contextlib.suppress(Exception):
                shm.close()
            with contextlib.suppress(Exception):
                shm.unlink()
        self._shm_resources.clear()


class UnitreeVisionWrapper(gym.Wrapper):
    """Augment :class:`UnitreeG1DirectEnv` observations with camera frames."""

    def __init__(
        self,
        env: gym.Env,
        *,
        simulation: Optional[bool] = None,
        image_args: Optional[SimpleNamespace] = None,
        copy_images: bool = True,
    ) -> None:
        super().__init__(env)
        if simulation is None:
            simulation = bool(getattr(env, "simulation_mode", False))
        self._image_client = UnitreeImageClient(simulation=simulation, image_args=image_args)
        self._copy_images = copy_images

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        robot_obs, info = self.env.reset(seed=seed, options=options)
        return self._compose(robot_obs), info

    def step(self, action):
        robot_obs, reward, terminated, truncated, info = self.env.step(action)
        obs = self._compose(robot_obs)
        return obs, reward, terminated, truncated, info

    def observe(self):
        robot_obs = self.env.observe()
        return self._compose(robot_obs)

    def close(self) -> None:
        try:
            self._image_client.close()
        finally:
            super().close()

    # ------------------------------------------------------------------ helpers
    def _compose(self, robot_obs: Any) -> Dict[str, Any]:
        obs: Dict[str, Any] = {"robot_state": robot_obs}
        obs.update(self._image_client.get_latest_frames(copy=self._copy_images))
        return obs

