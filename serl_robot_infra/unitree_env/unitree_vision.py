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
from multiprocessing import shared_memory
import threading

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from .unitree_safety_env import UnitreeSafetyWrapper


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
        if image_args is None:
            image_args = SimpleNamespace()
        if not hasattr(image_args, "sim"):
            setattr(image_args, "sim", simulation)

        info = self._setup_image_client(image_args)

        self._tv_img_array = info.get("tv_img_array")
        self._wrist_img_array = info.get("wrist_img_array")
        self._tv_img_shape = info.get("tv_img_shape")
        self._wrist_img_shape = info.get("wrist_img_shape")
        self._is_binocular = bool(info.get("is_binocular", False))
        self._has_wrist_cam = bool(info.get("has_wrist_cam", False))
        self._shm_resources = list(info.get("shm_resources", []))
        self._image_client = info.get("client")
        self._image_thread = info.get("thread")

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
        if getattr(self, "_image_client", None) is not None:
            self._image_client.running = False
        if getattr(self, "_image_thread", None) is not None:
            self._image_thread.join(timeout=1.0)
        for shm in self._shm_resources:
            with contextlib.suppress(Exception):
                shm.close()
            with contextlib.suppress(Exception):
                shm.unlink()
        self._shm_resources.clear()
        self._image_client = None
        self._image_thread = None

    @property
    def tv_img_shape(self):
        return self._tv_img_shape

    @property
    def wrist_img_shape(self):
        return self._wrist_img_shape

    @property
    def is_binocular(self) -> bool:
        return self._is_binocular

    @property
    def has_wrist_cam(self) -> bool:
        return self._has_wrist_cam

    def _setup_image_client(self, args: SimpleNamespace) -> Dict[str, Any]:
        from unitree_lerobot.eval_robot.image_server.image_client import ImageClient

        simulation = bool(getattr(args, "sim", False))
        default_sim_config = {
            "fps": 30,
            "head_camera_type": "opencv",
            "head_camera_image_shape": [480, 640],
            "head_camera_id_numbers": [0],
            "wrist_camera_type": "opencv",
            "wrist_camera_image_shape": [480, 640],
            "wrist_camera_id_numbers": [2, 4],
        }
        default_real_config = {
            "fps": 30,
            "head_camera_type": "opencv",
            "head_camera_image_shape": [480, 640],
            "head_camera_id_numbers": [4],
            "wrist_camera_type": "opencv",
            "wrist_camera_image_shape": [480, 640],
            "wrist_camera_id_numbers": [0, 2],
        }
        img_config = getattr(args, "image_config", None)
        if img_config is None:
            img_config = default_sim_config if simulation else default_real_config

        ASPECT_RATIO_THRESHOLD = 2.0
        head_ratio = img_config["head_camera_image_shape"][1] / img_config["head_camera_image_shape"][0]
        binocular = len(img_config.get("head_camera_id_numbers", [])) > 1 or head_ratio > ASPECT_RATIO_THRESHOLD
        has_wrist = "wrist_camera_type" in img_config and img_config.get("wrist_camera_type") is not None

        if binocular and not head_ratio > ASPECT_RATIO_THRESHOLD:
            tv_img_shape = (
                img_config["head_camera_image_shape"][0],
                img_config["head_camera_image_shape"][1] * 2,
                3,
            )
        else:
            tv_img_shape = (
                img_config["head_camera_image_shape"][0],
                img_config["head_camera_image_shape"][1],
                3,
            )

        tv_img_shm = shared_memory.SharedMemory(
            create=True, size=int(np.prod(tv_img_shape)) * np.uint8().itemsize
        )
        tv_img_array = np.ndarray(tv_img_shape, dtype=np.uint8, buffer=tv_img_shm.buf)

        wrist_img_array = None
        wrist_img_shape = None
        wrist_img_shm = None
        server_address = getattr(args, "server_address", "192.168.123.164")
        port = getattr(args, "port", 5555)
        image_show = getattr(args, "image_show", False)

        if has_wrist:
            wrist_img_shape = (
                img_config["wrist_camera_image_shape"][0],
                img_config["wrist_camera_image_shape"][1] * 2,
                3,
            )
            wrist_img_shm = shared_memory.SharedMemory(
                create=True, size=int(np.prod(wrist_img_shape)) * np.uint8().itemsize
            )
            wrist_img_array = np.ndarray(wrist_img_shape, dtype=np.uint8, buffer=wrist_img_shm.buf)
            image_client = ImageClient(
                tv_img_shape=tv_img_shape,
                tv_img_shm_name=tv_img_shm.name,
                wrist_img_shape=wrist_img_shape,
                wrist_img_shm_name=wrist_img_shm.name,
                server_address=server_address,
                port=port,
                image_show=image_show,
            )
        else:
            image_client = ImageClient(
                tv_img_shape=tv_img_shape,
                tv_img_shm_name=tv_img_shm.name,
                server_address=server_address,
                port=port,
                image_show=image_show,
            )

        thread = threading.Thread(target=image_client.receive_process, daemon=True)
        thread.start()

        shm_resources = [tv_img_shm]
        if wrist_img_shm is not None:
            shm_resources.append(wrist_img_shm)

        return {
            "tv_img_array": tv_img_array,
            "wrist_img_array": wrist_img_array,
            "tv_img_shape": tv_img_shape,
            "wrist_img_shape": wrist_img_shape,
            "is_binocular": binocular,
            "has_wrist_cam": has_wrist,
            "shm_resources": shm_resources,
            "client": image_client,
            "thread": thread,
        }


class UnitreeVisionWrapper(gym.Wrapper):
    """Augment :class:`UnitreeG1DirectEnv` observations with camera frames."""

    def __init__(
        self,
        env: gym.Env,
        *,
        simulation: Optional[bool] = None,
        image_args: Optional[SimpleNamespace] = None,
        copy_images: bool = True,
        enable_safety: bool = True,
        safety_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        base_env = env
        wrapped_env = env
        if enable_safety and not isinstance(env, UnitreeSafetyWrapper):
            wrapped_env = UnitreeSafetyWrapper(env, **(safety_kwargs or {}))

        super().__init__(wrapped_env)

        raw_env = wrapped_env.unwrapped
        if simulation is None:
            simulation = bool(getattr(raw_env, "simulation_mode", False))
        self._image_client = UnitreeImageClient(simulation=simulation, image_args=image_args)
        self._copy_images = copy_images
        self._base_env = base_env

        # Build observation space dict
        robot_state_space = getattr(wrapped_env, "observation_space", None)
        if robot_state_space is None:
            robot_state_space = spaces.Box(-np.inf, np.inf, shape=raw_env.observation_space.shape, dtype=np.float32)

        self._arm_dof = int(getattr(raw_env, "_arm_dof", 0))
        ee_dof = int(getattr(raw_env, "_ee_dof", 0))
        hand_pos_len = ee_dof * 2 if ee_dof > 0 else 0
        self._state_dim = self._arm_dof + hand_pos_len
        if self._state_dim <= 0:
            self._state_dim = robot_state_space.shape[0] if hasattr(robot_state_space, "shape") else 0
        state_shape = (self._state_dim,) if self._state_dim > 0 else robot_state_space.shape

        obs_spaces: Dict[str, spaces.Space] = {
            "robot_state": robot_state_space,
            # Alias required by learning code paths that expect a "state" key.
            "state": spaces.Box(-np.inf, np.inf, shape=state_shape, dtype=np.float32),
            "video.room_view": spaces.Box(
                low=0,
                high=255,
                shape=self._image_client.tv_img_shape,
                dtype=np.uint8,
            ),
        }

        if self._image_client.is_binocular and self._image_client.tv_img_shape is not None:
            room_shape = self._image_client.tv_img_shape
            half_width = room_shape[1] // 2
            split_shape = (room_shape[0], half_width, room_shape[2])
            obs_spaces["video.room_view_left"] = spaces.Box(0, 255, shape=split_shape, dtype=np.uint8)
            obs_spaces["video.room_view_right"] = spaces.Box(0, 255, shape=split_shape, dtype=np.uint8)

        if self._image_client.has_wrist_cam and self._image_client.wrist_img_shape is not None:
            wrist_shape = self._image_client.wrist_img_shape
            half_width = wrist_shape[1] // 2
            split_shape = (wrist_shape[0], half_width, wrist_shape[2])
            obs_spaces["video.left_wrist_view"] = spaces.Box(0, 255, shape=split_shape, dtype=np.uint8)
            obs_spaces["video.right_wrist_view"] = spaces.Box(0, 255, shape=split_shape, dtype=np.uint8)

        self.observation_space = spaces.Dict(obs_spaces)

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
        state_vec = np.asarray(robot_obs)
        if self._state_dim > 0 and state_vec.shape[0] >= self._state_dim:
            obs["state"] = state_vec[: self._state_dim]
        else:
            obs["state"] = state_vec
        obs.update(self._image_client.get_latest_frames(copy=self._copy_images))
        return obs
