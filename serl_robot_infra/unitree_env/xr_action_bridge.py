"""Shared-memory bridge to exchange teleop actions with Unitree env wrappers."""

from __future__ import annotations

import time
from multiprocessing import shared_memory
from typing import Optional, Tuple

import numpy as np

ACTION_DIM = 28
ACTION_SHM_NAME = "unitree_xr_action"
META_SHM_NAME = "unitree_xr_meta"


def _attach_shared_memory(name: str, size: int) -> Tuple[shared_memory.SharedMemory, bool]:
    try:
        shm = shared_memory.SharedMemory(name=name)
        created = False
    except FileNotFoundError:
        shm = shared_memory.SharedMemory(name=name, create=True, size=size)
        created = True
    return shm, created


class XRActionBuffer:
    def __init__(self) -> None:
        action_bytes = ACTION_DIM * np.float64().nbytes
        meta_bytes = 2 * np.float64().nbytes  # [sequence, timestamp]

        self.action_shm, created_action = _attach_shared_memory(ACTION_SHM_NAME, action_bytes)
        self.meta_shm, created_meta = _attach_shared_memory(META_SHM_NAME, meta_bytes)

        self.action = np.ndarray((ACTION_DIM,), dtype=np.float64, buffer=self.action_shm.buf)
        self.meta = np.ndarray((2,), dtype=np.float64, buffer=self.meta_shm.buf)

        if created_action:
            self.action.fill(0.0)
        if created_meta:
            self.meta.fill(0.0)

    def update(self, action: np.ndarray) -> None:
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (ACTION_DIM,):
            raise ValueError(f"Expected action of shape ({ACTION_DIM},), got {action.shape}")
        self.action[:] = action
        self.meta[0] += 1.0
        self.meta[1] = time.time()

    def read(self, last_seq: float) -> Tuple[np.ndarray, bool, float]:
        seq = self.meta[0]
        if seq == last_seq:
            return np.asarray(self.action, dtype=np.float32), False, last_seq
        return np.asarray(self.action, dtype=np.float32), True, seq

    def close(self) -> None:
        self.action_shm.close()
        self.meta_shm.close()


_XR_BUFFER: Optional[XRActionBuffer] = None


def publish_xr_action(action: np.ndarray) -> None:
    global _XR_BUFFER
    if _XR_BUFFER is None:
        _XR_BUFFER = XRActionBuffer()
    _XR_BUFFER.update(action)


def fetch_latest_action(last_sequence: float) -> Tuple[np.ndarray, bool, float]:
    global _XR_BUFFER
    if _XR_BUFFER is None:
        _XR_BUFFER = XRActionBuffer()
    return _XR_BUFFER.read(last_sequence)


def close_bridge() -> None:
    global _XR_BUFFER
    if _XR_BUFFER is not None:
        _XR_BUFFER.close()
        _XR_BUFFER = None
