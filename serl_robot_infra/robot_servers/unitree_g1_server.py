"""
HTTP wrapper for Unitree G1 + Dex3.

This server bridges HTTP requests to the direct Unitree environment that
uses the official SDK2 control stack.  It provides simple endpoints for
setting actions and retrieving the latest observation, similar to the
existing Franka server behaviour.
"""
from __future__ import annotations

import argparse
import threading
import time
from typing import Dict, Optional

import numpy as np
from flask import Flask, jsonify, request

from serl_robot_infra.unitree_env.unitree_robot_env import UnitreeG1DirectEnv


class UnitreeHTTPServer:
    """Expose UnitreeG1DirectEnv via HTTP."""

    def __init__(
        self,
        *,
        arm: str = "G1_29",
        ee: str = "dex3",
        motion_mode: bool = False,
        simulation: bool = False,
        action_dt: float = 0.02,
    ):
        self._env = UnitreeG1DirectEnv(
            arm=arm,
            ee=ee,
            motion_mode=motion_mode,
            simulation=simulation,
            action_dt=action_dt,
        )
        self._action_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._latest_obs, self._latest_info = self._env.reset()
        self._last_action = self._latest_obs[:28].copy()
        self._control_thread = threading.Thread(target=self._control_loop, name="unitree-http-loop", daemon=True)
        self._control_thread.start()

    def close(self):
        self._stop_event.set()
        if self._control_thread.is_alive():
            self._control_thread.join(timeout=1.0)
        self._env.close()

    def get_state(self) -> Dict[str, list]:
        with self._action_lock:
            obs = self._latest_obs.copy()
        arm_pos = obs[:14]
        dex_left = obs[14:21]
        dex_right = obs[21:28]
        arm_vel = obs[28:42]
        return {
            "arm_joint_positions": arm_pos.tolist(),
            "dex3_left_joint_positions": dex_left.tolist(),
            "dex3_right_joint_positions": dex_right.tolist(),
            "arm_joint_velocities": arm_vel.tolist(),
        }

    def _control_loop(self):
        """Continuously apply the most recent action so the DDS controller receives commands at 50 Hz."""
        while not self._stop_event.is_set():
            with self._action_lock:
                action = self._last_action.copy()
            obs, _, _, _, info = self._env.step(action)
            with self._action_lock:
                self._latest_obs = obs
                self._latest_info = info

    def set_action(self, action: np.ndarray) -> Dict[str, float]:
        with self._action_lock:
            self._last_action = np.asarray(action, dtype=np.float32)
            info = dict(self._latest_info)
        return info


def create_app(server: UnitreeHTTPServer) -> Flask:
    app = Flask(__name__)

    @app.route("/set_action", methods=["POST"])
    def set_action():
        payload = request.get_json(force=True, silent=True)
        if not payload or "action" not in payload:
            return jsonify({"error": "missing 'action' array"}), 400
        action = np.asarray(payload["action"], dtype=np.float32)
        if action.shape != (28,):
            return jsonify({"error": "action must be length 28 (14 arm + 14 dex3)"}), 400
        info = server.set_action(action)
        return jsonify({"status": "ok", "info": info})

    @app.route("/get_state", methods=["GET"])
    def get_state():
        return jsonify(server.get_state())

    return app


def main():
    parser = argparse.ArgumentParser(description="Unitree G1 minimal HTTP server")
    parser.add_argument("--port", type=int, default=6000)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--motion", action="store_true", help="use motion controller topic")
    parser.add_argument("--simulation", action="store_true", help="run in simulation mode")
    parser.add_argument("--arm", type=str, default="G1_29")
    parser.add_argument("--ee", type=str, default="dex3")
    parser.add_argument("--dt", type=float, default=0.02)
    args = parser.parse_args()

    server = UnitreeHTTPServer(
        arm=args.arm,
        ee=args.ee,
        motion_mode=args.motion,
        simulation=args.simulation,
        action_dt=args.dt,
    )
    app = create_app(server)
    try:
        app.run(host=args.host, port=args.port)
    finally:
        server.close()


if __name__ == "__main__":
    main()
