"""
Minimal HTTP bridge between Unitree G1 DDS topics and the SERL robot infra API.

The server exposes a lightweight REST interface that mimics the subset of
Franka's endpoints required for quick smoke tests:
    - /joint_position : position command for the dual arm (14 joints)
    - /open_gripper   : open command for the Dex3 hand (whole-hand open)
    - /close_gripper  : close command for the Dex3 hand (whole-hand close)
    - /getstate       : returns latest joint position/velocity snapshot

The implementation is intentionally conservative:
    * only position control is supported
    * torque commands and impedance parameters are omitted
    * safety checks (limits, collision handling) must be added before real use

Usage example:

    python -m serl_robot_infra.robot_servers.unitree_g1_server \\
        --interface enp3s0 --port 6000

The server depends on unitree_sdk2_python (DDS) and runs in the same style as
serl_robot_infra/robot_servers/franka_server.py.
"""
from __future__ import annotations

import os
import argparse
import threading
import time
import logging
from dataclasses import dataclass, field
from copy import deepcopy
from typing import Dict, List, Optional

from flask import Flask, jsonify, request

try:
    from unitree_sdk2py.core.channel import (
        ChannelFactoryInitialize,
        ChannelPublisher,
        ChannelSubscriber,
    )
    from unitree_sdk2py.core.channel_config import ChannelConfigAutoDetermine
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import (
        LowCmd_ as HgLowCmd,
        LowState_ as HgLowState,
    )
    from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
    from unitree_sdk2py.utils.crc import CRC
except ImportError as exc:
    raise ImportError(
        "unitree_sdk2_python must be installed to run the Unitree G1 server."
    ) from exc


# DDS topics used by Unitree SDK2 for low level control
LOWCMD_TOPIC = "rt/arm_sdk"
LOWSTATE_TOPIC = "rt/lowstate"


logger = logging.getLogger(__name__)


@dataclass
class JointStateSnapshot:
    """Simple structure for the latest DDS low state."""

    position: List[float] = field(default_factory=list)
    velocity: List[float] = field(default_factory=list)
    timestamp: float = 0.0
    mode_machine: int = 0


class G1ArmBridge:
    """DDS helper that mirrors essential portions of xr_teleoperate's arm controller.

    The bridge maintains a background subscriber thread that collects the latest
    low state message, while a publisher thread streaming the last commanded
    position message at a fixed rate.
    """

    ARM_JOINT_INDEXES: List[int] = [
        15, 16, 17, 18, 19, 20, 21,  # left arm
        22, 23, 24, 25, 26, 27, 28,  # right arm
    ]

    def __init__(
        self,
        *,
        use_motion_topic: bool = False,
        simulation_mode: bool = False,
        publish_rate_hz: float = 250.0,
    ):
        self._publish_period = 1.0 / publish_rate_hz
        self._msg_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._terminated = threading.Event()

        # Initialise DDS factory (0: robot, 1: simulation, same convention as SDK2)
        ChannelFactoryInitialize(1 if simulation_mode else 0)
        # Autodetermine network interface unless the environment already specifies one.
        if "CYCLONEDDS_URI" not in os.environ:
            os.environ["CYCLONEDDS_URI"] = ChannelConfigAutoDetermine

        lowcmd_topic = LOWCMD_TOPIC if use_motion_topic else "rt/lowcmd"
        self._publisher = ChannelPublisher(lowcmd_topic, HgLowCmd)
        self._publisher.Init()

        self._subscriber = ChannelSubscriber(LOWSTATE_TOPIC, HgLowState)
        self._subscriber.Init()

        self._crc = CRC()
        self._command_msg = unitree_hg_msg_dds__LowCmd_()
        self._command_msg.mode_pr = 0
        self._command_msg.mode_machine = 0  # 0: position control mode

        # Configure control gains (low stiffness for minimal testing)
        for idx in range(len(self._command_msg.motor_cmd)):
            cmd = self._command_msg.motor_cmd[idx]
            cmd.mode = 1
            cmd.kp = 40.0
            cmd.kd = 1.0

        self._latest_state: JointStateSnapshot = JointStateSnapshot(
            position=[0.0] * len(self._command_msg.motor_cmd),
            velocity=[0.0] * len(self._command_msg.motor_cmd),
            timestamp=time.time(),
        )

        self._update_crc()

        # Start threads
        self._initial_state_ready = threading.Event()
        self._state_thread = threading.Thread(
            target=self._state_loop, name="g1_state_loop", daemon=True
        )
        self._state_thread.start()

        self._publish_thread = threading.Thread(
            target=self._publish_loop, name="g1_publish_loop", daemon=True
        )
        self._publish_thread.start()

        if not self._initial_state_ready.wait(timeout=5.0):
            logger.warning("Timeout waiting for initial DDS state; mode_machine left at default.")
        else:
            self._command_msg.mode_machine = self._latest_state.mode_machine
            self._update_crc()

    def shutdown(self):
        self._terminated.set()
        self._state_thread.join(timeout=1.0)
        self._publish_thread.join(timeout=1.0)

    # --------------------------------------------------------------------- state
    def get_snapshot(self) -> JointStateSnapshot:
        with self._state_lock:
            return JointStateSnapshot(
                position=list(self._latest_state.position),
                velocity=list(self._latest_state.velocity),
                timestamp=self._latest_state.timestamp,
            )

    def _state_loop(self):
        while not self._terminated.is_set():
            msg = self._subscriber.Read(timeout=0.02)
            if msg is None:
                continue

            with self._state_lock:
                for idx, motor in enumerate(msg.motor_state):
                    self._latest_state.position[idx] = motor.q
                    self._latest_state.velocity[idx] = motor.dq
                self._latest_state.timestamp = time.time()
                self._latest_state.mode_machine = msg.mode_machine
            self._initial_state_ready.set()

    # ------------------------------------------------------------------- command
    def set_arm_joint_targets(self, target: List[float]):
        if len(target) != len(self.ARM_JOINT_INDEXES):
            raise ValueError(
                f"Expected {len(self.ARM_JOINT_INDEXES)} joint values, got {len(target)}"
            )

        with self._msg_lock:
            for joint_idx, value in zip(self.ARM_JOINT_INDEXES, target):
                cmd = self._command_msg.motor_cmd[joint_idx]
                cmd.q = float(value)
                cmd.dq = 0.0
                cmd.tau = 0.0
            self._update_crc()

    def set_hand_open_ratio(self, ratio: float):
        """Broadcast a simple open/close command by mirroring it on wrist yaw motors.

        Dex3 has many actuators; for this minimal bridge we reuse wrist yaw joints as
        placeholders so that downstream layers can test the HTTP path. Real projects
        must replace this with the proper Dex3 interface or separate publisher.
        """
        clamped = max(0.0, min(1.0, ratio))
        wrist_targets = [
            self.ARM_JOINT_INDEXES.index(idx)
            for idx in (21, 28)  # left/right wrist yaw
            if idx in self.ARM_JOINT_INDEXES
        ]

        with self._msg_lock:
            for local_idx in wrist_targets:
                motor_index = self.ARM_JOINT_INDEXES[local_idx]
                self._command_msg.motor_cmd[motor_index].q = clamped
            self._update_crc()

    def _publish_loop(self):
        while not self._terminated.is_set():
            start = time.time()
            with self._msg_lock:
                msg = deepcopy(self._command_msg)
            self._publisher.Write(msg)
            elapsed = time.time() - start
            sleep_time = self._publish_period - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _update_crc(self):
        self._command_msg.crc = self._crc.Crc(self._command_msg)


# ------------------------------------------------------------------------------
# Flask server
# ------------------------------------------------------------------------------


def create_app(bridge: G1ArmBridge) -> Flask:
    app = Flask(__name__)

    @app.route("/joint_position", methods=["POST"])
    def joint_position():
        payload = request.get_json(force=True, silent=True)
        if not payload or "joint_positions" not in payload:
            return jsonify({"error": "joint_positions missing"}), 400
        try:
            bridge.set_arm_joint_targets(payload["joint_positions"])
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        return jsonify({"status": "ok"})

    @app.route("/open_gripper", methods=["POST"])
    def open_gripper():
        bridge.set_hand_open_ratio(1.0)
        return jsonify({"status": "ok"})

    @app.route("/close_gripper", methods=["POST"])
    def close_gripper():
        bridge.set_hand_open_ratio(0.0)
        return jsonify({"status": "ok"})

    @app.route("/getstate", methods=["POST"])
    def get_state():
        snapshot = bridge.get_snapshot()
        response: Dict[str, Optional[List[float]]] = {
            "joint_positions": snapshot.position,
            "joint_velocities": snapshot.velocity,
            "timestamp": snapshot.timestamp,
        }
        return jsonify(response)

    return app


def main():
    parser = argparse.ArgumentParser(description="Unitree G1 minimal HTTP server")
    parser.add_argument("--port", type=int, default=6000, help="HTTP port to bind")
    parser.add_argument(
        "--host", type=str, default="0.0.0.0", help="Address to bind the Flask server"
    )
    parser.add_argument(
        "--motion-topic",
        action="store_true",
        help="Use the motion controller topic (rt/arm_sdk) instead of debug topic",
    )
    parser.add_argument(
        "--simulation",
        action="store_true",
        help="Initialise SDK2 in simulation mode (ChannelFactoryInitialize(1))",
    )
    args = parser.parse_args()

    bridge = G1ArmBridge(
        use_motion_topic=args.motion_topic, simulation_mode=args.simulation
    )
    app = create_app(bridge)
    try:
        app.run(host=args.host, port=args.port)
    finally:
        bridge.shutdown()


if __name__ == "__main__":
    main()
