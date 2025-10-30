"""
Minimal HTTP bridge between Unitree G1 DDS topics and the SERL robot infra API.

The server exposes a lightweight REST interface that mimics the subset of
Franka's endpoints required for quick smoke tests:
    - /joint_position : position command for 14 arm joints + 14 Dex3 finger joints
    - /open_gripper   : helper command to fully open both Dex3 hands
    - /close_gripper  : helper command to fully close both Dex3 hands
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
from enum import IntEnum
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
        HandCmd_ as HgHandCmd,
        HandState_ as HgHandState,
    )
    from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
    from unitree_sdk2py.idl.default import unitree_hg_msg_dds__HandCmd_
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


@dataclass
class Dex3StateSnapshot:
    left_position: List[float] = field(default_factory=list)
    right_position: List[float] = field(default_factory=list)
    left_velocity: List[float] = field(default_factory=list)
    right_velocity: List[float] = field(default_factory=list)
    timestamp: float = 0.0


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

        # Autodetermine network interface unless the environment already specifies one.
        if "CYCLONEDDS_URI" not in os.environ:
            os.environ["CYCLONEDDS_URI"] = ChannelConfigAutoDetermine
        # Initialise DDS factory (0: robot, 1: simulation, same convention as SDK2)
        ChannelFactoryInitialize(1 if simulation_mode else 0)

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
                mode_machine=self._latest_state.mode_machine,
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


class Dex3JointIndexLeft(IntEnum):
    Thumb0 = 0
    Thumb1 = 1
    Thumb2 = 2
    Middle0 = 3
    Middle1 = 4
    Index0 = 5
    Index1 = 6


class Dex3JointIndexRight(IntEnum):
    Thumb0 = 0
    Thumb1 = 1
    Thumb2 = 2
    Index0 = 3
    Index1 = 4
    Middle0 = 5
    Middle1 = 6


class Dex3Bridge:
    LEFT_CMD_TOPIC = "rt/dex3/left/cmd"
    RIGHT_CMD_TOPIC = "rt/dex3/right/cmd"
    LEFT_STATE_TOPIC = "rt/dex3/left/state"
    RIGHT_STATE_TOPIC = "rt/dex3/right/state"

    def __init__(self, publish_rate_hz: float = 100.0):
        self._publish_period = 1.0 / publish_rate_hz
        self._cmd_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._terminated = threading.Event()
        self._initial_state_ready = threading.Event()

        self._left_publisher = ChannelPublisher(self.LEFT_CMD_TOPIC, HgHandCmd)
        self._left_publisher.Init()
        self._right_publisher = ChannelPublisher(self.RIGHT_CMD_TOPIC, HgHandCmd)
        self._right_publisher.Init()

        self._left_subscriber = ChannelSubscriber(self.LEFT_STATE_TOPIC, HgHandState)
        self._left_subscriber.Init()
        self._right_subscriber = ChannelSubscriber(self.RIGHT_STATE_TOPIC, HgHandState)
        self._right_subscriber.Init()

        self._left_command = unitree_hg_msg_dds__HandCmd_()
        self._right_command = unitree_hg_msg_dds__HandCmd_()
        self._initialise_hand_command(self._left_command, Dex3JointIndexLeft)
        self._initialise_hand_command(self._right_command, Dex3JointIndexRight)

        self._latest_state = Dex3StateSnapshot(
            left_position=[0.0] * len(Dex3JointIndexLeft),
            right_position=[0.0] * len(Dex3JointIndexRight),
            left_velocity=[0.0] * len(Dex3JointIndexLeft),
            right_velocity=[0.0] * len(Dex3JointIndexRight),
            timestamp=time.time(),
        )

        self._state_thread = threading.Thread(
            target=self._state_loop, name="dex3_state_loop", daemon=True
        )
        self._state_thread.start()

        self._publish_thread = threading.Thread(
            target=self._publish_loop, name="dex3_publish_loop", daemon=True
        )
        self._publish_thread.start()

        if not self._initial_state_ready.wait(timeout=5.0):
            logger.warning(
                "Dex3Bridge: waiting for initial dexterous hand state timed out."
            )

    @staticmethod
    def _encode_mode(motor_id: int, status: int = 0x01, timeout: int = 0) -> int:
        mode = 0
        mode |= (motor_id & 0x0F)
        mode |= (status & 0x07) << 4
        mode |= (timeout & 0x01) << 7
        return mode

    def _initialise_hand_command(self, message: HgHandCmd, joint_enum: IntEnum):
        for idx in joint_enum:
            cmd = message.motor_cmd[idx]
            cmd.mode = self._encode_mode(idx)
            cmd.q = 0.0
            cmd.dq = 0.0
            cmd.tau = 0.0
            cmd.kp = 1.5
            cmd.kd = 0.2

    def shutdown(self):
        self._terminated.set()
        self._state_thread.join(timeout=1.0)
        self._publish_thread.join(timeout=1.0)

    def set_joint_targets(self, left: List[float], right: List[float]):
        if len(left) != len(Dex3JointIndexLeft) or len(right) != len(Dex3JointIndexRight):
            raise ValueError("Dex3Bridge expects 7 joint values per hand.")
        with self._cmd_lock:
            for idx, value in enumerate(left):
                self._left_command.motor_cmd[idx].q = float(value)
            for idx, value in enumerate(right):
                self._right_command.motor_cmd[idx].q = float(value)

    def get_snapshot(self) -> Dex3StateSnapshot:
        with self._state_lock:
            return Dex3StateSnapshot(
                left_position=list(self._latest_state.left_position),
                right_position=list(self._latest_state.right_position),
                left_velocity=list(self._latest_state.left_velocity),
                right_velocity=list(self._latest_state.right_velocity),
                timestamp=self._latest_state.timestamp,
            )

    def _state_loop(self):
        while not self._terminated.is_set():
            left_msg = self._left_subscriber.Read(timeout=0.02)
            right_msg = self._right_subscriber.Read(timeout=0.02)
            if left_msg is None or right_msg is None:
                continue
            with self._state_lock:
                for idx in Dex3JointIndexLeft:
                    self._latest_state.left_position[idx] = left_msg.motor_state[idx].q
                    self._latest_state.left_velocity[idx] = left_msg.motor_state[idx].dq
                for idx in Dex3JointIndexRight:
                    self._latest_state.right_position[idx] = right_msg.motor_state[idx].q
                    self._latest_state.right_velocity[idx] = right_msg.motor_state[idx].dq
                self._latest_state.timestamp = time.time()
            self._initial_state_ready.set()

    def _publish_loop(self):
        while not self._terminated.is_set():
            start = time.time()
            with self._cmd_lock:
                left_cmd = deepcopy(self._left_command)
                right_cmd = deepcopy(self._right_command)
            self._left_publisher.Write(left_cmd)
            self._right_publisher.Write(right_cmd)
            elapsed = time.time() - start
            sleep_time = self._publish_period - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

# ------------------------------------------------------------------------------
# Flask server
# ------------------------------------------------------------------------------


def create_app(arm_bridge: G1ArmBridge, dex_bridge: Dex3Bridge) -> Flask:
    app = Flask(__name__)

    @app.route("/joint_position", methods=["POST"])
    def joint_position():
        payload = request.get_json(force=True, silent=True)
        if not payload or "joint_positions" not in payload:
            return jsonify({"error": "joint_positions missing"}), 400
        try:
            joint_positions = payload["joint_positions"]
            if len(joint_positions) != 28:
                return jsonify({"error": "Expected 28 joint values (14 arm + 14 dex3)"}), 400
            arm_targets = joint_positions[:14]
            dex_left = joint_positions[14:21]
            dex_right = joint_positions[21:]
            arm_bridge.set_arm_joint_targets(arm_targets)
            dex_bridge.set_joint_targets(dex_left, dex_right)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        return jsonify({"status": "ok"})

    @app.route("/open_gripper", methods=["POST"])
    def open_gripper():
        dex_bridge.set_joint_targets([1.0] * 7, [1.0] * 7)
        return jsonify({"status": "ok"})

    @app.route("/close_gripper", methods=["POST"])
    def close_gripper():
        dex_bridge.set_joint_targets([0.0] * 7, [0.0] * 7)
        return jsonify({"status": "ok"})

    @app.route("/getstate", methods=["POST"])
    def get_state():
        arm_snapshot = arm_bridge.get_snapshot()
        dex_snapshot = dex_bridge.get_snapshot()
        response: Dict[str, Optional[List[float]]] = {
            "arm_joint_positions": arm_snapshot.position[: len(G1ArmBridge.ARM_JOINT_INDEXES)],
            "arm_joint_velocities": arm_snapshot.velocity[: len(G1ArmBridge.ARM_JOINT_INDEXES)],
            "dex3_left_joint_positions": dex_snapshot.left_position,
            "dex3_right_joint_positions": dex_snapshot.right_position,
            "dex3_left_joint_velocities": dex_snapshot.left_velocity,
            "dex3_right_joint_velocities": dex_snapshot.right_velocity,
            "mode_machine": arm_snapshot.mode_machine,
            "timestamp": max(arm_snapshot.timestamp, dex_snapshot.timestamp),
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

    arm_bridge = G1ArmBridge(
        use_motion_topic=args.motion_topic, simulation_mode=args.simulation
    )
    dex_bridge = Dex3Bridge()
    app = create_app(arm_bridge, dex_bridge)
    try:
        app.run(host=args.host, port=args.port)
    finally:
        arm_bridge.shutdown()
        dex_bridge.shutdown()


if __name__ == "__main__":
    main()
