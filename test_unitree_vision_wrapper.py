#!/usr/bin/env python3
"""
Quick sanity check for the Unitree vision wrapper on real hardware.

This script instantiates the UnitreeInstallTrocar environment, performs a reset,
and optionally steps the policy with zero actions to pull fresh camera frames.
It prints the observation keys, dtypes, and shapes so you can verify that the
vision streams are populated before launching BC evaluation.
"""

import argparse
import time

import numpy as np

from examples.experiments.unitree_install_trocar.config import TrainConfig


def describe(obs, prefix=""):
    for key, value in obs.items():
        if isinstance(value, dict):
            describe(value, prefix=f"{prefix}{key}.")
            continue
        arr = np.asarray(value)
        if arr.size:
            stats = f"min={arr.min()}, max={arr.max()}"
        else:
            stats = "empty"
        print(f"{prefix}{key}: shape={arr.shape}, dtype={arr.dtype}, {stats}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--steps",
        type=int,
        default=0,
        help="Number of zero-action steps to execute after reset.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Optional delay (seconds) between steps to give image streams time to refresh.",
    )
    parser.add_argument(
        "--video_host",
        type=str,
        default=None,
        help="Override Unitree image server host (defaults to config value).",
    )
    parser.add_argument(
        "--video_port",
        type=int,
        default=None,
        help="Override Unitree image server port (defaults to config value).",
    )
    parser.add_argument(
        "--disable_wrist",
        action="store_true",
        help="Disable wrist camera streams when testing vision.",
    )
    parser.add_argument(
        "--simulation",
        action="store_true",
        help="Connect to the simulator instead of the real robot.",
    )
    args = parser.parse_args()

    config = TrainConfig()
    if args.video_host:
        config.vision_params.server_address = args.video_host
    if args.video_port is not None:
        config.vision_params.port = args.video_port
    if args.disable_wrist:
        config.vision_params.enable_wrist = False
    config.unitree_params.simulation = args.simulation
    env = config.get_environment(fake_env=False, save_video=False, classifier=True)

    try:
        obs, info = env.reset()
        print("Reset info:", info)
        print("Observation summary after reset:")
        describe(obs)

        action = np.zeros(env.action_space.shape, dtype=np.float32)
        for step in range(args.steps):
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"\nStep {step + 1}: reward={reward}, terminated={terminated}, truncated={truncated}")
            describe(obs)
            if args.sleep > 0:
                time.sleep(args.sleep)
            if terminated or truncated:
                print("Environment signaled done; resetting...")
                obs, info = env.reset()
    finally:
        env.close()


if __name__ == "__main__":
    main()
