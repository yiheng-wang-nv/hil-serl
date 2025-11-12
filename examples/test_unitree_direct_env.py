#!/usr/bin/env python3
"""
Low-level sanity check for UnitreeG1DirectEnv.

This script exercises the bare UnitreeG1DirectEnv without safety/vision wrappers.
It resets the robot, prints the initial observation/action bounds, sends a few
zero actions (optionally), and reports any controller/runtime faults.
"""

import argparse
import time

import numpy as np

from serl_robot_infra.unitree_env.unitree_robot_env import UnitreeG1DirectEnv


def describe_bounds(env: UnitreeG1DirectEnv):
    print("Action space low (first 10):", env.action_space.low[:10])
    print("Action space high (first 10):", env.action_space.high[:10])
    print("Observation space shape:", env.observation_space.shape)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=0, help="Zero-action steps after reset.")
    parser.add_argument("--sleep", type=float, default=0.0, help="Delay between steps (seconds).")
    parser.add_argument("--simulation", action="store_true", help="Connect to simulator instead of real robot.")
    parser.add_argument("--motion_mode", action="store_true", help="Enable Unitree motion blending.")
    parser.add_argument("--action_dt", type=float, default=0.02, help="Control period (seconds).")
    args = parser.parse_args()

    env = UnitreeG1DirectEnv(
        simulation=args.simulation,
        motion_mode=args.motion_mode,
        action_dt=args.action_dt,
    )

    try:
        describe_bounds(env)
        obs, info = env.reset()
        print("Initial observation (first 20 entries):", np.asarray(obs)[:20])
        print("Reset info:", info)

        action = env._initial_action if hasattr(env, "_initial_action") else np.zeros(env.action_space.shape)
        action = np.asarray(action, dtype=np.float32)

        for step in range(args.steps):
            obs, reward, terminated, truncated, step_info = env.step(action)
            print(
                f"Step {step + 1}: reward={reward}, terminated={terminated}, "
                f"truncated={truncated}, obs[:5]={np.asarray(obs)[:5]}"
            )
            if args.sleep > 0:
                time.sleep(args.sleep)
            if terminated or truncated:
                print("Environment finished episode; resetting...")
                obs, info = env.reset()
    finally:
        env.close()


if __name__ == "__main__":
    main()
