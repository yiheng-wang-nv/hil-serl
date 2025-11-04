#!/usr/bin/env python3
"""
Offline replay utility for Unitree G1 + Dex3.

Given a LeRobot dataset episode, the tool replays the stored joint commands
through ``UnitreeG1DirectEnv`` so that the robot (or simulator) reproduces the
recorded motion.  The implementation mirrors Unitree's official replay script
but reuses the infrastructure wrapped inside HIL-SERL.
"""
from __future__ import annotations

import argparse
import time
from typing import Iterable, Tuple

import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset

from serl_robot_infra.unitree_env import (
    UnitreeG1DirectEnv,
    UnitreeSafetyWrapper,
    UnitreeVisionWrapper,
)


def _load_episode(
    repo_id: str,
    episode: int,
    root: str | None,
) -> Tuple[Iterable[np.ndarray], dict]:
    """Load a single episode from a LeRobot dataset."""
    dataset = LeRobotDataset(repo_id=repo_id, root=root, episodes=[episode])
    actions = dataset.hf_dataset.select_columns("action")
    start_idx = dataset.episode_data_index["from"][0].item()
    first_step = dataset[start_idx]
    return (np.asarray(step["action"], dtype=np.float32) for step in actions), first_step


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Replay Unitree G1 + Dex3 trajectories.")
    parser.add_argument("--repo-id", required=True, help="LeRobot dataset repo identifier.")
    parser.add_argument("--episode", type=int, default=0, help="Episode index to replay.")
    parser.add_argument("--root", type=str, default=None, help="Optional local dataset cache root.")
    parser.add_argument("--simulation", action="store_true", help="Use simulator DDS channel (ChannelFactoryInitialize(1)).")
    parser.add_argument("--frequency", type=float, default=50.0, help="Replay frequency in Hz (default: 50).")
    parser.add_argument("--vision", action="store_true", help="Attach UnitreeVisionWrapper to stream RGB frames.")
    parser.add_argument("--no-safety", action="store_true", help="Disable UnitreeSafetyWrapper (use with caution).")
    parser.add_argument("--settle", type=float, default=1.0, help="Seconds to wait after the initial pose is commanded.")
    parser.add_argument("--log-every", type=int, default=100, help="Print progress every N steps.")
    args = parser.parse_args(argv)

    action_dt = 1.0 / max(1e-6, args.frequency)
    control_env = UnitreeG1DirectEnv(
        arm="G1_29",
        ee="dex3",
        simulation=args.simulation,
        action_dt=action_dt,
    )
    env = control_env
    if args.vision:
        env = UnitreeVisionWrapper(env)
    if not args.no_safety:
        env = UnitreeSafetyWrapper(
            env,
            go_home_steps=int(max(1, args.frequency * 4.0)),
            settle_time=0.5,
            soft_start_duration=5.0,
            health_timeout=0.5,
        )

    actions, first_step = _load_episode(args.repo_id, args.episode, args.root)

    try:
        obs, info = env.reset()
        print(f"[replay] reset complete (safety={info.get('safety_reset', False)})")

        # Align the initial pose with the dataset before replaying the sequence.
        arm_target = np.asarray(first_step["observation.state"][: control_env._arm_dof], dtype=np.float32)  # noqa: SLF001
        tau = control_env._arm_ik.solve_tau(arm_target)  # noqa: SLF001
        control_env._arm_ctrl.ctrl_dual_arm(arm_target, tau)  # noqa: SLF001
        if control_env._has_dex3 and "observation.state" in first_step:  # noqa: SLF001
            full_state = np.asarray(first_step["observation.state"], dtype=np.float32)
            if full_state.shape[0] >= (control_env._arm_dof + 2 * control_env._ee_dof):  # noqa: SLF001
                left = full_state[control_env._arm_dof : control_env._arm_dof + control_env._ee_dof]  # noqa: SLF001
                right = full_state[
                    control_env._arm_dof + control_env._ee_dof : control_env._arm_dof + 2 * control_env._ee_dof  # noqa: SLF001
                ]
                with control_env._ee_shared_mem["lock"]:  # noqa: SLF001
                    control_env._ee_shared_mem["left"][:] = left  # noqa: SLF001
                    control_env._ee_shared_mem["right"][:] = right  # noqa: SLF001
        time.sleep(max(0.0, args.settle))

        print(f"[replay] starting episode {args.episode} at {args.frequency:.1f} Hz")
        for step_idx, action in enumerate(actions):
            loop_start = time.perf_counter()
            env.step(action)
            if args.log_every and (step_idx % args.log_every == 0):
                print(f"[replay] step {step_idx}")
            elapsed = time.perf_counter() - loop_start
            delay = action_dt - elapsed
            if delay > 0:
                time.sleep(delay)
    except KeyboardInterrupt:
        print("\n[replay] interrupted by user")
    finally:
        env.close()
        print("[replay] resources released")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
