#!/usr/bin/env python3
"""Collect Unitree demonstrations using the new SDK2 pipeline."""

from __future__ import annotations

import argparse
import copy
import datetime
import os
import pickle as pkl
from pathlib import Path

import numpy as np
from absl import app
from absl import flags
from tqdm import tqdm

from unitree_examples.mappings import CONFIG_MAPPING

FLAGS = flags.FLAGS

flags.DEFINE_string("exp_name", "unitree_assemble", "Experiment key registered in CONFIG_MAPPING.")
flags.DEFINE_integer("successes_needed", 10, "Number of successful episodes to record.")
flags.DEFINE_string(
    "output_dir",
    "./demo_data",
    "Directory to store the serialized demonstrations.",
)


def _make_env(exp_name: str):
    if exp_name not in CONFIG_MAPPING:
        raise ValueError(f"Unknown experiment '{exp_name}'. Available keys: {list(CONFIG_MAPPING.keys())}")
    config = CONFIG_MAPPING[exp_name]()
    env = config.get_environment(fake_env=False, save_video=False, classifier=False)
    return env


def _default_action(env):
    try:
        return np.zeros(env.action_space.shape, dtype=np.float32)
    except AttributeError:
        sample = env.action_space.sample()
        return np.zeros_like(sample)


def main(_) -> None:
    env = _make_env(FLAGS.exp_name)

    obs, info = env.reset()
    print("[unitree_record_demos] Reset complete. Waiting for teleop inputs …")

    successes = 0
    success_goal = FLAGS.successes_needed
    transitions = []
    trajectory = []
    returns = 0.0
    progress = tqdm(total=success_goal)

    while successes < success_goal:
        action = _default_action(env)
        if isinstance(info, dict) and "intervene_action" in info and info["intervene_action"] is not None:
            action = np.asarray(info["intervene_action"], dtype=np.float32)

        next_obs, reward, terminated, truncated, info = env.step(action)
        returns += reward

        transition = copy.deepcopy(
            dict(
                observations=obs,
                actions=action,
                next_observations=next_obs,
                rewards=reward,
                dones=terminated,
                truncated=truncated,
                infos=info,
            )
        )
        trajectory.append(transition)

        obs = next_obs

        done = terminated or truncated
        if done:
            succeed = bool(info.get("succeed", False)) if isinstance(info, dict) else False
            if succeed:
                transitions.extend(copy.deepcopy(trajectory))
                successes += 1
                progress.update(1)
                progress.set_description(f"Return: {returns:.2f}")
            else:
                progress.set_description("Return: {:.2f} (failed)".format(returns))

            trajectory.clear()
            returns = 0.0
            obs, info = env.reset()

    progress.close()

    output_root = Path(FLAGS.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_path = output_root / f"{FLAGS.exp_name}_{success_goal}_demos_{timestamp}.pkl"
    with file_path.open("wb") as f:
        pkl.dump(transitions, f)

    print(f"[unitree_record_demos] Saved {success_goal} demonstrations to {file_path}")


if __name__ == "__main__":
    app.run(main)
