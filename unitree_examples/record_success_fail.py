import copy
import datetime
import os
import pickle as pkl
import threading

import numpy as np
from pathlib import Path
from absl import app, flags
from pynput import keyboard
from tqdm import tqdm

from unitree_examples.mappings import CONFIG_MAPPING
from unitree_examples.utils import observation_to_teleop, vector_to_teleop_action

FLAGS = flags.FLAGS
flags.DEFINE_string("exp_name", "unitree_assemble", "Experiment key registered in CONFIG_MAPPING.")
flags.DEFINE_integer("successes_needed", 200, "Number of successful transitions to collect.")


# Global state modified by keyboard callbacks
recording = False
await_label = False
current_segment = []
pending_segment = None
label_queue = []  # list of ("success"|"failure", segment)
state_lock = threading.Lock()


def _print(msg: str) -> None:
    print(f"[record_success_fail] {msg}")


def _start_recording():
    global recording, current_segment
    recording = True
    current_segment = []
    _print("Recording started. Press 'e' to finish this segment.")


def _queue_segment():
    global recording, await_label, pending_segment, current_segment
    recording = False
    if current_segment:
        pending_segment = list(current_segment)
        current_segment = []
        await_label = True
        _print("Segment captured. Label with '1' (success) or '0' (failure).")
    else:
        _print("Segment was empty; nothing to label.")


def _enqueue_labeled_segment(label: str):
    global await_label, pending_segment
    if pending_segment:
        label_queue.append((label, list(pending_segment)))
        pending_segment = None
        await_label = False
        if label == "success":
            _print("Segment queued as SUCCESS.")
        else:
            _print("Segment queued as FAILURE.")
    else:
        _print("No pending segment to label.")


def on_press(key):
    global recording, await_label
    try:
        char = key.char
    except AttributeError:
        char = None

    with state_lock:
        if char == 's':
            if await_label:
                _print("Label the pending segment first (press 1/0).")
            elif recording:
                _print("Already recording. Press 'e' to end the current segment.")
            else:
                _start_recording()
        elif char == 'e':
            if recording:
                _queue_segment()
            else:
                _print("No active recording to end.")
        elif char == '1':
            if await_label:
                _enqueue_labeled_segment("success")
            else:
                _print("No pending segment to label as success.")
        elif char == '0':
            if await_label:
                _enqueue_labeled_segment("failure")
            else:
                _print("No pending segment to label as failure.")


def main(_):
    global recording, await_label, current_segment, pending_segment

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    assert FLAGS.exp_name in CONFIG_MAPPING, "Experiment not found."
    config = CONFIG_MAPPING[FLAGS.exp_name]()
    env = config.get_environment(fake_env=False, save_video=False, classifier=False)

    obs, info = env.reset()
    success_segments: list = []
    failure_segments: list = []
    success_goal = FLAGS.successes_needed
    pbar = tqdm(total=success_goal)

    _print("Controls: 's' start recording, 'e' end recording, '1' success, '0' failure. Press Ctrl+C to exit.")

    try:
        while len(success_segments) < success_goal:
            action_vector = np.zeros(env.action_space.shape, dtype=np.float32)
            if isinstance(info, dict) and "intervene_action_vector" in info:
                action_vector = np.asarray(info["intervene_action_vector"], dtype=np.float32)

            next_obs, reward, done, truncated, info = env.step(action_vector)

            transition = dict(
                observations=observation_to_teleop(obs),
                actions=info.get("intervene_action", vector_to_teleop_action(action_vector)),
                next_observations=observation_to_teleop(next_obs),
                rewards=reward,
                masks=1.0 - done,
                dones=done,
                infos=info,
            )

            with state_lock:
                if recording:
                    current_segment.append(copy.deepcopy(transition))

                while label_queue:
                    label, segment = label_queue.pop(0)
                    if label == "success":
                        success_segments.append(segment)
                        pbar.update(1)
                    else:
                        failure_segments.append(segment)

            obs = next_obs

            if done or truncated:
                with state_lock:
                    if recording and current_segment:
                        _print("Episode ended during recording; segment queued for labeling.")
                        _queue_segment()
                obs, info = env.reset()
    except KeyboardInterrupt:
        _print("Interrupted by user.")
    finally:
        listener.stop()

    base_dir = Path("classifier_data") / FLAGS.exp_name
    success_dir = base_dir / "success"
    failure_dir = base_dir / "failure"
    success_dir.mkdir(parents=True, exist_ok=True)
    failure_dir.mkdir(parents=True, exist_ok=True)

    for idx, segment in enumerate(success_segments):
        file_path = success_dir / f"episode_{idx:04d}.pkl"
        with file_path.open("wb") as f:
            pkl.dump(segment, f)
    print(f"Saved {len(success_segments)} success segments under {success_dir}")

    for idx, segment in enumerate(failure_segments):
        file_path = failure_dir / f"episode_{idx:04d}.pkl"
        with file_path.open("wb") as f:
            pkl.dump(segment, f)
    print(f"Saved {len(failure_segments)} failure segments under {failure_dir}")


if __name__ == "__main__":
    app.run(main)
