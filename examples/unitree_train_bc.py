#!/usr/bin/env python3

import os
import time
from typing import Dict, Iterable, Tuple, List

import json
import cv2
TARGET_IMAGE_SIZE = (128, 128)


import gymnasium as gym
import jax
import numpy as np
import tqdm
from absl import app, flags
from flax.core import frozen_dict
from flax.training import checkpoints
from gymnasium.wrappers.record_episode_statistics import RecordEpisodeStatistics

from serl_launcher.agents.continuous.bc import BCAgent
from serl_launcher.data.data_store import MemoryEfficientReplayBufferDataStore
from serl_launcher.utils.launcher import make_bc_agent, make_wandb_logger

from experiments.mappings import CONFIG_MAPPING

FLAGS = flags.FLAGS

flags.DEFINE_string("exp_name", None, "Name of experiment corresponding to folder.")
flags.DEFINE_integer("seed", 42, "Random seed.")

flags.DEFINE_string("dataset_dir", None, "Path to XR teleoperation dataset (directory with episode_* or data.json).")

flags.DEFINE_string("bc_checkpoint_path", None, "Path to save checkpoints.")
flags.DEFINE_integer("eval_n_trajs", 0, "Number of trajectories to evaluate.")
flags.DEFINE_integer("train_steps", 20_000, "Number of pretraining steps.")
flags.DEFINE_boolean("save_video", False, "Save video of the evaluation rollouts.")

flags.DEFINE_boolean(
    "debug", False, "Debug mode (disables wandb logging)."
)

DEVICES = jax.devices()
PRIMARY_DEVICE = DEVICES[0]


class OfflineUnitreeEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, observation_space: gym.Space, action_space: gym.Space, sample_observation: Dict):
        super().__init__()
        self.observation_space = observation_space
        self.action_space = action_space
        self._sample_observation = sample_observation

    def reset(self, *, seed: int | None = None, options: Dict | None = None):
        return self._sample_observation.copy(), {}

    def step(self, action: np.ndarray):
        raise RuntimeError("OfflineUnitreeEnv does not implement step(); it is a placeholder for dataset training.")

#
# XR teleoperation raw dataset loader (no torch, no HuggingFace dependency)
#
class XRTeleopDataset:
    def __init__(self, root: str):
        self.root = os.path.abspath(root)
        # Determine episodes
        if os.path.isfile(os.path.join(self.root, "data.json")):
            self.episode_dirs = [self.root]
        else:
            # Look for episode_* subdirs
            subdirs = [
                os.path.join(self.root, d)
                for d in sorted(os.listdir(self.root))
                if d.startswith("episode_") and os.path.isdir(os.path.join(self.root, d))
            ]
            if not subdirs:
                raise ValueError(f"No episode_* directories or data.json found under {self.root}")
            self.episode_dirs = subdirs

        # Build flat frames index and episode boundaries
        self._frames = []  # each entry: dict with keys we need
        self._from = []
        self._to = []
        cursor = 0
        for ep_dir in self.episode_dirs:
            data_path = os.path.join(ep_dir, "data.json")
            with open(data_path, "r") as f:
                data_json = json.load(f)
            frames = data_json.get("data", [])
            for fr in frames:
                # Compose state/action (28-dim): [left_arm(7), right_arm(7), left_ee(7), right_ee(7)]
                def _get_qpos(tree: dict, key: str) -> np.ndarray:
                    arr = np.array(tree.get(key, {}).get("qpos", []), dtype=np.float32)
                    return arr
                states = fr.get("states", {})
                actions = fr.get("actions", {})
                state = np.concatenate(
                    [
                        _get_qpos(states, "left_arm"),
                        _get_qpos(states, "right_arm"),
                        _get_qpos(states, "left_ee"),
                        _get_qpos(states, "right_ee"),
                    ],
                    axis=0,
                ).astype(np.float32)
                action = np.concatenate(
                    [
                        _get_qpos(actions, "left_arm"),
                        _get_qpos(actions, "right_arm"),
                        _get_qpos(actions, "left_ee"),
                        _get_qpos(actions, "right_ee"),
                    ],
                    axis=0,
                ).astype(np.float32)
                # Image paths
                colors = fr.get("colors", {})
                cam_room = os.path.join(ep_dir, colors.get("color_0", ""))
                cam_left = os.path.join(ep_dir, colors.get("color_1", ""))
                cam_right = os.path.join(ep_dir, colors.get("color_2", ""))
                self._frames.append(
                    {
                        # Match the keys expected downstream by config.dataset_* mapping
                        "observation.state": state,
                        "action": action,
                        "observation.images.cam_room": cam_room,
                        "observation.images.cam_left_wrist": cam_left,
                        "observation.images.cam_right_wrist": cam_right,
                    }
                )
            self._from.append(cursor)
            cursor += len(frames)
            self._to.append(cursor)

        self._from = np.array(self._from, dtype=np.int64)
        self._to = np.array(self._to, dtype=np.int64)

    def __len__(self) -> int:
        return len(self._frames)

    @property
    def num_episodes(self) -> int:
        return len(self._from)

    @property
    def episode_data_index(self) -> Dict[str, np.ndarray]:
        return {"from": self._from, "to": self._to}

    def _load_image(self, path: str) -> np.ndarray:
        if not path or not os.path.isfile(path):
            raise FileNotFoundError(f"Image path not found: {path}")
        img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise OSError(f"Failed to read image: {path}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return img_rgb  # (H, W, 3) uint8

    def __getitem__(self, idx: int) -> Dict:
        rec = self._frames[idx]
        # Load images on demand
        out = {
            "observation.state": rec["observation.state"],
            "action": rec["action"],
            "observation.images.cam_room": self._load_image(rec["observation.images.cam_room"]),
            "observation.images.cam_left_wrist": self._load_image(rec["observation.images.cam_left_wrist"]),
            "observation.images.cam_right_wrist": self._load_image(rec["observation.images.cam_right_wrist"]),
        }
        return out

def _convert_image(image) -> np.ndarray:
    # Expect numpy HWC image; return (1, H, W, C) float32 in [0, 255]
    if image.ndim != 3 or image.shape[-1] not in (1, 3):
        raise ValueError(f"Expected HWC image with 1 or 3 channels; got {image.shape}.")
    if image.dtype != np.float32:
        image = image.astype(np.float32)
    return np.expand_dims(image, axis=0)

def _build_observation_dict(
    frame: Dict,
    config,
    camera_map: Dict[str, str],
    enabled_cameras: Iterable[str],
) -> Dict[str, np.ndarray]:
    observation = {"state": np.asarray(frame[config.dataset_state_key], dtype=np.float32)}

    for dataset_key, obs_key in camera_map.items():
        if obs_key not in enabled_cameras:
            continue
        if dataset_key not in frame:
            continue
        image = _convert_image(frame[dataset_key])
        resized_frames = []
        for i in range(image.shape[0]):
            frame_i = image[i]
            frame_i = cv2.resize(frame_i.astype(np.float32), TARGET_IMAGE_SIZE[::-1], interpolation=cv2.INTER_LINEAR)
            resized_frames.append(frame_i.astype(np.float32))
        observation[obs_key] = np.stack(resized_frames, axis=0)

    return observation


def _detect_available_cameras(
    sample_frame: Dict,
    config,
) -> List[str]:
    enabled = []
    for dataset_key, obs_key in config.dataset_camera_map.items():
        if dataset_key in sample_frame:
            enabled.append(obs_key)
    return enabled


def _build_observation_space(sample_observation: Dict[str, np.ndarray]) -> gym.Space:
    spaces_dict = {
        "state": gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=sample_observation["state"].shape,
            dtype=np.float32,
        )
    }

    for key, value in sample_observation.items():
        if key == "state":
            continue
        spaces_dict[key] = gym.spaces.Box(
            low=0,
            high=255,
            shape=value.shape,
            dtype=np.float32,
        )

    return gym.spaces.Dict(spaces_dict)


def _build_action_space(sample_action: np.ndarray) -> gym.Space:
    # Use dataset-derived action dimension; keep bounds broad
    shape = sample_action.shape
    low = -np.inf * np.ones(shape, dtype=np.float32)
    high = np.inf * np.ones(shape, dtype=np.float32)
    return gym.spaces.Box(low=low, high=high, dtype=np.float32)


def _prepare_offline_env(dataset, config) -> Tuple[OfflineUnitreeEnv, Dict[str, np.ndarray], np.ndarray, List[str]]:
    if len(dataset) == 0:
        raise ValueError("Dataset is empty. Ensure the dataset_dir points to a valid dataset with at least one frame.")

    sample_frame = dataset[0]
    available_cameras = _detect_available_cameras(sample_frame, config)

    if not available_cameras:
        raise ValueError(
            "No vision modalities detected or enabled. The current BC implementation expects at least one image key."
        )

    sample_observation = _build_observation_dict(
        sample_frame,
        config,
        config.dataset_camera_map,
        available_cameras,
    )

    sample_action = np.asarray(sample_frame[config.dataset_action_key], dtype=np.float32)

    observation_space = _build_observation_space(sample_observation)
    action_space = _build_action_space(sample_action)

    offline_env = OfflineUnitreeEnv(observation_space, action_space, sample_observation)
    return offline_env, sample_observation, sample_action, available_cameras


def _estimate_num_transitions(dataset) -> int:
    from_indices = dataset.episode_data_index["from"]
    to_indices = dataset.episode_data_index["to"]
    lengths = to_indices - from_indices
    return int(np.sum(np.maximum(lengths - 1, 0)))


def populate_replay_buffer_from_dataset(
    dataset,
    config,
    replay_buffer: MemoryEfficientReplayBufferDataStore,
    enabled_cameras: Iterable[str],
) -> int:
    total_transitions = _estimate_num_transitions(dataset)
    pbar = tqdm.tqdm(
        total=total_transitions,
        desc="Loading demonstrations",
        dynamic_ncols=True,
    )

    inserted = 0
    episode_data_index = dataset.episode_data_index

    for ep_idx in range(dataset.num_episodes):
        start = int(episode_data_index["from"][ep_idx].item())
        end = int(episode_data_index["to"][ep_idx].item())

        if end - start < 2:
            continue

        for idx in range(start, end - 1):
            current_frame = dataset[idx]
            next_frame = dataset[idx + 1]

            observations = _build_observation_dict(
                current_frame,
                config,
                config.dataset_camera_map,
                enabled_cameras,
            )
            next_observations = _build_observation_dict(
                next_frame,
                config,
                config.dataset_camera_map,
                enabled_cameras,
            )

            action = np.asarray(current_frame[config.dataset_action_key], dtype=np.float32)
            done = idx + 1 == end - 1

            transition = dict(
                observations=observations,
                actions=action,
                next_observations=next_observations,
                rewards=np.float32(0.0),
                masks=np.float32(0.0 if done else 1.0),
                dones=done,
            )

            replay_buffer.insert(transition)
            inserted += 1
            pbar.update(1)

    pbar.close()
    return inserted


def print_green(x):
    return print("\033[92m {}\033[00m".format(x))


def print_yellow(x):
    return print("\033[93m {}\033[00m".format(x))


def eval_policy(env, bc_agent: BCAgent, sampling_rng):
    success_counter = 0
    time_list = []
    for episode in range(FLAGS.eval_n_trajs):
        obs, _ = env.reset()
        done = False
        start_time = time.time()
        while not done:
            sampling_rng, key = jax.random.split(sampling_rng)
            actions = bc_agent.sample_actions(
                observations=jax.device_put(obs, PRIMARY_DEVICE),
                seed=key,
            )
            actions = np.asarray(jax.device_get(actions))
            next_obs, reward, done, truncated, info = env.step(actions)
            obs = next_obs
            if done:
                if reward:
                    dt = time.time() - start_time
                    time_list.append(dt)
                success_counter += reward
                print(f"{success_counter}/{episode + 1}")
    if FLAGS.eval_n_trajs:
        print(f"success rate: {success_counter / FLAGS.eval_n_trajs}")
    if time_list:
        print(f"average time: {np.mean(time_list)}")


def train_bc_agent(
    bc_agent: BCAgent,
    bc_replay_buffer,
    train_steps: int,
    batch_size: int,
    log_period: int,
    checkpoint_path: str | None,
    wandb_logger=None,
):
    bc_replay_iterator = bc_replay_buffer.get_iterator(
        sample_args={
            "batch_size": batch_size,
            "pack_obs_and_next_obs": False,
        },
        device=PRIMARY_DEVICE,
    )
    for step in tqdm.tqdm(
        range(train_steps),
        dynamic_ncols=True,
        desc="bc_pretraining",
    ):
        batch = next(bc_replay_iterator)
        bc_agent, bc_update_info = bc_agent.update(batch)
        if wandb_logger and step % log_period == 0:
            wandb_logger.log({"bc": bc_update_info}, step=step)
        if checkpoint_path and step > train_steps - 100 and step % 10 == 0:
            checkpoints.save_checkpoint(
                os.path.abspath(checkpoint_path),
                bc_agent.state,
                step=step,
                keep=5,
            )

    print_green("BC pretraining finished.")
    if checkpoint_path:
        print_green(f"Checkpoints saved under {checkpoint_path}")


def main(_):
    assert FLAGS.exp_name in CONFIG_MAPPING, "Experiment folder not found."
    config = CONFIG_MAPPING[FLAGS.exp_name]()

    eval_mode = FLAGS.eval_n_trajs > 0

    if not eval_mode:
        if FLAGS.dataset_dir is None:
            raise ValueError("--dataset_dir must be provided for training.")
        if FLAGS.bc_checkpoint_path is None:
            raise ValueError("--bc_checkpoint_path must be specified to store checkpoints.")
        if os.path.isdir(os.path.join(FLAGS.bc_checkpoint_path, f"checkpoint_{FLAGS.train_steps}")):
            raise ValueError(
                f"Checkpoint for step {FLAGS.train_steps} already exists in {FLAGS.bc_checkpoint_path}. "
                "Please choose a new path or remove the existing checkpoint."
            )

        dataset = XRTeleopDataset(FLAGS.dataset_dir)

        offline_env, sample_observation, sample_action, enabled_cameras = _prepare_offline_env(dataset, config)

        config.image_keys = enabled_cameras

        sample_obs_frozen = frozen_dict.freeze(sample_observation)

        bc_agent: BCAgent = make_bc_agent(
            seed=FLAGS.seed,
            sample_obs=sample_obs_frozen,
            sample_action=sample_action,
            image_keys=config.image_keys,
            encoder_type=config.encoder_type,
        )

        replay_buffer = MemoryEfficientReplayBufferDataStore(
            offline_env.observation_space,
            offline_env.action_space,
            capacity=config.replay_buffer_capacity,
            image_keys=tuple(config.image_keys),
        )

        num_inserted = populate_replay_buffer_from_dataset(
            dataset,
            config,
            replay_buffer,
            enabled_cameras=config.image_keys,
        )
        print_green(f"Loaded {num_inserted} transitions into the replay buffer.")

        wandb_logger = make_wandb_logger(
            project="hil-serl",
            description=FLAGS.exp_name,
            debug=FLAGS.debug,
        )

        bc_agent.config["batch_size"] = config.batch_size
        train_bc_agent(
            bc_agent=bc_agent,
            bc_replay_buffer=replay_buffer,
            train_steps=FLAGS.train_steps,
             batch_size=config.batch_size,
            log_period=config.log_period,
            checkpoint_path=FLAGS.bc_checkpoint_path,
            wandb_logger=wandb_logger,
        )

    else:
        env = config.get_environment(
            fake_env=False,
            save_video=FLAGS.save_video,
            classifier=True,
        )
        env = RecordEpisodeStatistics(env)

        sample_obs, _ = env.reset()
        sample_action = env.action_space.sample().astype(np.float32)

        bc_agent: BCAgent = make_bc_agent(
            seed=FLAGS.seed,
            sample_obs=frozen_dict.freeze(sample_obs),
            sample_action=sample_action,
            image_keys=tuple(config.image_keys),
            encoder_type=config.encoder_type,
        )

        if FLAGS.bc_checkpoint_path is None:
            raise ValueError("--bc_checkpoint_path must point to a trained checkpoint when running evaluation.")

        bc_ckpt = checkpoints.restore_checkpoint(
            FLAGS.bc_checkpoint_path,
            bc_agent.state,
        )
        bc_agent = bc_agent.replace(state=bc_ckpt)

        print_green("Starting evaluation rollouts.")
        rng = jax.random.PRNGKey(FLAGS.seed)
        sampling_rng = jax.device_put(rng, PRIMARY_DEVICE)
        eval_policy(env, bc_agent, sampling_rng)


if __name__ == "__main__":
    app.run(main)

