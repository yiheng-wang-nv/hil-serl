#!/usr/bin/env python3

from __future__ import annotations

import os
import re
from dataclasses import dataclass
import math
from typing import Dict, List, Tuple, Sequence, Iterator

import cv2
import jax
import jax.numpy as jnp
import numpy as np
import optax
from absl import app, flags
from flax.training import checkpoints
from tqdm import tqdm

from serl_launcher.networks.reward_classifier import create_classifier
from serl_launcher.vision.data_augmentations import batched_random_crop

flags.DEFINE_string("exp_name", None, "Experiment name (unused, for consistency).")
flags.DEFINE_string("data_dir", None, "Root directory containing class subfolders.")
flags.DEFINE_integer("num_classes", 6, "Number of reward classes.")
flags.DEFINE_integer("val_episodes", 10, "Number of distinct episodes reserved for validation.")
flags.DEFINE_integer("num_epochs", 150, "Training epochs.")
flags.DEFINE_integer("batch_size", 256, "Batch size.")
flags.DEFINE_integer("image_height", 128, "Resize height.")
flags.DEFINE_integer("image_width", 128, "Resize width.")
flags.DEFINE_string("output_dir", "classifier_ckpt", "Directory to save checkpoints.")
flags.DEFINE_integer("val_freq", 1, "Validate every N epochs.")
flags.DEFINE_integer("seed", 0, "Random seed for shuffling episodes.")

FLAGS = flags.FLAGS

FRAME_REGEX = re.compile(r"(episode_(\d+)_(\d+))_color_(\d)\.\w+$")
CAMERA_INDICES: Sequence[int] = (0, 1, 2)
CAMERA_KEY_MAP = {
    0: "video.room_view",
    1: "video.left_wrist_view",
    2: "video.right_wrist_view",
}
IMAGE_KEYS = tuple(CAMERA_KEY_MAP[idx] for idx in CAMERA_INDICES)


@dataclass
class Sample:
    image_paths: Tuple[str, ...]
    label: int
    episode_id: int


def list_images(class_dir: str) -> List[str]:
    entries = [name for name in os.listdir(class_dir) if name.lower().endswith(".jpg")]
    entries.sort()
    return [os.path.join(class_dir, name) for name in entries]


def parse_frame_record(path: str):
    basename = os.path.basename(path)
    match = FRAME_REGEX.match(basename)
    if not match:
        raise ValueError(f"Filename {basename} does not match expected pattern 'episode_xxxx_xxxxxx_color_i.ext'")
    frame_key = match.group(1)
    episode_id = int(match.group(2))
    cam_idx = int(match.group(4))
    return frame_key, episode_id, cam_idx


def load_dataset(root: str, num_classes: int) -> Dict[int, List[Sample]]:
    episode_map: Dict[int, List[Sample]] = {}
    skipped = 0
    for class_idx in range(num_classes):
        class_dir = os.path.join(root, f"class_{class_idx}")
        if not os.path.isdir(class_dir):
            raise ValueError(f"Missing directory: {class_dir}")
        image_files = list_images(class_dir)
        if not image_files:
            raise ValueError(f"No images found in {class_dir}")
        per_frame: Dict[Tuple[int, str], Dict[int, str]] = {}
        for fname in image_files:
            frame_key, ep_id, cam_idx = parse_frame_record(fname)
            per_frame.setdefault((ep_id, frame_key), {})[cam_idx] = fname
        for (ep_id, frame_key), cam_dict in sorted(per_frame.items()):
            if not all(idx in cam_dict for idx in CAMERA_INDICES):
                skipped += 1
                continue
            ordered_paths = tuple(cam_dict[idx] for idx in CAMERA_INDICES)
            sample = Sample(image_paths=ordered_paths, label=class_idx, episode_id=ep_id)
            episode_map.setdefault(ep_id, []).append(sample)
    if not episode_map:
        raise ValueError("No data loaded; check paths.")
    if skipped:
        print(f"Warning: skipped {skipped} frames lacking full camera triplets.")
    return episode_map


def split_episodes(
    episodes: Dict[int, List[Sample]],
    val_episodes: int,
    seed: int,
) -> Tuple[List[Sample], List[Sample], Dict[str, List[int]]]:
    ep_ids = sorted(episodes.keys())
    rng = np.random.default_rng(seed)
    rng.shuffle(ep_ids)
    val_set = ep_ids[:val_episodes]
    train_set = ep_ids[val_episodes:]
    if not train_set or not val_set:
        raise ValueError("Split resulted in empty train or val set; adjust val_episodes.")

    train_samples = [sample for ep in train_set for sample in episodes[ep]]
    val_samples = [sample for ep in val_set for sample in episodes[ep]]
    split_info = {"train_ids": train_set, "val_ids": val_set}
    return train_samples, val_samples, split_info


def _read_image(path: str, height: int, width: int) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Failed to read {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)
    return img


def load_observations(samples: Sequence[Sample], height: int, width: int) -> Dict[str, np.ndarray]:
    buffers = {key: [] for key in IMAGE_KEYS}
    for sample in samples:
        for cam_idx, key in CAMERA_KEY_MAP.items():
            path = sample.image_paths[cam_idx]
            buffers[key].append(_read_image(path, height, width))
    return {key: np.stack(images, axis=0).astype(np.uint8) for key, images in buffers.items()}


def batched_augment(rng, obs_dict):
    return {
        key: batched_random_crop(value, rng, padding=4, num_batch_dims=1)
        for key, value in obs_dict.items()
    }


def add_stack_dim(obs_dict):
    return {key: value[:, None, ...] for key, value in obs_dict.items()}


def update_confusion(confusion: np.ndarray, preds: np.ndarray, labels: np.ndarray):
    preds = preds.astype(np.int32)
    labels = labels.astype(np.int32)
    np.add.at(confusion, (labels, preds), 1)


def compute_macro_f1(confusion: np.ndarray) -> float:
    tp = np.diag(confusion).astype(np.float32)
    precision_den = confusion.sum(axis=0).astype(np.float32)
    recall_den = confusion.sum(axis=1).astype(np.float32)
    precision = np.divide(tp, precision_den, out=np.zeros_like(tp), where=precision_den > 0)
    recall = np.divide(tp, recall_den, out=np.zeros_like(tp), where=recall_den > 0)
    denom = precision + recall
    f1_per_class = np.divide(2 * precision * recall, denom, out=np.zeros_like(tp), where=denom > 0)
    return float(np.mean(f1_per_class)) if f1_per_class.size else float("nan")


def iterate_batches(
    samples: List[Sample],
    batch_size: int,
    shuffle: bool,
    rng: np.random.Generator | None,
    drop_last: bool,
) -> Iterator[Tuple[List[Sample], np.ndarray]]:
    if not samples:
        return
    indices = np.arange(len(samples))
    if shuffle:
        if rng is None:
            rng = np.random.default_rng()
        rng.shuffle(indices)
    for start in range(0, len(indices), batch_size):
        batch_idx = indices[start : start + batch_size]
        if drop_last and len(batch_idx) < batch_size:
            break
        batch = [samples[i] for i in batch_idx]
        labels = np.array([s.label for s in batch], dtype=np.int32)
        yield batch, labels


def main(argv):
    del argv
    if FLAGS.data_dir is None:
        raise ValueError("--data_dir must be specified")

    episodes = load_dataset(FLAGS.data_dir, FLAGS.num_classes)
    train_samples, val_samples, split_info = split_episodes(episodes, FLAGS.val_episodes, FLAGS.seed)
    print(
        f"Loaded {len(train_samples)} train samples across {len(split_info['train_ids'])} episodes "
        f"(IDs: {sorted(split_info['train_ids'])})."
    )
    print(
        f"Loaded {len(val_samples)} val samples across {len(split_info['val_ids'])} episodes "
        f"(IDs: {sorted(split_info['val_ids'])})."
    )

    preview_count = min(len(train_samples), FLAGS.batch_size)
    sample_chunk = train_samples[:preview_count]
    if not sample_chunk:
        raise ValueError("Training set is empty after loading; cannot proceed.")
    sample_obs = load_observations(sample_chunk, FLAGS.image_height, FLAGS.image_width)
    sample_obs = add_stack_dim(sample_obs)

    rng = jax.random.PRNGKey(FLAGS.seed)
    train_shuffle_rng = np.random.default_rng(FLAGS.seed)
    rng, key = jax.random.split(rng)
    classifier = create_classifier(
        key,
        sample_obs,
        image_keys=list(IMAGE_KEYS),
        n_way=FLAGS.num_classes,
    )

    @jax.jit
    def train_step(state, obs, labels, key):
        labels = labels.squeeze().astype(jnp.int32)
        one_hot = jax.nn.one_hot(labels, FLAGS.num_classes)

        def loss_fn(params):
            logits = state.apply_fn({"params": params}, obs, rngs={"dropout": key}, train=True)
            loss = optax.softmax_cross_entropy(logits, one_hot).mean()
            preds = jnp.argmax(logits, axis=-1)
            return loss, preds

        (loss, preds), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        acc = jnp.mean(preds == labels)
        new_state = state.apply_gradients(grads=grads)
        return new_state, loss, acc, preds

    @jax.jit
    def eval_step(state, obs, labels):
        labels = labels.squeeze().astype(jnp.int32)
        logits = state.apply_fn({"params": state.params}, obs, train=False)
        loss = optax.softmax_cross_entropy(logits, jax.nn.one_hot(labels, FLAGS.num_classes)).mean()
        preds = jnp.argmax(logits, axis=-1)
        acc = jnp.mean(preds == labels)
        return loss, acc, preds

    drop_last = len(train_samples) >= FLAGS.batch_size
    num_train_batches = (
        len(train_samples) // FLAGS.batch_size
        if drop_last
        else math.ceil(len(train_samples) / max(1, FLAGS.batch_size))
    )
    best_val_f1 = -float("inf")
    for epoch in range(FLAGS.num_epochs):
        train_loss_sum = 0.0
        train_correct = 0.0
        train_count = 0
        train_confusion = np.zeros((FLAGS.num_classes, FLAGS.num_classes), dtype=np.int64)
        batch_iter = iterate_batches(
            train_samples,
            FLAGS.batch_size,
            shuffle=True,
            rng=train_shuffle_rng,
            drop_last=drop_last,
        )
        batch_progress = tqdm(
            total=num_train_batches,
            desc=f"Epoch {epoch + 1}/{FLAGS.num_epochs}",
            leave=False,
        )
        for batch_samples, labels in batch_iter:
            obs = load_observations(batch_samples, FLAGS.image_height, FLAGS.image_width)
            rng, aug_key = jax.random.split(rng)
            obs = batched_augment(aug_key, obs)
            obs = add_stack_dim(obs)
            rng, step_key = jax.random.split(rng)
            classifier, tr_loss, tr_acc, batch_preds = train_step(classifier, obs, labels, step_key)
            batch_size_actual = labels.shape[0]
            train_loss_sum += float(tr_loss) * batch_size_actual
            train_correct += float(tr_acc) * batch_size_actual
            train_count += batch_size_actual
            update_confusion(train_confusion, np.asarray(batch_preds), labels)
            batch_progress.update(1)
        batch_progress.close()
        train_loss_epoch = train_loss_sum / train_count if train_count else float("nan")
        train_acc_epoch = train_correct / train_count if train_count else float("nan")
        train_f1_epoch = compute_macro_f1(train_confusion)

        if (epoch + 1) % FLAGS.val_freq == 0 or epoch == 0:
            val_loss_sum = 0.0
            val_acc_sum = 0.0
            val_count = 0
            val_confusion = np.zeros((FLAGS.num_classes, FLAGS.num_classes), dtype=np.int64)
            for val_samples_batch, val_labels in iterate_batches(
                val_samples,
                FLAGS.batch_size,
                shuffle=False,
                rng=None,
                drop_last=False,
            ):
                val_obs = load_observations(val_samples_batch, FLAGS.image_height, FLAGS.image_width)
                val_obs = add_stack_dim(val_obs)
                batch_loss, batch_acc, val_preds = eval_step(classifier, val_obs, val_labels)
                batch_size_actual = val_labels.shape[0]
                val_loss_sum += float(batch_loss) * batch_size_actual
                val_acc_sum += float(batch_acc) * batch_size_actual
                val_count += batch_size_actual
                update_confusion(val_confusion, np.asarray(val_preds), val_labels)
            if val_count > 0:
                val_loss = val_loss_sum / val_count
                val_acc = val_acc_sum / val_count
            else:
                val_loss = float("nan")
                val_acc = float("nan")
            val_f1 = compute_macro_f1(val_confusion)
            print(
                f"Epoch {epoch + 1:04d}: "
                f"train_loss={train_loss_epoch:.4f}, train_acc={train_acc_epoch:.4f}, train_f1={train_f1_epoch:.4f}; "
                f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}, val_f1={val_f1:.4f}"
            )
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                checkpoints.save_checkpoint(
                    FLAGS.output_dir,
                    classifier,
                    step=epoch + 1,
                    overwrite=True,
                    prefix="best_",
                )
                print(f"Saved new best checkpoint with val_f1={val_f1:.4f} at epoch {epoch + 1}.")

    os.makedirs(FLAGS.output_dir, exist_ok=True)
    checkpoints.save_checkpoint(FLAGS.output_dir, classifier, step=FLAGS.num_epochs, overwrite=True, prefix="")
    print(f"Saved final classifier checkpoint to {FLAGS.output_dir}")


if __name__ == "__main__":
    app.run(main)
