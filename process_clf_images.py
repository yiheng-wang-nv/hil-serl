#!/usr/bin/env python3
"""
Utility to tidy Unitree XR teleop episodes and split multi-camera frames into
reward-class folders for classifier training.

Steps per dataset:
1. Remove redundant folders (`audios/`, `depths/`) under each episode.
2. Read `labels.csv` to obtain stage boundaries for every episode.
3. For each frame, keep the original three camera images (`*_color_0/1/2.jpg`)
   and copy them into `labels/class_X/`, preserving per-camera suffixes.
"""

import argparse
import csv
import shutil
from collections import defaultdict
from pathlib import Path

CATEGORIES = [f"class_{i}" for i in range(6)]


def cleanup_episode(episode_path: Path):
    """Delete redundant folders (audios/depths) while keeping color_* images."""
    extra_folders = ["audios", "depths"]
    for folder_name in extra_folders:
        folder_path = episode_path / folder_name
        if folder_path.exists() and folder_path.is_dir():
            shutil.rmtree(folder_path)
            print(f"  Deleted {folder_name}/")


def read_labels_csv(csv_path: Path) -> dict:
    """Load stage boundaries from labels.csv."""
    labels = {}
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        reader.fieldnames = [field.strip() for field in reader.fieldnames]
        for row in reader:
            row = {k.strip(): v.strip() for k, v in row.items()}
            episode_num = int(row["episode"])
            labels[episode_num] = {
                "pickup_left": int(row["pickup_left"]),
                "pickup_right": int(row["pickup_right"]),
                "align_central_aperture": int(row["align_central_aperture"]),
                "assemble": int(row["assemble"]),
                "placement": int(row["placement"]),
                "end": int(row["end"]),
            }
    return labels


def get_category_for_frame(frame_num: int, episode_labels: dict) -> str | None:
    if frame_num < episode_labels["pickup_left"] - 10:
        return "class_0"
    if episode_labels["pickup_left"] <= frame_num < episode_labels["pickup_right"] - 10:
        return "class_1"
    if episode_labels["pickup_right"] <= frame_num < episode_labels["align_central_aperture"] - 10:
        return "class_2"
    if episode_labels["align_central_aperture"] <= frame_num < episode_labels["assemble"] - 10:
        return "class_3"
    if episode_labels["assemble"] <= frame_num < episode_labels["placement"] - 10:
        return "class_4"
    if episode_labels["placement"] <= frame_num < episode_labels["end"]:
        return "class_5"
    return None


def collect_frame_ids(colors_path: Path) -> list[str]:
    frame_ids = set()
    for cam_idx in (0, 1, 2):
        for img_path in colors_path.glob(f"*_color_{cam_idx}.jpg"):
            frame_ids.add(img_path.stem.rsplit("_color_", 1)[0])
    if not frame_ids:
        for img_path in colors_path.glob("*_color_*.jpg"):
            frame_ids.add(img_path.stem.rsplit("_color_", 1)[0])
    return sorted(frame_ids)


def process_episode(
    episode_path: Path,
    episode_num: int,
    episode_labels: dict,
    category_folders: dict[str, Path],
):
    colors_path = episode_path / "colors"
    if not colors_path.exists():
        print(f"  Warning: {colors_path} missing, skip labels split.")
        return

    frame_ids = collect_frame_ids(colors_path)
    if not frame_ids:
        print(f"  Warning: No *_color_*.jpg found under {colors_path}")
        return

    category_counts = defaultdict(int)
    for frame_id in frame_ids:
        try:
            frame_num = int(frame_id)
        except ValueError:
            print(f"    Skip frame {frame_id}: cannot parse integer frame id.")
            continue

        category = get_category_for_frame(frame_num, episode_labels)
        if category is None:
            continue

        missing_cam = False
        for cam_idx in (0, 1, 2):
            src = colors_path / f"{frame_id}_color_{cam_idx}.jpg"
            if not src.exists():
                missing_cam = True
                break

        if missing_cam:
            print(f"    Skip frame {frame_id}: incomplete camera set.")
            continue

        for cam_idx in (0, 1, 2):
            src = colors_path / f"{frame_id}_color_{cam_idx}.jpg"
            dest_name = f"{episode_path.name}_{frame_id}_color_{cam_idx}.jpg"
            dest = category_folders[category] / dest_name
            shutil.copy2(src, dest)
        category_counts[category] += 1

    summary = ", ".join(f"{k}:{v}" for k, v in sorted(category_counts.items()))
    print(f"    Copied frame triplets -> {summary}")


def main():
    parser = argparse.ArgumentParser(
        description="Clean Unitree episodes and split multi-camera frames into reward classes."
    )
    parser.add_argument(
        "base_path",
        type=str,
        help="Base directory containing episode folders (e.g., /path/to/install_trocar_from_tray)",
    )
    parser.add_argument(
        "--labels-csv",
        type=str,
        default=None,
        help="Path to labels.csv (defaults to <base_path>/labels.csv)",
    )

    args = parser.parse_args()
    base_path = Path(args.base_path)
    if not base_path.exists():
        print(f"Error: Base path {base_path} does not exist!")
        return

    labels_csv = Path(args.labels_csv) if args.labels_csv else base_path / "labels.csv"
    if not labels_csv.exists():
        print(f"Error: labels.csv not found at {labels_csv}")
        return

    labels = read_labels_csv(labels_csv)
    labels_root = base_path / "labels"
    labels_root.mkdir(exist_ok=True)
    category_folders = {cat: (labels_root / cat) for cat in CATEGORIES}
    for folder in category_folders.values():
        folder.mkdir(exist_ok=True)

    episode_folders = sorted(
        d for d in base_path.iterdir() if d.is_dir() and d.name.startswith("episode_")
    )
    if not episode_folders:
        print(f"Error: No episode folders found in {base_path}")
        return

    print(f"Found {len(episode_folders)} episodes. Starting cleanup + split...\n")
    processed = skipped = 0
    for episode_path in episode_folders:
        print(f"Processing {episode_path.name}...")
        cleanup_episode(episode_path)
        episode_num = int(episode_path.name.split("_")[1])
        if episode_num not in labels:
            print("  No labels found for this episode, skipping label split.")
            skipped += 1
            continue
        process_episode(episode_path, episode_num, labels[episode_num], category_folders)
        processed += 1

    print("\nDone.")
    print(f"Processed episodes: {processed}")
    print(f"Skipped episodes (missing labels or colors): {skipped}")
    for category, folder in category_folders.items():
        count = len(list(folder.glob("*.jpg")))
        print(f"  {category}: {count} files")


if __name__ == "__main__":
    main()
