      
#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Threaded version of MultiLeRobotDataset for parallel dataset loading.

This module provides MultiLeRobotDatasetThreaded which is a drop-in replacement
for MultiLeRobotDataset but with multi-threaded loading for improved performance
when loading many datasets.

Usage:
    # In factory.py, replace:
    from lerobot.datasets.lerobot_dataset import MultiLeRobotDataset

    # With:
    from lerobot.datasets.lerobot_dataset_v2_threaded import MultiLeRobotDatasetThreaded as MultiLeRobotDataset
"""
from __future__ import annotations

import logging
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Any

import datasets
import numpy as np
import torch
import torch.utils

from lerobot.constants import HF_LEROBOT_HOME
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.video_utils import VideoFrame, get_safe_default_codec


logger = logging.getLogger(__name__)


class MultiLeRobotDatasetThreaded(torch.utils.data.Dataset):
    """Multi-dataset variant of LeRobotDataset with threaded dataset construction.

    This class is a drop-in replacement for MultiLeRobotDataset that loads datasets
    in parallel using multiple threads, significantly reducing initialization time
    when working with many datasets.

    The underlying `LeRobotDataset`s are effectively concatenated, and this class
    adopts much of the API structure of `LeRobotDataset`.

    Args:
        repo_ids: List of dataset repository IDs to load.
        root: Root directory for dataset storage. Defaults to HF_LEROBOT_HOME.
        episodes: Optional dict mapping repo_id to list of episode indices to load.
        image_transforms: Optional image transformation function.
        delta_timestamps: Optional dict of delta timestamps for each feature.
        tolerances_s: Optional dict mapping repo_id to tolerance in seconds.
        download_videos: Whether to download video files. Defaults to True.
        video_backend: Video backend to use for decoding. Defaults to auto-detect.
        use_ref: Whether to use reference frames. Defaults to True.
        max_workers: Maximum number of threads for parallel loading.
                     Defaults to min(8, len(repo_ids)).
        show_progress: Whether to log loading progress. Defaults to True.

    Example:
        >>> dataset = MultiLeRobotDatasetThreaded(
        ...     repo_ids=["dataset1", "dataset2", "dataset3"],
        ...     max_workers=4,
        ...     show_progress=True,
        ... )
        >>> print(f"Loaded {dataset.num_episodes} episodes")
    """

    weight = 1.0
    ds_type = "action"

    def __init__(
        self,
        repo_ids: list[str],
        root: str | Path | None = None,
        episodes: dict | None = None,
        image_transforms: Callable | None = None,
        delta_timestamps: dict[list[float]] | None = None,
        tolerances_s: dict | None = None,
        download_videos: bool = True,
        video_backend: str | None = None,
        use_ref: bool = True,
        max_workers: int | None = None,
        show_progress: bool = True,
    ):
        super().__init__()
        self.repo_ids = repo_ids
        self.root = Path(root) if root else HF_LEROBOT_HOME
        self.tolerances_s = tolerances_s if tolerances_s else dict.fromkeys(repo_ids, 0.0001)
        self.image_transforms = image_transforms
        self.delta_timestamps = delta_timestamps
        self.video_backend = video_backend if video_backend is not None else get_safe_default_codec()
        self._show_progress = show_progress

        # Store parameters for dataset construction
        self._episodes = episodes
        self._download_videos = download_videos
        self._use_ref = use_ref

        # Load datasets with threading
        self._datasets = self._load_datasets_threaded(max_workers)

        if len(self._datasets) == 0:
            raise RuntimeError(
                f"Failed to load any datasets from {len(repo_ids)} provided repo_ids. "
                "Check the logs for details on individual failures."
            )

        # Disable any data keys that are not common across all datasets
        self.disabled_features = set()
        # Note: Feature intersection check is commented out in original code
        # to allow heterogeneous datasets

        # Aggregate stats across datasets
        self.stats = self._aggregate_stats()

    def _load_datasets_threaded(self, max_workers: int | None = None) -> list[LeRobotDataset]:
        """Load all datasets using multiple threads.

        Args:
            max_workers: Maximum number of worker threads. If None, uses min(8, len(repo_ids)).

        Returns:
            List of successfully loaded LeRobotDataset instances.
        """
        num_repos = len(self.repo_ids)
        if num_repos == 0:
            return []

        workers = max_workers if max_workers is not None else min(8, num_repos)

        # For single dataset or single worker, use sequential loading
        if num_repos <= 1 or workers <= 1:
            return self._load_datasets_sequential()

        datasets_list: list[LeRobotDataset] = []
        failed_repos: list[tuple[str, str]] = []

        # Thread-safe counters for progress tracking
        lock = Lock()
        completed_count = [0]

        start_time = time.time()
        if self._show_progress:
            logger.info(f"Starting parallel load of {num_repos} datasets with {workers} workers...")

        def load_single_dataset(repo_id: str) -> tuple[str, LeRobotDataset | None, str | None]:
            """Load a single dataset and return (repo_id, dataset, error_message)."""
            try:
                ds = LeRobotDataset(
                    repo_id,
                    root=self.root / repo_id,
                    episodes=self._episodes[repo_id] if self._episodes else None,
                    image_transforms=self.image_transforms,
                    delta_timestamps=self.delta_timestamps,
                    tolerance_s=self.tolerances_s[repo_id],
                    download_videos=self._download_videos,
                    video_backend=self.video_backend,
                    use_ref=self._use_ref,
                )

                # Update progress counter
                with lock:
                    completed_count[0] += 1
                    if self._show_progress:
                        logger.info(
                            f"[{completed_count[0]}/{num_repos}] Loaded: {repo_id} "
                            f"({ds.num_episodes} episodes, {ds.num_frames} frames)"
                        )

                return (repo_id, ds, None)
            except Exception as e:
                with lock:
                    completed_count[0] += 1
                    if self._show_progress:
                        logger.warning(f"[{completed_count[0]}/{num_repos}] Failed to load: {repo_id}")

                return (repo_id, None, str(e))

        # Use ThreadPoolExecutor for parallel loading
        with ThreadPoolExecutor(max_workers=workers) as executor:
            # Submit all tasks
            future_to_repo = {
                executor.submit(load_single_dataset, repo_id): repo_id
                for repo_id in self.repo_ids
            }

            # Collect results as they complete
            for future in as_completed(future_to_repo):
                repo_id, ds, error = future.result()
                if ds is not None:
                    datasets_list.append(ds)
                else:
                    failed_repos.append((repo_id, error))

        elapsed_time = time.time() - start_time

        # Log summary
        if self._show_progress:
            logger.info(
                f"Dataset loading complete in {elapsed_time:.2f}s: "
                f"{len(datasets_list)} succeeded, {len(failed_repos)} failed"
            )

        if failed_repos:
            for repo_id, error in failed_repos:
                logger.warning(f"Failed to load {repo_id}: {error}")

        # Sort datasets by their original order in repo_ids for deterministic behavior
        repo_id_order = {repo_id: idx for idx, repo_id in enumerate(self.repo_ids)}
        datasets_list.sort(key=lambda ds: repo_id_order.get(ds.repo_id, float('inf')))

        return datasets_list

    def _load_datasets_sequential(self) -> list[LeRobotDataset]:
        """Load datasets sequentially (fallback for single dataset or single worker)."""
        datasets_list = []
        for repo_id in self.repo_ids:
            try:
                ds = LeRobotDataset(
                    repo_id,
                    root=self.root / repo_id,
                    episodes=self._episodes[repo_id] if self._episodes else None,
                    image_transforms=self.image_transforms,
                    delta_timestamps=self.delta_timestamps,
                    tolerance_s=self.tolerances_s[repo_id],
                    download_videos=self._download_videos,
                    video_backend=self.video_backend,
                    use_ref=self._use_ref,
                )
                datasets_list.append(ds)
                if self._show_progress:
                    logger.info(f"Loaded: {repo_id} ({ds.num_episodes} episodes)")
            except Exception as e:
                logger.warning(f"Failed to load {repo_id}: {e}")
                continue
        return datasets_list

    def _aggregate_stats(self) -> dict[str, Any]:
        """Aggregate statistics across all loaded datasets.

        This method handles various action formats from different robot types:
        - Standard 'action' key
        - Galaxea format: action.left_arm, action.left_gripper, etc.
        - Agibot format: actions.end.position, actions.end.orientation, etc.

        Returns:
            Aggregated statistics dictionary with normalized action stats.
        """
        aggregated_stats: dict[str, Any] = {}

        for dataset in self._datasets:
            meta = dataset.meta
            stats = meta.stats.copy()  # Make a copy to avoid modifying original

            if "action" not in stats:
                # Handle Galaxea format
                if "action.left_gripper" in stats and "action.left_arm" in stats:
                    left_action_min = np.concatenate(
                        [stats["action.left_arm"]["min"], stats["action.left_gripper"]["min"]],
                        axis=-1
                    )
                    right_action_min = np.concatenate(
                        [stats["action.right_arm"]["min"], stats["action.right_gripper"]["min"]],
                        axis=-1
                    )
                    action_min = np.concatenate([left_action_min, right_action_min], axis=-1)

                    left_action_max = np.concatenate(
                        [stats["action.left_arm"]["max"], stats["action.left_gripper"]["max"]],
                        axis=-1
                    )
                    right_action_max = np.concatenate(
                        [stats["action.right_arm"]["max"], stats["action.right_gripper"]["max"]],
                        axis=-1
                    )
                    action_max = np.concatenate([left_action_max, right_action_max], axis=-1)
                    stats["action"] = {"max": action_max, "min": action_min}

                # Handle Agibot format
                elif "actions.end.position" in stats:
                    left_action_min = np.concatenate(
                        [
                            stats["actions.end.position"]["min"][0, :],
                            stats["actions.end.orientation"]["min"][0, :],
                            stats["actions.effector.position"]["min"][0, None],
                        ],
                        axis=-1,
                    )
                    right_action_min = np.concatenate(
                        [
                            stats["actions.end.position"]["min"][1, :],
                            stats["actions.end.orientation"]["min"][1, :],
                            stats["actions.effector.position"]["min"][1, None],
                        ],
                        axis=-1,
                    )
                    action_min = np.concatenate([left_action_min, right_action_min], axis=-1)

                    left_action_max = np.concatenate(
                        [
                            stats["actions.end.position"]["max"][0, :],
                            stats["actions.end.orientation"]["max"][0, :],
                            stats["actions.effector.position"]["max"][0, None],
                        ],
                        axis=-1,
                    )
                    right_action_max = np.concatenate(
                        [
                            stats["actions.end.position"]["max"][1, :],
                            stats["actions.end.orientation"]["max"][1, :],
                            stats["actions.effector.position"]["max"][1, None],
                        ],
                        axis=-1,
                    )
                    action_max = np.concatenate([left_action_max, right_action_max], axis=-1)
                    stats["action"] = {"max": action_max, "min": action_min}

            # Pad action stats to fixed size (32)
            if "action" in stats:
                padded_min = np.zeros((32,))
                padded_min[: len(stats["action"]["min"])] = stats["action"]["min"]
                stats["action"]["min"] = padded_min

                padded_max = np.zeros((32,))
                padded_max[: len(stats["action"]["max"])] = stats["action"]["max"]
                stats["action"]["max"] = padded_max

                # Aggregate across datasets
                if "action" not in aggregated_stats:
                    aggregated_stats["action"] = {
                        "min": stats["action"]["min"].copy(),
                        "max": stats["action"]["max"].copy(),
                    }
                else:
                    aggregated_stats["action"]["min"] = np.minimum(
                        aggregated_stats["action"]["min"],
                        stats["action"]["min"]
                    )
                    aggregated_stats["action"]["max"] = np.maximum(
                        aggregated_stats["action"]["max"],
                        stats["action"]["max"]
                    )

        return aggregated_stats

    @property
    def repo_id_to_index(self) -> dict[str, int]:
        """Return a mapping from dataset repo_id to dataset index."""
        return {repo_id: i for i, repo_id in enumerate(self.repo_ids)}

    @property
    def repo_index_to_id(self) -> dict[int, str]:
        """Return the inverse mapping of repo_id_to_index."""
        return {v: k for k, v in self.repo_id_to_index.items()}

    @property
    def fps(self) -> int:
        """Frames per second used during data collection."""
        return self._datasets[0].meta.info["fps"]

    @property
    def video(self) -> bool:
        """Returns True if this dataset loads video frames from mp4 files."""
        return self._datasets[0].meta.info.get("video", False)

    @property
    def features(self) -> datasets.Features:
        """Combined features from all datasets."""
        features = {}
        for dataset in self._datasets:
            features.update(
                {k: v for k, v in dataset.hf_features.items() if k not in self.disabled_features}
            )
        return features

    @property
    def camera_keys(self) -> list[str]:
        """Keys to access image and video stream from cameras."""
        keys = []
        for key, feats in self.features.items():
            if isinstance(feats, (datasets.Image, VideoFrame)):
                keys.append(key)
        return keys

    @property
    def video_frame_keys(self) -> list[str]:
        """Keys to access video frames that require decoding into images."""
        video_frame_keys = []
        for key, feats in self.features.items():
            if isinstance(feats, VideoFrame):
                video_frame_keys.append(key)
        return video_frame_keys

    @property
    def num_frames(self) -> int:
        """Total number of samples/frames across all datasets."""
        return sum(d.num_frames for d in self._datasets)

    @property
    def num_episodes(self) -> int:
        """Total number of episodes across all datasets."""
        return sum(d.num_episodes for d in self._datasets)

    @property
    def tolerance_s(self) -> float:
        """Tolerance in seconds for timestamp matching."""
        return 1 / self.fps - 1e-4

    def __len__(self) -> int:
        return self.num_frames

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Get a sample from a randomly selected dataset.

        Note: The idx is used as a random seed for reproducibility.
        """
        np.random.seed(idx)
        dataset = self._datasets[np.random.choice(np.arange(len(self._datasets)))]
        item = dataset[int(np.random.choice(np.arange(len(dataset))))]
        item["dataset_index"] = torch.tensor(0)  # TODO: implement proper dataset indexing

        for data_key in self.disabled_features:
            if data_key in item:
                del item[data_key]

        return item

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(\n"
            f"  Repository IDs: {self.repo_ids},\n"
            f"  Loaded Datasets: {len(self._datasets)}/{len(self.repo_ids)},\n"
            f"  Number of Samples: {self.num_frames},\n"
            f"  Number of Episodes: {self.num_episodes},\n"
            f"  Type: {'video (.mp4)' if self.video else 'image (.png)'},\n"
            f"  Recorded Frames per Second: {self.fps},\n"
            f"  Camera Keys: {self.camera_keys},\n"
            f"  Video Frame Keys: {self.video_frame_keys if self.video else 'N/A'},\n"
            f"  Transformations: {self.image_transforms},\n"
            f")"
        )


# Alias for backward compatibility and easy drop-in replacement
MultiLeRobotDataset = MultiLeRobotDatasetThreaded

    