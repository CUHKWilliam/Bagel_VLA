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
import logging
from pprint import pformat

import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.lerobot_dataset import (
    LeRobotDataset,
    LeRobotDatasetMetadata,
    MultiLeRobotDataset,
)
from lerobot.datasets.utils import get_delta_indices
from lerobot.datasets.transforms import ImageTransforms

from lerobot.datasets.vqa_dataset import(
    VQADataset,
    MultiVQADataset,
)
from lerobot.datasets.video_dataset import (
    VideoDataset,
    MultiVideoDataset,
)
import glob
import numpy as np
import copy
import pickle
import os
from torch.utils.data import ConcatDataset, Dataset

IMAGENET_STATS = {
    "mean": [[[0.485]], [[0.456]], [[0.406]]],  # (c,1,1)
    "std": [[[0.229]], [[0.224]], [[0.225]]],  # (c,1,1)
}


def resolve_delta_timestamps(
    cfg: PreTrainedConfig, ds_meta: LeRobotDatasetMetadata
) -> dict[str, list] | None:
    """Resolves delta_timestamps by reading from the 'delta_indices' properties of the PreTrainedConfig.

    Args:
        cfg (PreTrainedConfig): The PreTrainedConfig to read delta_indices from.
        ds_meta (LeRobotDatasetMetadata): The dataset from which features and fps are used to build
            delta_timestamps against.

    Returns:
        dict[str, list] | None: A dictionary of delta_timestamps, e.g.:
            {
                "observation.state": [-0.04, -0.02, 0]
                "observation.action": [-0.02, 0, 0.02]
            }
            returns `None` if the resulting dict is empty.
    """
    delta_timestamps = {}
    for key in ds_meta.features.keys():
        if key == "next.reward" and cfg.reward_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.reward_delta_indices]
        if "action" in key and cfg.action_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.action_delta_indices]
        if key.startswith("observation.") and cfg.observation_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.observation_delta_indices]
    if len(delta_timestamps) == 0:
        delta_timestamps = None

    return delta_timestamps


class ConcatDatasetWithIndex(Dataset):
    def __init__(self, inputs):
        self.ds = ConcatDataset(inputs)
        self.stats = None
        for inp in inputs:
            if hasattr(inp, "stats"):
                self.stats = inp.stats

    def __getitem__(self, index):
        while True:
            try:
                data = self.ds.__getitem__(index)
                break
            except:
                print(f'fail loading at {index}')
                index += 1
                continue
        data['data_index'] = index
        return data
    
    def __len__(self, ):
        return len(self.ds)

def make_dataset(cfg: TrainPipelineConfig, accelerator=None) -> LeRobotDataset | MultiLeRobotDataset:
    """Handles the logic of setting up delta timestamps and image transforms before creating a dataset.

    Args:
        cfg (TrainPipelineConfig): A TrainPipelineConfig config which contains a DatasetConfig and a PreTrainedConfig.

    Raises:
        NotImplementedError: The MultiLeRobotDataset is currently deactivated.

    Returns:
        LeRobotDataset | MultiLeRobotDataset
    """
    image_transforms = (
        ImageTransforms(cfg.dataset.image_transforms) if cfg.dataset.image_transforms.enable else None
    )
    all_datasets = []
    if cfg.dataset.repo_id is not None:
        if "," in cfg.dataset.repo_id:
            cfg.dataset.repo_id = cfg.dataset.repo_id.split(",")
        if isinstance(cfg.dataset.repo_id, str) and not "*" in cfg.dataset.repo_id:
            ds_meta = LeRobotDatasetMetadata(
                cfg.dataset.repo_id, root=cfg.dataset.root, revision=cfg.dataset.revision
            )
            delta_timestamps = resolve_delta_timestamps(cfg.policy, ds_meta)
            dataset = LeRobotDataset(
                cfg.dataset.repo_id,
                root=cfg.dataset.root,
                episodes=cfg.dataset.episodes,
                delta_timestamps=delta_timestamps,
                image_transforms=image_transforms,
                revision=cfg.dataset.revision,
                video_backend=cfg.dataset.video_backend,
                use_ref=cfg.policy.use_ref,
            )
            # if cfg.dataset.use_imagenet_stats:
            #     for key in dataset.meta.camera_keys:
            #         for stats_type, stats in IMAGENET_STATS.items():
            #             dataset.meta.stats[key][stats_type] = torch.tensor(stats, dtype=torch.float32)
        else:
            if isinstance(cfg.dataset.repo_id, str):
                cfg.dataset.repo_id = glob.glob(cfg.dataset.repo_id)
            elif isinstance(cfg.dataset.repo_id, list):
                repo_id2 = []
                for a_repo_id in cfg.dataset.repo_id:
                    if "*" in a_repo_id:
                        repo_id2 += glob.glob(a_repo_id)
                    else:
                        repo_id2.append(a_repo_id)
                cfg.dataset.repo_id = repo_id2
            dataset = MultiLeRobotDataset(
                cfg.dataset.repo_id,
                # TODO(aliberts): add proper support for multi dataset
                # delta_timestamps=delta_timestamps,
                image_transforms=image_transforms,
                video_backend=cfg.dataset.video_backend,
                episodes=cfg.dataset.episodes,
                use_ref=cfg.policy.use_ref,
            )
            for a_dataset in dataset._datasets:
                ds_meta = LeRobotDatasetMetadata(a_dataset.repo_id, root=a_dataset.root, revision=a_dataset.revision)
                delta_timestamps = resolve_delta_timestamps(cfg.policy, ds_meta)
                a_dataset.delta_timestamps = delta_timestamps
                a_dataset.delta_indices = get_delta_indices(a_dataset.delta_timestamps, a_dataset.fps)

            # dataset.meta = copy.deepcopy(dataset._datasets[0].meta)
            # if cfg.dataset.use_imagenet_stats:
            #     for a_dataset in dataset._datasets:
            #         for key in dataset.meta.camera_keys:
            #             for stats_type, stats in IMAGENET_STATS.items():
            #                 dataset.meta.stats[key][stats_type] = torch.tensor(stats, dtype=torch.float32)
        all_datasets.append(dataset)

    if cfg.dataset.vqa_repo_id is not None:
        if "," in cfg.dataset.vqa_repo_id and not "*" in cfg.dataset.vqa_repo_id:
            cfg.dataset.vqa_repo_id = cfg.dataset.vqa_repo_id.split(',')
        if isinstance(cfg.dataset.vqa_repo_id, str):
            dataset = VQADataset(
                cfg.dataset.vqa_repo_id,
                transform=image_transforms,
            )
        else:
            if isinstance(cfg.dataset.vqa_repo_id, str):
                cfg.dataset.vqa_repo_id = glob.glob(cfg.dataset.vqa_repo_id)
            elif isinstance(cfg.dataset.repo_id, list):
                repo_id2 = []
                for a_repo_id in cfg.dataset.vqa_repo_id:
                    repo_id2 += glob.glob(a_repo_id)
                cfg.dataset.vqa_repo_id = repo_id2
            
            dataset = MultiVQADataset(
                cfg.dataset.vqa_repo_id,
                transform=image_transforms,
            )
        all_datasets.append(dataset)
    if cfg.dataset.video_repo_id is not None:
        if "," in cfg.dataset.video_repo_id and not "*" in cfg.dataset.video_repo_id:
            cfg.dataset.video_repo_id = cfg.dataset.video_repo_id.split(',')
        if isinstance(cfg.dataset.video_repo_id, str):
            dataset = VideoDataset(
                cfg.dataset.video_repo_id,
                transform=image_transforms,
            )
        else:
            if isinstance(cfg.dataset.video_repo_id, str):
                cfg.dataset.video_repo_id = glob.glob(cfg.dataset.video_repo_id)
            elif isinstance(cfg.dataset.video_repo_id, list):
                repo_id2 = []
                for a_repo_id in cfg.dataset.video_repo_id:
                    repo_id2 += glob.glob(a_repo_id)
                cfg.dataset.video_repo_id = repo_id2
            
            dataset = MultiVideoDataset(
                cfg.dataset.video_repo_id,
                transform=image_transforms,
            )
        all_datasets.append(dataset)

    num_frames = 0
    num_episodes = 0
    for ds in all_datasets:
        num_frames += ds.num_frames
        num_episodes += ds.num_episodes
    dataset = ConcatDatasetWithIndex(all_datasets)
    # dataset = ConcatDataset(all_datasets)
    dataset.num_frames = num_frames
    dataset.num_episodes = num_episodes
    
    val_sample_weights_dict = {}
    dataset_types = []
    sample_weights = []
    for a_dataset in dataset.ds.datasets:
        sample_weights += [a_dataset.weight] * a_dataset.__len__()
        dataset_types += [a_dataset.ds_type] * a_dataset.__len__()
    sample_weights = np.array(sample_weights)
    dataset_types = np.array(dataset_types)
    all_dataset_types = np.unique(dataset_types)
    train_sample_weights = copy.deepcopy(sample_weights)
    for dataset_type in all_dataset_types:
        data_idx_same_t = np.where(dataset_types == dataset_type)[0]
        val_data_idx_same_t = np.random.choice(data_idx_same_t, size=cfg.val_sample_num)
        val_sample_weights_dict[dataset_type] = np.zeros_like(sample_weights)
        val_sample_weights_dict[dataset_type][val_data_idx_same_t] = train_sample_weights[val_data_idx_same_t]
        train_sample_weights[val_data_idx_same_t] = 0.0
    train_sample_weights = np.array(train_sample_weights)

    return dataset, train_sample_weights, val_sample_weights_dict
