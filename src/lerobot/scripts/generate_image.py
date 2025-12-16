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

# import swanlab
# swanlab.sync_wandb()

from tqdm import tqdm
import logging
import time
from contextlib import nullcontext
from pprint import pformat
from typing import Any

import torch
from termcolor import colored
from torch.amp import GradScaler
from torch.optim import Optimizer
import copy
from lerobot.datasets.factory import make_dataset
from lerobot.datasets.sampler import EpisodeAwareSampler
from lerobot.datasets.utils import cycle
from lerobot.envs.factory import make_env
from lerobot.optim.factory import make_optimizer_and_scheduler
from lerobot.policies.factory import make_policy
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import get_device_from_parameters
from lerobot.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.utils.random_utils import set_seed
from lerobot.utils.train_utils import (
    get_step_checkpoint_dir,
    get_step_identifier,
    load_training_state,
    save_checkpoint,
    update_last_checkpoint,
)
from lerobot.utils.utils import (
    format_big_number,
    get_safe_torch_device,
    has_method,
    init_logging,
)
from lerobot.utils.wandb_utils import WandBLogger
from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
import os
import numpy as np
import cv2
from lerobot.configs.train import TrainPipelineConfig
from torch.utils.data import WeightedRandomSampler
import pickle
import multiprocessing

@parser.wrap()
def predict_images(cfg: TrainPipelineConfig):
    cfg.type = "pi0"
    cfg.validate()
    save_path = "generated_images"
    os.makedirs(save_path, exist_ok=True)
    dataset, train_sample_weights, val_sample_weights_dict = make_dataset(cfg)
    
    # Create environment used for evaluating checkpoints during training on simulation data.
    # On real-world data, no need to create an environment as evaluations are done outside train.py,
    # using the eval.py instead, with gym_dora environment and dora-rs.
   
    cfg.policy.device = "cpu"
    policy = make_policy(
        cfg=cfg.policy,
    ).cuda()
    torch.cuda.empty_cache()

    checkpoint_path = cfg.output_dir / "checkpoints" / "last"
    _, _, _, _, train_sample_weights, val_sample_weights_dict, train_sample_seen = load_training_state(checkpoint_path, None, None)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=0, # multiprocessing.cpu_count(), # cfg.num_workers, ## TODO: set worker
        batch_size=1,
        # shuffle=shuffle,
        pin_memory=False,
        drop_last=False,
    )
    flag_tokens_full = True
    data_iter = iter(dataloader)
    for idx in range(100):
        start_time = time.perf_counter()
        data_batch = next(data_iter)
        for k in data_batch.keys():
            if isinstance(data_batch[k], torch.Tensor):
                data_batch[k] = data_batch[k].cuda()
        generated_image, current_image = policy.generate_image(data_batch)
        generated_image.save(os.path.join(save_path, f"{idx}_next_frame.png"))
        current_image.save(os.path.join(save_path, f"{idx}_current_frame.png"))
        import ipdb;ipdb.set_trace()


if __name__ == "__main__":
    init_logging()
    predict_images()
