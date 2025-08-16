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
import time
from contextlib import nullcontext
from pprint import pformat
from typing import Any
import pickle
import torch
from termcolor import colored
from torch.amp import GradScaler
from torch.optim import Optimizer
import copy
from lerobot.common.datasets.factory import make_dataset
from lerobot.common.datasets.sampler import EpisodeAwareSampler
from lerobot.common.datasets.utils import cycle
from lerobot.common.envs.factory import make_env
from lerobot.common.optim.factory import make_optimizer_and_scheduler
from lerobot.common.policies.factory import make_policy
from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.policies.utils import get_device_from_parameters
from lerobot.common.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.common.utils.random_utils import set_seed
from lerobot.common.utils.train_utils import (
    get_step_checkpoint_dir,
    get_step_identifier,
    load_training_state,
    save_checkpoint,
    update_last_checkpoint,
)
from lerobot.common.utils.utils import (
    format_big_number,
    get_safe_torch_device,
    has_method,
    init_logging,
)
from lerobot.common.utils.wandb_utils import WandBLogger
from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from accelerate import Accelerator
from accelerate.utils import set_seed as accelerate_set_seed
import os
import numpy as np
import cv2
from lerobot.constants import (
    CHECKPOINTS_DIR,
    LAST_CHECKPOINT_LINK,
    PRETRAINED_MODEL_DIR,
    TRAINING_STATE_DIR,
    TRAINING_STEP,
)
from lerobot.common.policies.pi0.modeling_pi0 import PI0Policy
@parser.wrap()
def convert(cfg: TrainPipelineConfig):
    cfg.resume = True
    cfg.validate()
    logging.info(pformat(cfg.to_dict()))

   # Initialize accelerator
    from accelerate.utils import DistributedDataParallelKwargs

    from lerobot.common.utils.wandb_utils import cfg_to_group, get_wandb_run_id_from_filesystem

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator()

   # Setup device - accelerator handles device placement
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    # Create dataset
    if accelerator.is_main_process:
        logging.info("Creating dataset")
    dataset = make_dataset(cfg)

    if accelerator.is_main_process:
        logging.info("Creating policy")
    cfg.policy.device = "cpu"
    policy = make_policy(
        cfg=cfg.policy,
        ds_meta=dataset.meta,
    ).cpu()
    torch.cuda.empty_cache()
    
    checkpoint_path = cfg.output_dir / "checkpoints" / "last"
    meta_path = checkpoint_path / "meta.pkl"
    if torch.cuda.current_device() == 0:
        pickle.dump(dataset.meta, open(meta_path, 'wb'))
    # Prepare for distributed training
    '''
    policy = PI0Policy.from_pretrained(checkpoint_path / PRETRAINED_MODEL_DIR)
    policy = accelerator.prepare(policy)
    policy.save_checkpoint(checkpoint_path / PRETRAINED_MODEL_DIR)
    import ipdb;ipdb.set_trace()
    '''

    policy = accelerator.prepare(policy)
    policy.load_checkpoint(checkpoint_path / PRETRAINED_MODEL_DIR)
    policy = accelerator.unwrap_model(policy)
    policy.save_pretrained(checkpoint_path / "hf_model", save_function=accelerator.save, is_main_process=accelerator.is_main_process)

if __name__ == "__main__":
    init_logging()
    convert()
