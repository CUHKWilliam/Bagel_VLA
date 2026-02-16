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
import random
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
from accelerate import Accelerator
from accelerate.utils import set_seed as accelerate_set_seed
import os
import numpy as np
import cv2
from lerobot.configs.train import TrainPipelineConfig
from torch.utils.data import WeightedRandomSampler
import pickle
import multiprocessing
import wandb
import pickle
import torch.distributed as dist
from transformers import AutoTokenizer, AutoProcessor

# import pdb; pdb.set_trace()
import os
os.environ['NCCL_DEBUG'] = 'INFO'
os.environ['NCCL_DEBUG_SUBSYS'] = 'ALL'
os.environ['NCCL_ASYNC_ERROR_HANDLING'] = '1'
os.environ['NCCL_BLOCKING_WAIT'] = '1'
wandb.login()


@parser.wrap()
def train(cfg: TrainPipelineConfig):
    # Initialize accelerator
    from accelerate.utils import DistributedDataParallelKwargs

    from lerobot.utils.wandb_utils import cfg_to_group, get_wandb_run_id_from_filesystem

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        # mixed_precision="no",
        gradient_accumulation_steps=1,
        # log_with="wandb" if cfg.wandb.enable else None,
        kwargs_handlers=[ddp_kwargs],
        # project_dir=cfg.output_dir,
    )


   
    def seed_worker(worker_id):
        """Set seed for each worker"""
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)


    # Setup device - accelerator handles device placement
    torch.backends.cuda.matmul.allow_tf32 = True
 
    all_datasets, train_sample_weights, val_sample_weights_dict, stats = make_dataset(cfg, accelerator.process_index)


    dataloaders = [torch.utils.data.DataLoader(
        dataset,
        num_workers=4, # multiprocessing.cpu_count(), # cfg.num_workers, ## TODO: set worker
        batch_size=1,
        shuffle=False,
        drop_last=False,
        worker_init_fn=seed_worker,
    ) for dataset in all_datasets]
    print("number of dataloader:", len(dataloaders))


    print("collect action_data...")
    actions = []
    dataloader_iters = [iter(dataloader) for dataloader in dataloaders]
    fast_tokenizer_path = "/dataset_rc_mm/tangwl3@xiaopeng.com/fast_tokenizer/fast_tokenizer"
    tokenizer = AutoProcessor.from_pretrained(fast_tokenizer_path, trust_remote_code=True, )

    for idx in tqdm(range(40000)):
        dataloader_iter = np.random.choice(dataloader_iters)
        data_batch = next(dataloader_iter)
        action =  data_batch['action'][0]
        action_pad = torch.concatenate([action, torch.zeros((action.size(0), 50 - action.size(1)))], dim=-1)
        actions.append(action_pad)

    actions = torch.stack(actions, dim=0)
    actions -= torch.from_numpy(stats['action']['min'])
    actions /= torch.from_numpy(stats['action']['max']- stats['action']['min']) + 1e-6
    actions = actions * 2 - 1
    actions = torch.clamp(actions, -1, 1)
    tokenizer = tokenizer.fit(actions)
    tokenizer.save_pretrained("outputs/gr00t_tokenizer")
if __name__ == "__main__":
    init_logging()
    train()