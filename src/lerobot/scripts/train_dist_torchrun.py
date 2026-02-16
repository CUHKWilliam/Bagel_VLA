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

from tqdm import tqdm
import logging
import time
from contextlib import nullcontext
from pprint import pformat
from typing import Any
import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from termcolor import colored
from torch.amp import GradScaler
from torch.optim import Optimizer
import copy
import numpy as np
import cv2
import pickle
import wandb

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
from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from torch.utils.data import WeightedRandomSampler, DistributedSampler
from torch.utils.data.distributed import DistributedSampler
import deepspeed

wandb.login()

class CustomWeightedRandomSampler(WeightedRandomSampler):
    """WeightedRandomSampler except allows for more than 2^24 samples to be sampled"""
    def __init__(self, *args, **kwargs):
        self.world_size = kwargs.pop('world_size')
        self.rank = kwargs.pop('rank')
        super().__init__(*args, **kwargs)

    def __iter__(self):
        # Zero out weights for samples not assigned to this rank
        for idx in range(self.world_size):
            if idx != self.rank:
                self.weights[idx::self.world_size] = 0
        
        # Normalize weights for this rank
        rank_weights = self.weights / torch.sum(self.weights)
        
        # Sample only from this rank's portion
        rand_tensor = np.random.choice(
            range(0, len(self.weights)),
            size=self.num_samples,
            p=rank_weights.numpy(),
            replace=self.replacement
        )
        rand_tensor = torch.from_numpy(rand_tensor)
        return iter(rand_tensor.tolist())


def update_policy(
    train_metrics: MetricsTracker,
    policy: PreTrainedPolicy,
    batch: Any,
    step: int = 0,
    optimizer = None,
    model_engine = None 
) -> tuple[MetricsTracker, dict]:
    start_time = time.perf_counter()
    device = get_device_from_parameters(policy)

    policy.train()
    loss, output_dict = model_engine(batch)
    model_engine.backward(loss)
    model_engine.step()

    loss_value = loss.item()
    mse_loss_value = output_dict['mse'].item()
    ce_loss_value = output_dict['ce'].item()

    train_metrics.loss = loss_value
    train_metrics.ce = ce_loss_value
    train_metrics.mse = mse_loss_value
    train_metrics.lr = model_engine.get_lr()[0]
    train_metrics.update_s = time.perf_counter() - start_time
    return train_metrics, output_dict


def setup_distributed():
    """Initialize distributed training"""
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    if world_size > 1:
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(local_rank)
    
    return rank, local_rank, world_size


def cleanup_distributed():
    """Cleanup distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()


@parser.wrap()
def train(cfg: TrainPipelineConfig):
    cfg.type = "pi0"
    cfg.validate()
    
    # Setup distributed training
    rank, local_rank, world_size = setup_distributed()
    
    # Initialize logging only on main process
    if rank == 0:
        init_logging()
        logging.info(pformat(cfg.to_dict()))
    
    # Initialize wandb only on main process
    if rank == 0 and cfg.wandb.enable and cfg.wandb.project:
        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=cfg.job_name,
            notes=cfg.wandb.notes,
            config=cfg.to_dict(),
            dir=cfg.output_dir,
            resume="must" if cfg.resume else None,
            id=cfg.wandb.run_id if cfg.wandb.run_id else None,
        )
    else:
        wandb_logger = None
        if rank == 0:
            logging.info(colored("Logs will be saved locally.", "yellow", attrs=["bold"]))

    # Set seed for reproducibility
    seed = 0
    torch.manual_seed(seed )
    np.random.seed(seed )
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed )
    
    # Setup device
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    # Create dataset
    if rank == 0:
        logging.info("Creating dataset")
    
    all_datasets, train_sample_weights, val_sample_weights_dict, stats = make_dataset(cfg, dist.get_rank())
    
    # Create environment for evaluation (if needed)
    eval_envs = None
    if False:  # TODO: Add condition for evaluation
        pass  # Keep your existing evaluation setup
    
    if rank == 0:
        logging.info("Creating policy")
    
    cfg.policy.device = "cpu"
    policy = make_policy(
        cfg=cfg.policy,
        ds_stats=stats,
    ).to(device)

    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)


    ds_config = {
        "train_batch_size": cfg.batch_size * world_size,  # Global batch size
        "train_micro_batch_size_per_gpu": cfg.batch_size,
        "gradient_accumulation_steps": 1,
        "steps_per_print": cfg.log_freq if cfg.log_freq > 0 else 100,
        
        "zero_optimization": {
            "stage": 2,  # Enable ZeRO-2 optimization
            "allgather_partitions": True,
            "allgather_bucket_size": 2e8,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 2e8,
            "contiguous_gradients": True,
            "cpu_offload": False,  # Set to True if you want CPU offloading
        },
        
        "fp16": {
            "enabled": False,
            "loss_scale": 0,
            "loss_scale_window": 1000,
            "initial_scale_power": 16,
            "hysteresis": 2,
            "min_loss_scale": 1
        },
        "bf16": {
            "enabled": True
        },
        
        "gradient_clipping": cfg.clip_grad_norm if hasattr(cfg, 'clip_grad_norm') else 1.0,
        "wall_clock_breakdown": False,
        
        "optimizer": {
            "type": "AdamW",
            "params": {
            "lr": 1e-4,
            "weight_decay": 1e-10,
            "eps": 1e-10,
            "betas": [0.95, 0.99]
            } 
        },
        "scheduler": {
            "type": "WarmupDecayLR",
            "params": {
                "warmup_min_lr": 1e-5,
                "warmup_max_lr": 1e-5,
                "warmup_num_steps": 200,
                "total_num_steps": 300000
            }
        },
        
        "flops_profiler": {
            "enabled": False,
            "profile_step": 1,
            "module_depth": -1,
            "top_modules": 1,
            "detailed": True,
        },
        
        "checkpoint": {
            "use_node_local_storage": True,
            "save_universal_checkpoint_format": True,
        }
    }
    
    
    model_engine, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=policy,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        config=ds_config,
        model_parameters=policy.parameters(),
        dist_init_required=False  # Already initialized by torchrun
    )
    
    torch.cuda.empty_cache()
    
    if rank == 0:
        logging.info("Creating optimizer and scheduler")
    
    step = 0
    tokens = 0
    onestep = 0
    train_sample_seen = np.zeros_like(train_sample_weights).astype(np.float32)
    
    # Create dataloader
    if hasattr(cfg.policy, "drop_n_last_frames"):
        shuffle = False
        sampler = EpisodeAwareSampler(
            dataset.episode_data_index,
            drop_n_last_frames=cfg.policy.drop_n_last_frames,
            shuffle=True,
        )
    else:
        train_sampler = CustomWeightedRandomSampler(
            weights=train_sample_weights, 
            num_samples=len(train_sample_weights),
            world_size=world_size,
            rank=rank
        )
    
    dataloaders = []
    for i, dataset in enumerate(all_datasets):
        
        dataloader = torch.utils.data.DataLoader(
            dataset,
            num_workers=4,
            batch_size=1,
            # sampler=sampler,
            drop_last=False,
            pin_memory=True,
        )
        dataloaders.append(dataloader)
    
    print(f"Rank {rank}: number of dataloader:", len(dataloaders))
    
    
    def get_model_param_count(model, trainable_only=False):
        def numel(p):
            try:
                return p.ds_numel
            except:
                return p.numel()
        return sum(numel(p) for p in model.parameters() if not trainable_only or p.requires_grad)
    
    # Log training info (only on main process)
    # Log training info (only on main process)
    num_learnable_params = get_model_param_count(policy, trainable_only=True)
    num_total_params = get_model_param_count(policy, trainable_only=False)

    logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")
    if cfg.env is not None:
        logging.info(f"{cfg.env.task=}")
    logging.info(f"{cfg.steps=} ({format_big_number(cfg.steps)})")
    num_datasets = len(all_datasets)
    logging.info("============ Dataset Recipe =================")
    num_frames = 0
    num_episodes = 0
    for ds in all_datasets:
        if hasattr(ds, "repo_id"):
            logging.info(f"{ds.repo_id=}: {ds.num_frames=} ({format_big_number(ds.num_frames)}) {ds.num_episodes=} ({format_big_number(ds.num_episodes)}) {ds.weight=} {ds.ds_type=}")
        else:
            logging.info(f"{ds.repo_ids=}: {ds.num_frames=} ({format_big_number(ds.num_frames)}) {ds.num_episodes=} ({format_big_number(ds.num_episodes)}) {ds.weight=} {ds.ds_type=}")
        num_frames += ds.num_frames
        num_episodes += ds.num_episodes
    logging.info("============ Dataset Recipe End =================")
    logging.info(f"{num_learnable_params=} ({format_big_number(num_learnable_params)})")
    logging.info(f"{num_total_params=} ({format_big_number(num_total_params)})")
    logging.info(f"Number of processes: {world_size}")
    logging.info(f"Device: {device}")
    
    train_metrics = {
        "loss": AverageMeter("loss", ":.3f"),
        "ce": AverageMeter("ce", ":.3f"),
        "mse": AverageMeter("mse", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        "update_s": AverageMeter("updt_s", ":.3f"),
        "dataloading_s": AverageMeter("data_s", ":.3f"),
    }
    
    train_tracker = MetricsTracker(
        num_frames, num_episodes, train_metrics, initial_step=step,
    )
    
    policy.train()
    
    if rank == 0:
        logging.info("Start offline training on a fixed dataset")
    
    # Create iterator from dataloader
    seq_dataloader = policy.dataset(dataloaders, policy.tokenize_action)
    

    flag_tokens_full = True
    for _ in range(step, cfg.steps):
        start_time = time.perf_counter()
        data_batch = next(seq_dataloader)
        
        # Move data to device
        data_batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in data_batch.items()}
        
        
        train_tracker.dataloading_s = time.perf_counter() - start_time

        train_tracker, output_dict = update_policy(
            train_tracker,
            policy,
            data_batch,
            step,
            optimizer=optimizer,
            model_engine=model_engine,
        )
        
        
        step += len(data_batch.get('sample_lens', [1]))
        onestep += 1
        num_tokens = data_batch.get('sequence_length', 1)
        tokens += num_tokens
        num_tokens, step = train_tracker.step(num_tokens, add_steps=len(data_batch.get('sample_lens', [1])))

        is_log_step = cfg.log_freq > 0 and step % cfg.log_freq == 0
        is_saving_step = onestep % cfg.save_freq == 0 or onestep == cfg.steps
        is_eval_step = cfg.eval_freq > 0 and step % cfg.eval_freq == 0
        is_eval_step = False

        if cfg.save_checkpoint and is_saving_step and rank == 0:
            logging.info(f"Checkpoint policy after step {step}")
            checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, cfg.steps, step)
            
            save_checkpoint(
                checkpoint_dir, step, tokens, cfg, policy, 
                None, None, train_sample_weights, val_sample_weights_dict, train_sample_seen
            )
            update_last_checkpoint(checkpoint_dir)
            pickle.dump(
                policy.dataset_stats, 
                open(os.path.join(checkpoint_dir, "dataset_stats.pkl"), 'wb')
            )
        
        if is_log_step and rank == 0:
            logging.info(train_tracker)
            if cfg.wandb.enable:
                wandb_log_dict = train_tracker.to_dict()
                wandb.log(wandb_log_dict, step=step)
            train_tracker.reset_averages()
  
    
    # Cleanup distributed training
    cleanup_distributed()
    
    if rank == 0:
        logging.info("End of training")


if __name__ == "__main__":
    train()