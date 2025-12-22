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
wandb.login()

class CustomWeightedRandomSampler(WeightedRandomSampler):
    """WeightedRandomSampler except allows for more than 2^24 samples to be sampled"""
    def __init__(self, *args, **kwargs):
        self.accelerator = kwargs.pop('accelerator')
        super().__init__(*args, **kwargs)
        self.rand_tensor = None

    def __iter__(self):
        for idx in range(self.accelerator.num_processes):
            if idx != self.accelerator.process_index:
                self.weights[self.accelerator.process_index::self.accelerator.num_processes] = 0
        rand_tensor = np.random.choice(range(0, len(self.weights)),
                                       size=self.num_samples,
                                       p=self.weights.numpy() / torch.sum(self.weights).numpy(),
                                       replace=self.replacement)
        rand_tensor = torch.from_numpy(rand_tensor)
        self.rand_tensor = rand_tensor
        return iter(rand_tensor.tolist())


def update_policy(
    train_metrics: MetricsTracker,
    policy: PreTrainedPolicy,
    batch: Any,
    accelerator: Accelerator,
    step: int = 0,
) -> tuple[MetricsTracker, dict]:
    start_time = time.perf_counter()
    device = get_device_from_parameters(policy)

    policy.train()
    loss, output_dict = policy.forward(batch)
    # policy.select_action(batch)
    policy.backward(loss)
    policy.step()
    # Gather metrics across all processes
    loss_value = accelerator.gather(loss.detach()).mean().item()
    mse = output_dict['mse']
    ce = output_dict['ce']
    mse_loss_value = mse.detach().mean().item()
    ce_loss_value = ce.detach().mean().item()
    # mse_loss_value = accelerator.gather(mse.detach()).mean().item()
    # ce_loss_value = accelerator.gather(ce.detach()).mean().item()

    # grad_norm_value = accelerator.gather(grad_norm).mean().item()

    train_metrics.loss = loss.item()
    train_metrics.ce = ce.item()
    train_metrics.mse = mse.item()
    # train_metrics.grad_norm = grad_norm.item()
    train_metrics.lr = policy.get_lr()[0]
    train_metrics.update_s = time.perf_counter() - start_time
    return train_metrics, output_dict


@parser.wrap()
def train(cfg: TrainPipelineConfig):
    cfg.type = "pi0"
    cfg.validate()
    logging.info(pformat(cfg.to_dict()))

    if cfg.seed is not None:
        set_seed(cfg.seed)
    
    # Initialize accelerator
    from accelerate.utils import DistributedDataParallelKwargs

    from lerobot.utils.wandb_utils import cfg_to_group, get_wandb_run_id_from_filesystem

    # ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        # mixed_precision="no",
        # gradient_accumulation_steps=1,
        # log_with="wandb" if cfg.wandb.enable else None,
        # kwargs_handlers=[ddp_kwargs],
        # project_dir=cfg.output_dir,
    )
    if cfg.wandb.enable and cfg.wandb.project:
        wandb_logger = WandBLogger(cfg, accelerator)
    else:
        wandb_logger = None
        logging.info(colored("Logs will be saved locally.", "yellow", attrs=["bold"]))

    '''
    accelerator.init_trackers(
        project_name=cfg.wandb.project,
        init_kwargs={
            "wandb": {
                "entity": cfg.wandb.entity,
                "name": cfg.job_name,
                "notes": cfg.wandb.notes,
                "tags": cfg_to_group(cfg, return_list=True),
                "dir": cfg.output_dir,
                "config": cfg.to_dict(),
                "save_code": False,
                "job_type": "train_eval",
                "mode": cfg.wandb.mode if cfg.wandb.mode in ["online", "offline", "disabled"] else "online",
                "resume": "must" if cfg.resume else None,
                "id": cfg.wandb.run_id
                if cfg.wandb.run_id
                else (get_wandb_run_id_from_filesystem(cfg.output_dir) if cfg.resume else None),
            }
        },
    )
    '''

    # Set seed for reproducibility
    if cfg.seed is not None:
        accelerate_set_seed(cfg.seed)

    # Setup device - accelerator handles device placement
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    # Create dataset
    if accelerator.is_main_process:
        logging.info("Creating dataset")
    dataset, train_sample_weights, val_sample_weights_dict = make_dataset(cfg, accelerator)
     
    # Create environment used for evaluating checkpoints during training on simulation data.
    # On real-world data, no need to create an environment as evaluations are done outside train.py,
    # using the eval.py instead, with gym_dora environment and dora-rs.
    eval_envs = None
    if False:
    # if cfg.eval_freq > 0: ## TODO:
        logging.info("Creating libero env")
        from libero.libero import benchmark
        from libero.libero.envs import OffScreenRenderEnv
        from libero.libero import get_libero_path
        benchmark_dict = benchmark.get_benchmark_dict()
        task_suite_name = "libero_90" # can also choose libero_spatial, libero_object, etc.
        task_suite = benchmark_dict[task_suite_name]()
        task_ids = list(range(200,))
        eval_envs = []
        for task_id in task_ids:
            task = task_suite.get_task(task_id)
            task_name = task.name
            ## TODO: just for debug now
            print(task_name)
            if task_name != "KITCHEN_SCENE1_open_the_bottom_drawer_of_the_cabinet":
                continue
            task_description = task.language
            task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
            print(f"[info] retrieving task {task_id} from suite {task_suite_name}, the " + \
                f"language instruction is {task_description}, and the bddl file is {task_bddl_file}")
            
            # step over the environment
            env_args = {
                "bddl_file_name": task_bddl_file,
                "camera_heights": 256,
                "camera_widths": 256
            }
            for _ in range(2):
                env = OffScreenRenderEnv(**env_args)
                env.seed(0)
                env.reset()
                eval_envs.append(env)
            break
        
    if accelerator.is_main_process:
        logging.info("Creating policy")
    cfg.policy.device = "cpu"
    policy = make_policy(
        cfg=cfg.policy,
        # ds_meta=dataset.meta,
    ).cpu()
    torch.cuda.empty_cache()
    if accelerator.is_main_process:
        logging.info("Creating optimizer and scheduler")
    # optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)

    step = 0  # number of policy updates (forward + backward + optim)
    tokens = 0
    onestep = 0
    train_sample_seen = np.zeros_like(train_sample_weights).astype(np.float32)
    # create dataloader for offline training
    if hasattr(cfg.policy, "drop_n_last_frames"):
        shuffle = False
        sampler = EpisodeAwareSampler(
            dataset.episode_data_index,
            drop_n_last_frames=cfg.policy.drop_n_last_frames,
            shuffle=True,
        )
    else:
        shuffle = True
        sampler = None
    if cfg.resume:
        checkpoint_path = cfg.output_dir / "checkpoints" / "last"
        step, tokens, _, _, train_sample_weights, val_sample_weights_dict, train_sample_seen = load_training_state(checkpoint_path, None, None)
    train_sampler = CustomWeightedRandomSampler(weights=train_sample_weights, num_samples=len(train_sample_weights), accelerator=accelerator)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=8, # multiprocessing.cpu_count(), # cfg.num_workers, ## TODO: set worker
        batch_size=1,
        shuffle=False,
        # sampler=train_sampler,
        pin_memory=True,
        drop_last=False,
    )
    def get_model_param_count(model, trainable_only=False):
        def numel(p):
            try:
                return p.ds_numel
            except:
                return p.numel()
        return sum(numel(p) for p in model.parameters() if not trainable_only or p.requires_grad)
    
    # Prepare for distributed training
    policy, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        policy, 
        None,
        dataloader, 
        None,
    )
 
    # Log training info (only on main process)
    if accelerator.is_main_process:
        num_learnable_params = get_model_param_count(policy, trainable_only=True)
        num_total_params = get_model_param_count(policy, trainable_only=False)

        logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")
        if cfg.env is not None:
            logging.info(f"{cfg.env.task=}")
        logging.info(f"{cfg.steps=} ({format_big_number(cfg.steps)})")
        num_datasets = len(dataset.ds.datasets)
        logging.info("============ Dataset Recipe =================")
        for dataset_idx in range(num_datasets):
            ds = dataset.ds.datasets[dataset_idx]
            if hasattr(ds, "repo_id"):
                logging.info(f"{ds.repo_id=}: {ds.num_frames=} ({format_big_number(ds.num_frames)}) {ds.num_episodes=} ({format_big_number(ds.num_episodes)}) {ds.weight=} {ds.ds_type=}")
            else:
                logging.info(f"{ds.repo_ids=}: {ds.num_frames=} ({format_big_number(ds.num_frames)}) {ds.num_episodes=} ({format_big_number(ds.num_episodes)}) {ds.weight=} {ds.ds_type=}")
        logging.info("============ Dataset Recipe End =================")
        logging.info(f"{num_learnable_params=} ({format_big_number(num_learnable_params)})")
        logging.info(f"{num_total_params=} ({format_big_number(num_total_params)})")
        logging.info(f"Number of processes: {accelerator.num_processes}")
        logging.info(f"Device: {accelerator.device}")
        logging.info(f"Mixed precision: {accelerator.mixed_precision}")
   
    train_metrics = {
        "loss": AverageMeter("loss", ":.3f"),
        "ce": AverageMeter("ce", ":.3f"),
        "mse": AverageMeter("mse", ":.3f"),
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        "update_s": AverageMeter("updt_s", ":.3f"),
        "dataloading_s": AverageMeter("data_s", ":.3f"),
    }
    train_tracker = MetricsTracker(
        dataset.num_frames, dataset.num_episodes, train_metrics, accelerator=accelerator, initial_step=step,
    )
    policy.train()
    if accelerator.is_main_process:
        logging.info("Start offline training on a fixed dataset")
    # Create iterator from dataloader
    seq_dataloader = policy.dataset(dataloader, policy.tokenize_action)
    flag_tokens_full = True
    for _ in range(step, cfg.steps):
    # for _ in range(step, cfg.steps):
        start_time = time.perf_counter()
        try:
            data_batch, data_indexes = next(seq_dataloader)
        except:
            print('fetch next frame error!')
            seq_dataloader = policy.dataset(dataloader, policy.tokenize_action)
            continue
        train_tracker.dataloading_s = time.perf_counter() - start_time
        train_tracker, output_dict = update_policy(
                train_tracker,
                policy,
                data_batch,
                accelerator,
                step,
        )

        # Note: eval and checkpoint happens *after* the `step`th training update has completed, so we
        # increment `step` here.
        if tokens <= cfg.dataset.token_num * 1e9:
            train_sample_seen[torch.cat(data_indexes).detach().cpu().numpy().astype(np.int64)] = 1
            flag_token_full = False
        else:
            if not flag_token_full:
                train_sample_seen = accelerator.gather(torch.tensor(train_sample_seen).cuda()[None, :]).any(0).float().cpu().numpy()
                train_sample_weights[np.where(train_sample_seen != 1)[0]] *= 0
                train_sampler = CustomWeightedRandomSampler(weights=train_sample_weights, num_samples=len(train_sample_weights), accelerator=accelerator)
                dataloader = torch.utils.data.DataLoader(
                    dataset,
                    num_workers=0, # multiprocessing.cpu_count(), # cfg.num_workers, ## TODO: set worker
                    batch_size=1,
                    shuffle=True,
                    pin_memory=True,
                    drop_last=False,
                )
                seq_dataloader = policy.dataset(dataloader, policy.tokenize_action)
            flag_token_full = True
        step += len(data_batch['sample_lens'])
        onestep += 1
        num_tokens = data_batch['sequence_length']
        tokens += num_tokens
        num_tokens, step = train_tracker.step(num_tokens, add_steps=len(data_batch['sample_lens']))

        is_log_step = cfg.log_freq > 0 and step % cfg.log_freq < accelerator.num_processes
        is_saving_step = onestep % cfg.save_freq == 0 or onestep == cfg.steps
        is_eval_step = cfg.eval_freq > 0 and step % cfg.eval_freq < accelerator.num_processes
        is_eval_step = False

        if cfg.save_checkpoint and is_saving_step:
            accelerator.wait_for_everyone()
        
        if cfg.save_checkpoint and is_saving_step:
            logging.info(f"Checkpoint policy after step {step}")
            checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, cfg.steps, step)
            unwrapped_policy = accelerator.unwrap_model(policy)
            train_sample_seen = accelerator.gather(torch.tensor(train_sample_seen).cuda()[None, :]).any(0).float()
            if accelerator.is_main_process:
                save_checkpoint(checkpoint_dir, step, tokens, cfg, unwrapped_policy, optimizer, lr_scheduler, train_sample_weights, val_sample_weights_dict, train_sample_seen)
                update_last_checkpoint(checkpoint_dir)
        if cfg.save_checkpoint and is_saving_step:
            accelerator.wait_for_everyone()


        if is_log_step and accelerator.is_main_process:
            print("logging.....")
            logging.info(train_tracker)
            if wandb_logger:
                wandb_log_dict = train_tracker.to_dict()
                observation_images = []
                # for key in batch.keys():
                #     if "images." in key and "observation" in key:
                #         observation_images.append((batch[key][0].detach().cpu().numpy().transpose((1, 2, 0)) * 255).astype(np.uint8))
                # observation_image = cv2.hconcat(observation_images)
                # wandb_log_dict.update({"observation": [observation_image]})
                # predict_action = str(output_dict['predict_action'].view(-1).tolist())
                # gt_action = str(output_dict['gt_action'].tolist())
                # wandb_log_dict.update({"action": [{"gt_action": gt_action, "predicted_action": predict_action}]})
                wandb_logger.log_dict(wandb_log_dict, step=tokens)
            train_tracker.reset_averages()
        
        if is_eval_step:
            step_id = get_step_identifier(step, cfg.steps)
            logging.info(f"Eval policy at step {step}")

            # Unwrap model for evaluation
            # unwrapped_policy = accelerator.unwrap_model(policy)
            # unwrapped_policy.eval()
            
            ## TODO: validation
            logging.info("validation begins")
            ds_types = val_sample_weights_dict.keys()
            val_loss_dict = {}
            validation_metrics = {}
            for ds_type in ds_types:
                dl_iter_val = iter(dataloader)
                val_total_steps = cfg.val_sample_num ## TODO:
                all_loss_values = torch.tensor(0.).float().cuda()
                all_mse_values = torch.tensor(0.).float().cuda()
                all_ce_values = torch.tensor(0.).float().cuda()
                # val_sampler = torch.utils.data.WeightedRandomSampler(weights=val_sample_weights_dict[ds_type], num_samples=len(train_sample_weights))
                val_sampler = CustomWeightedRandomSampler(weights=val_sample_weights_dict[ds_type], num_samples=len(val_sample_weights_dict[ds_type]), accelerator=accelerator)
                val_dataloader = torch.utils.data.DataLoader(
                    dataset,
                    num_workers=64, # cfg.num_workers, ## TODO: set worker
                    batch_size=1,
                    sampler=val_sampler,
                    pin_memory=True,
                    drop_last=False,
                )
                val_seq_dataloader = policy.dataset(val_dataloader, policy.tokenize_action)
                for val_step in tqdm(range(val_total_steps)):
                    dl_iter = iter(val_seq_dataloader)
                    val_data_batch, _ = next(dl_iter)          
                    with torch.no_grad():
                        loss, output_dict = policy.forward(val_data_batch)
                    loss_value = loss.detach().mean()
                    mse = output_dict['mse']
                    ce = output_dict['ce']
                    all_loss_values += loss_value / val_total_steps
                    all_mse_values += mse / val_total_steps
                    all_ce_values += ce / val_total_steps

                # mse_loss_value = accelerator.gather(all_mse_values.detach()).mean().item()
                # ce_loss_value = accelerator.gather(all_ce_values.detach()).mean().item()
                # loss_value = accelerator.gather(all_loss_values.detach()).mean().item()
                ce_loss_value = all_loss_values.detach().mean().item()
                mse_loss_value = all_mse_values.detach().mean().item()
                loss_value = all_loss_values.detach().mean().item()

                validation_metrics.update({
                    f"{ds_type}_loss": AverageMeter("loss", ":3f"),
                    f"{ds_type}_ce": AverageMeter("ce", ":.3f"),
                    f"{ds_type}_mse": AverageMeter("mse", ":.3f"),
                })
                val_loss_dict[f'{ds_type}_loss'] = loss_value
                val_loss_dict[f'{ds_type}_ce'] = ce_loss_value
                val_loss_dict[f'{ds_type}_mse'] = mse_loss_value
            validation_tracker = MetricsTracker(
                dataset.num_frames, dataset.num_episodes, validation_metrics, accelerator=accelerator
            )
            for val_loss_key in val_loss_dict.keys():
                setattr(validation_tracker, val_loss_key, val_loss_dict[val_loss_key])
            print("validation end")
            val_tracker_dict = validation_tracker.to_dict() 
            wandb_log_dict = {**val_tracker_dict}
            if wandb_logger:
                wandb_logger.log_dict(wandb_log_dict, step=step, mode="validation")


        if False:
            process_index = accelerator.process_index
            num_processes = accelerator.num_processes
            local_eval_envs = eval_envs[accelerator.process_index::accelerator.num_processes] if accelerator.process_index in list(range(len(eval_envs))) else None
            with (
                torch.no_grad(),
            ):
                eval_info = eval_policy(
                    local_eval_envs,
                    unwrapped_policy,
                    cfg.eval.n_episodes,
                    videos_dir=cfg.output_dir / "eval" / f"videos_step_{step_id}",
                    max_episodes_rendered=4,
                    start_seed=0,
                )
            eval_metrics = {
                "avg_sum_reward": AverageMeter("∑rwrd", ":.3f", accelerator),
                "pc_success": AverageMeter("success", ":.1f", accelerator),
                "eval_s": AverageMeter("eval_s", ":.3f", accelerator),
            }
            eval_tracker = MetricsTracker(
                cfg.batch_size * accelerator.num_processes, 
                dataset.num_frames, 
                dataset.num_episodes, 
                eval_metrics, 
                initial_step=step,
                accelerator=accelerator,
            )
            eval_tracker.eval_s = eval_info["aggregated"].pop("eval_s")

            eval_tracker.avg_sum_reward = eval_info["aggregated"].pop("avg_sum_reward")
            eval_tracker.pc_success = eval_info["aggregated"].pop("pc_success")
            if accelerator.is_main_process:
                eval_tracker_dict = eval_tracker.to_dict()
                eval_tracker_dict["video_paths"] = [eval_info['per_episode'][i]['video_path'] for i in range(len(eval_info['per_episode']))]
                eval_tracker_dict["observation_predicted_images"] = [eval_info['per_episode'][i]['observation_predicted_images'] for i in range(len(eval_info['per_episode']))]
                eval_info.pop("per_episode")
                logging.info(eval_tracker)
                wandb_log_dict = {**eval_tracker_dict, **eval_info}
                if wandb_logger:
                    wandb_logger.log_dict(wandb_log_dict, step=tokens, mode="eval")
            # Set back to training mode
            print("eval log dict done")
            policy.train()
    # Wait for all processes to finish
    accelerator.wait_for_everyone()

    if eval_envs:
        for eval_env in eval_envs:
            eval_env.close()
    if accelerator.is_main_process:
        logging.info("End of training")


if __name__ == "__main__":
    init_logging()
    train()
