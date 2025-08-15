import logging
import time
from contextlib import nullcontext
from pprint import pformat
from typing import Any

import torch
from termcolor import colored
from torch.amp import GradScaler
from torch.optim import Optimizer

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
from lerobot.scripts.eval import eval_policy, validate_policy
from accelerate import Accelerator
from accelerate.utils import set_seed as accelerate_set_seed
import os
import numpy as np
import cv2
from lerobot.common.envs.utils import add_envs_task, check_env_attributes_and_types, preprocess_observation
import torch
import pickle

communication_file = f"/root/Bagel_VLA/outputs/communication0.pkl"
communication_lock_file = f"/root/Bagel_VLA/outputs/communication_lock0.txt"

open(communication_lock_file, "w").write("0")

def receive():
    while True:
        if open(communication_lock_file, "r").read().strip() == "1":
            data = pickle.load(open(communication_file, "rb"))
            break
    return data

def listen_and_process_input(llm_model):
    data = receive()
    command = data['command']
    if command == "generate":
        print(f"receiving data")
        data2 = data['params']
        observation = preprocess_observation(data2)
        observation = {
            key: observation[key].cuda().unsqueeze(0) for key in observation
        }
        observation['task'] = [data2['task']]
        action_pred = llm_model.select_action(observation)
        outputs = action_pred
        print("outputs:", outputs)
        print(f"output generated")
    if command == "reset":
        llm_model.reset()
        outputs = "OK"
    pickle.dump(outputs, open(communication_file, "wb"))
    open(communication_lock_file, "w").write("0")



@parser.wrap()
def server_start(cfg: TrainPipelineConfig):
    cfg.resume = True
    cfg.validate()
    logging.info(pformat(cfg.to_dict()))

    if cfg.seed is not None:
        set_seed(cfg.seed)

    # Initialize accelerator
    from accelerate.utils import DistributedDataParallelKwargs

    from lerobot.common.utils.wandb_utils import cfg_to_group, get_wandb_run_id_from_filesystem

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        mixed_precision="no",
        gradient_accumulation_steps=cfg.policy.gradient_accumulation_steps,
        log_with="wandb" if cfg.wandb.enable else None,
        kwargs_handlers=[ddp_kwargs],
        project_dir=cfg.output_dir,
    )
    # Setup device - accelerator handles device placement
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    # Create dataset
    if accelerator.is_main_process:
        logging.info("Creating dataset")

    if accelerator.is_main_process:
        logging.info("Creating policy")
    cfg.policy.device = "cpu"
    ds_meta_path = cfg.output_dir / "checkpoints" / "ds_meta.pkl"
    if os.path.exists(ds_meta_path):
        ds_meta = pickle.load(open(ds_meta_path, 'rb'))
    else:
        dataset = make_dataset(cfg)
        ds_meta = dataset.meta

    policy = make_policy(
        cfg=cfg.policy,
        ds_meta=ds_meta,
    ).cpu()
    
    model_path = cfg.load_bin
    py_ckpt = torch.load(open(model_path, 'rb'), map_location="cuda:0")

    policy.load_state_dict(py_ckpt, strict=True)
    policy = accelerator.prepare(policy)
    policy = accelerator.unwrap_model(policy)
    print('server start')
    while True:
        listen_and_process_input(policy)



if __name__ == "__main__":
    init_logging()
    server_start() 

