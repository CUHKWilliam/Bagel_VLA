"""
This script demonstrates how to evaluate a pretrained smolVLA policy on the LIBERO benchmark.
"""

import collections
import dataclasses
import logging
import math
import pathlib
import os

import cv2
import draccus
import imageio
import numpy as np
import torch
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from tqdm import tqdm
from lerobot.configs import parser
from lerobot.policies.pi0.modeling_pi0 import PI0Policy
from lerobot.policies.factory import make_policy
import pickle
from lerobot.configs.train import TrainPipelineConfig
import time

os.environ["TOKENIZERS_PARALLELISM"] = "false"

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data



def normalize_gripper_action(action, binarize=True):
    """
    Changes gripper action (last dimension of action vector) from [0,1] to [-1,+1].
    Necessary for some environments (not Bridge) because the dataset wrapper standardizes gripper actions to [0,1].
    Note that unlike the other action dimensions, the gripper action is not normalized to [-1,+1] by default by
    the dataset wrapper.

    Normalization formula: y = 2 * (x - orig_low) / (orig_high - orig_low) - 1
    """
    # Just normalize the last action to [-1,+1].
    orig_low, orig_high = 0.0, 1.0
    action[..., -1] = 2 * (action[..., -1] - orig_low) / (orig_high - orig_low) - 1

    if binarize:
        # Binarize to -1 or +1.
        action[..., -1] = np.sign(action[..., -1])

    return action


def invert_gripper_action(action):
    """
    Flips the sign of the gripper action (last dimension of action vector).
    This is necessary for some environments where -1 = open, +1 = close, since
    the RLDS dataloader aligns gripper actions such that 0 = close, 1 = open.
    """
    action[..., -1] = action[..., -1] * -1.0
    return action


class Args:
    """
    Evaluation arguments for smolVLA on LIBERO.
    """
    # --- LIBERO environment-specific parameters ---
    task_suite_name: str = "libero_spatial"
    """Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90"""
    num_steps_wait: int = 10
    """Number of steps to wait for objects to stabilize in sim."""
    num_trials_per_task: int = 1
    """Number of rollouts per task."""

    # --- Evaluation arguments ---
    video_out_path: str = "./outputs/eval_videos"
    """Path to save videos."""
    device: str = "cuda"
    """Device to use for evaluation."""

    seed: int = 7
    """Random Seed (for reproducibility)"""

    checkpoint_dir: str = None

args = Args()

@parser.wrap()
def push_to_hub(cfg: TrainPipelineConfig) -> None:
    cfg.type = "pi0"
    cfg.resume = True
    cfg.validate()
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    checkpoint_path = cfg.output_dir / "checkpoints" / "last" 
    policy = PI0Policy.from_pretrained(checkpoint_path / "pretrained_model")
    policy.push_to_hub("CUHKWilliam/openpi_libero_step-20k")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    push_to_hub()
