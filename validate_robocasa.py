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
from tqdm import tqdm
from lerobot.configs import parser
from lerobot.policies.pi0.modeling_pi0 import PI0Policy
from lerobot.policies.factory import make_policy
import pickle
from lerobot.configs.train import TrainPipelineConfig
import time
import robocasa  # noqa: F401
import robosuite  # noqa: F401
from robocasa.utils.gym_utils import GrootRoboCasaEnv  # noqa: F401
import gymnasium as gym
from lerobot.datasets.factory import make_dataset
from lerobot.datasets.lerobot_dataset import (
    LeRobotDataset,
    LeRobotDatasetMetadata,
    MultiLeRobotDataset,
)
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.transforms import ImageTransforms

os.environ["TOKENIZERS_PARALLELISM"] = "false"

SHOW_PREDICT_IMAGE = True ## TODO:

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
    """Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90"""
    num_steps_wait: int = 10
    """Number of steps to wait for objects to stabilize in sim."""
    num_trials_per_task: int = 20
    """Number of rollouts per task."""

    # --- Evaluation arguments ---
    video_out_path: str = "./outputs/video/bagel-full-gen-unified_libero"
    """Path to save videos."""
    device: str = "cuda"
    """Device to use for evaluation."""

    seed: int = 7
    """Random Seed (for reproducibility)"""

    checkpoint_dir: str = None

args = Args()

@parser.wrap()
def eval(cfg: TrainPipelineConfig) -> None:
    cfg.type = "pi0"
    cfg.resume = True
    cfg.validate()
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    checkpoint_path = cfg.output_dir / "checkpoints" / "00000285000" 
    policy = PI0Policy.from_pretrained(checkpoint_path / "pretrained_model", dataset_stats = pickle.load(open(os.path.join(checkpoint_path, "dataset_stats.pkl"), 'rb')))
    policy.to('cuda:0')
    policy.eval()
    
    env_name = "gr1_unified/PnPCupToDrawerClose_GR1ArmsAndWaistFourierHands_Env"
    data_path = "/dataset_rc_mm/share/datasets/huggingface.co/nvidia/PhysicalAI-Robotics-GR00T-Teleop-Sim/LeRobot/gr1_unified.PnPBottleToCabinetClose"

    ds_meta = LeRobotDatasetMetadata(data_path)
    image_transforms = (
        ImageTransforms(cfg.dataset.image_transforms) if cfg.dataset.image_transforms.enable else None
    )
    delta_timestamps = resolve_delta_timestamps(cfg.policy, ds_meta)
    dataset = LeRobotDataset(
        data_path,
        root=None,
        episodes=None,
        delta_timestamps=delta_timestamps,
        image_transforms=image_transforms,
        revision=None,
        video_backend=cfg.dataset.video_backend,
        use_ref=cfg.policy.use_ref,
    )
    stats = dataset.stats
   
    env = gym.make(env_name, enable_render=True)    
    task_description = env_name.split("/")[1].split("_")[0]
    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    task_episodes, task_successes = 0, 0
    total_episodes = 0
    
    # Reset environment and policy
    obs, _ = env.reset()
    policy.reset()

    # Setup
    t = 0
    frames = []

    # Add initial frame
    agentview_image = np.ascontiguousarray(obs["video.ego_view_bg_crop_pad_res256_freq20"])
    last_predict_image = np.zeros_like(agentview_image).astype(np.uint8)
    # frames.append(agentview_image)
    # import ipdb; ipdb.set_trace())
    logging.info(f"Starting episode {task_episodes+1}...")
    data = dataset[0]
    idx = 0
    while True:
        data = dataset.__getitem__(idx, random=False)
        if data['episode_index'] > 0:
            break
        if idx > 200:
            break
        print("idx:", idx)
        actions = data['action']
        for action in actions:
            idx += 1
            if isinstance(obs, tuple):
                agentview_image = np.ascontiguousarray(obs[0]["video.ego_view_bg_crop_pad_res256_freq20"][:, :, ::-1])
            else:
                agentview_image = np.ascontiguousarray(obs["video.ego_view_bg_crop_pad_res256_freq20"][:, :, ::-1])
            
            frames.append(agentview_image)
            cv2.imwrite('debug.png', agentview_image)
            state = np.concatenate([
                obs['state.left_arm'], obs['state.left_hand'], obs['state.right_arm'], obs['state.right_hand'], np.zeros(3,), obs['state.waist']
            ], axis=0)
            
            observation = {
                "observation.images.image": torch.from_numpy(agentview_image / 255.0)
                .permute(2, 0, 1)
                .to(torch.float32)
                .to(args.device).unsqueeze(0),
                "task": [task_description],
                "observation.state": torch.from_numpy(state).float().cuda()
            }
            # Query model to get action
            with torch.inference_mode():
                action_tensor, predict_image = policy.select_action(observation, prior=None)
            action = action_tensor.cpu().numpy()[0]
            if predict_image is not None:
                last_predict_image = np.asarray(predict_image)
                frames[-1] = cv2.hconcat([frames[-1], last_predict_image])

            action = {
                "action.left_arm": action[:7],
                "action.left_hand": action[7:13],
                "action.right_arm": action[13:20],
                "action.right_hand": action[20:26],
                "action.neck": action[26:29],
                "action.waist": action[29:32],
            }
            obs, reward, termination, truncation, env_info  = env.step(action)
            agentview_image = np.ascontiguousarray(obs["video.ego_view_bg_crop_pad_res256_freq20"][:, :, ::-1])            
            cv2.imwrite('debug.png', agentview_image)
    # Save a replay video of the episode
    video_path = (
        pathlib.Path(args.video_out_path) / f"replay_task_{task_description}.mp4"
    )
    fps = 30
    width, height, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer =  cv2.VideoWriter(video_path, fourcc, fps, (height, width)) 
    for image in frames:
        writer.write(image)
    writer.release()
    logging.info(f"Saved video to {video_path}")
    import ipdb;ipdb.set_trace()


def _quat2axisangle(quat):
    """
    Copied from robosuite:
    https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    eval()
