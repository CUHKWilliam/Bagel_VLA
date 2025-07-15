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
"""Evaluate a policy on an environment by running rollouts and computing metrics.

Usage examples:

You want to evaluate a model from the hub (eg: https://huggingface.co/lerobot/diffusion_pusht)
for 10 episodes.

```
python lerobot/scripts/eval.py \
    --policy.path=lerobot/diffusion_pusht \
    --env.type=pusht \
    --eval.batch_size=10 \
    --eval.n_episodes=10 \
    --use_amp=false \
    --device=cuda
```

OR, you want to evaluate a model checkpoint from the LeRobot training script for 10 episodes.
```
python lerobot/scripts/eval.py \
    --policy.path=outputs/train/diffusion_pusht/checkpoints/005000/pretrained_model \
    --env.type=pusht \
    --eval.batch_size=10 \
    --eval.n_episodes=10 \
    --use_amp=false \
    --device=cuda
```

Note that in both examples, the repo/folder should contain at least `config.json` and `model.safetensors` files.

You can learn about the CLI options for this script in the `EvalPipelineConfig` in lerobot/configs/eval.py
"""

import json
import logging
import threading
import time
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from pprint import pformat
from typing import Callable

import einops
import gymnasium as gym
import numpy as np
import torch
from termcolor import colored
from torch import Tensor, nn
from tqdm import trange

from lerobot.common.envs.factory import make_env
from lerobot.common.envs.utils import add_envs_task, check_env_attributes_and_types, preprocess_observation
from lerobot.common.policies.factory import make_policy
from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.policies.utils import get_device_from_parameters
from lerobot.common.utils.io_utils import write_video
from lerobot.common.utils.random_utils import set_seed
from lerobot.common.utils.utils import (
    get_safe_torch_device,
    init_logging,
    inside_slurm,
)
from lerobot.configs import parser
from lerobot.configs.eval import EvalPipelineConfig
import cv2
import libero
import copy
def rollout(
    env,
    env_id,
    policy: PreTrainedPolicy,
    seeds: list[int] | None = None,
    return_observations: bool = False,
    render_callback: Callable[[gym.vector.VectorEnv], None] | None = None,
) -> dict:
    """Run a batched policy rollout once through a batch of environments.

    Note that all environments in the batch are run until the last environment is done. This means some
    data will probably need to be discarded (for environments that aren't the first one to be done).

    The return dictionary contains:
        (optional) "observation": A dictionary of (batch, sequence + 1, *) tensors mapped to observation
            keys. NOTE that this has an extra sequence element relative to the other keys in the
            dictionary. This is because an extra observation is included for after the environment is
            terminated or truncated.
        "action": A (batch, sequence, action_dim) tensor of actions applied based on the observations (not
            including the last observations).
        "reward": A (batch, sequence) tensor of rewards received for applying the actions.
        "success": A (batch, sequence) tensor of success conditions (the only time this can be True is upon
            environment termination/truncation).
        "done": A (batch, sequence) tensor of **cumulative** done conditions. For any given batch element,
            the first True is followed by True's all the way till the end. This can be used for masking
            extraneous elements from the sequences above.

    Args:
        env: The batch of environments.
        policy: The policy. Must be a PyTorch nn module.
        seeds: The environments are seeded once at the start of the rollout. If provided, this argument
            specifies the seeds for each of the environments.
        return_observations: Whether to include all observations in the returned rollout data. Observations
            are returned optionally because they typically take more memory to cache. Defaults to False.
        render_callback: Optional rendering callback to be used after the environments are reset, and after
            every step.
    Returns:
        The dictionary described above.
    """
    assert isinstance(policy, nn.Module), "Policy must be a PyTorch nn module."
    device = get_device_from_parameters(policy)

    # Reset the policy and environments.
    policy.reset()
    observations = env.step([0] * 7)
    observation1 = observations[0]['agentview_image']
    observation2 = observations[0]['robot0_eye_in_hand_image']
    raw_observation = {
        "pixels":{
            "agentview_image": observation1,
            "robot0_eye_in_hand_image": observation2,
        }
    }
    context_raw_observation = copy.deepcopy(raw_observation)

    if render_callback is not None:
        render_callback(env, env_id)

    all_observations = []
    all_actions = []
    all_rewards = []
    all_successes = []
    all_dones = []

    step = 0
    # Keep track of which environments are done.
    done = False
    # max_steps = env.call("_max_episode_steps")[0]
    max_steps = 50 ## TODO:
    # check_env_attributes_and_types(env)
    observation_predicted_images = []
    step_idx = 0
    UPDATE_CONTEXT_EVERY = 50
    UPDATE_CONTEXT_OBSERVATION_EVERY = 25
    while not done:
        if step_idx == UPDATE_CONTEXT_EVERY or step_idx == 0:
            past_key_values, newlens, new_rope = None, None, None
        # Numpy array to tensor and changing dictionary keys to LeRobot policy format.
        observation = preprocess_observation(raw_observation)
        if return_observations:
            all_observations.append(deepcopy(observation))
        observation = {
            key: observation[key].to(device, non_blocking=device.type == "cuda").unsqueeze(0) for key in observation
        }
        observation['task'] = [env.language_instruction]
        context_observation = copy.deepcopy(observation)
        with torch.inference_mode():
            actions, predicted_images, past_key_values, newlens, new_rope = policy.select_action(observation, context_observation, past_key_values=past_key_values, newlens=newlens, new_rope=new_rope, step_idx=step_idx)
        observation_image = cv2.hconcat([raw_observation['pixels']['agentview_image'], raw_observation['pixels']['robot0_eye_in_hand_image']])
        if predicted_images is not None:
            observation_predicted_image = cv2.vconcat([observation_image, np.asarray(predicted_images[0])])
        else:
            observation_predicted_image = observation_image
        observation_predicted_images.append(observation_predicted_image)
        # Convert to CPU / numpy.
        if isinstance(actions, torch.Tensor):
            actions = actions.to("cpu").numpy()
        success = False
        for action in actions:
            # Apply the next action.
            try:
                new_observation, reward, done, info = env.step(action)
                step_idx += 1
                success = env.check_success()
                if success:
                    break
            except:
                done = True
                break
            if render_callback is not None:
                render_callback(env, env_id)
        # VectorEnv stores is_success in `info["final_info"][env_index]["is_success"]`. "final_info" isn't
        # available of none of the envs finished.
        successes = success

        all_actions.append(torch.from_numpy(action))
        all_rewards.append(torch.from_numpy(np.array(reward)))
        all_dones.append(torch.from_numpy(np.array(done)))
        all_successes.append(torch.tensor(int(successes)).bool())

        step += 1
        raw_observation['pixels'] = {
            "agentview_image": new_observation['agentview_image'],
            "robot0_eye_in_hand_image": new_observation['robot0_eye_in_hand_image'],
        }
        if step_idx == UPDATE_CONTEXT_IMAGE_EVERY:
            context_raw_observation = copy.deepcopy(raw_observation)
            step_idx = 0
    # Track the final observation.
    if return_observations:
        observation = preprocess_observation(observation)
        all_observations.append(deepcopy(observation))

    # Stack the sequence along the first dimension so that we have (batch, sequence, *) tensors.
    ret = {
        "action": torch.stack(all_actions, dim=1),
        "reward": torch.tensor(all_rewards),
        "success": torch.tensor(all_successes),
        "done": torch.tensor(all_dones),
        "observation_predicted_images": observation_predicted_images,
    }
    if return_observations:
        stacked_observations = {}
        for key in all_observations[0]:
            stacked_observations[key] = torch.stack([obs[key] for obs in all_observations], dim=1)
        ret["observation"] = stacked_observations
    
    if hasattr(policy, "use_original_modules"):
        policy.use_original_modules()
    return ret


def eval_policy(
    envs,
    policy: PreTrainedPolicy,
    n_episodes: int,
    max_episodes_rendered: int = 0,
    videos_dir: Path | None = None,
    return_episode_data: bool = False,
    start_seed: int | None = None,
) -> dict:
    """
    Args:
        env: The batch of environments.
        policy: The policy.
        n_episodes: The number of episodes to evaluate.
        max_episodes_rendered: Maximum number of episodes to render into videos.
        videos_dir: Where to save rendered videos.
        return_episode_data: Whether to return episode data for online training. Incorporates the data into
            the "episodes" key of the returned dictionary.
        start_seed: The first seed to use for the first individual rollout. For all subsequent rollouts the
            seed is incremented by 1. If not provided, the environments are not manually seeded.
    Returns:
        Dictionary with metrics and data regarding the rollouts.
    """
    if max_episodes_rendered > 0 and not videos_dir:
        raise ValueError("If max_episodes_rendered > 0, videos_dir must be provided.")

    if not isinstance(policy, PreTrainedPolicy):
        raise ValueError(
            f"Policy of type 'PreTrainedPolicy' is expected, but type '{type(policy)}' was provided."
        )

    start = time.time()
    policy.eval()

    # Determine how many batched rollouts we need to get n_episodes. Note that if n_episodes is not evenly
    # divisible by env.num_envs we end up discarding some data in the last batch.
    num_envs = len(envs)

    # Keep track of some metrics.
    sum_rewards = []
    max_rewards = []
    all_successes = []
    observation_predicted_images = []
    threads = []  # for video saving threads
    n_episodes_rendered = 0  # for saving the correct number of videos

    # Callback for visualization.
    def render_frame(env, env_id):
        # noqa: B023
        if n_episodes_rendered >= max_episodes_rendered:
            return
        image = env.step([0] * 7)[0]['agentview_image']
        if env_id not in ep_frames.keys():
            ep_frames[env_id] = []
        ep_frames[env_id].append(image)

    if max_episodes_rendered > 0:
        video_paths: list[str] = []

    if return_episode_data:
        episode_data: dict | None = None

    # we dont want progress bar when we use slurm, since it clutters the logs
    for batch_ix in range(n_episodes):
        # Cache frames for rendering videos. Each item will be (b, h, w, c), and the list indexes the rollout
        # step.
        if max_episodes_rendered > 0:
            ep_frames = {}

        rollout_data = {
            "done": [],
            "reward": [],
            "success": [],
            "observation_predicted_images": [],
        }
        for env_idx, env in enumerate(envs):
            env.seed(batch_ix)
            env.reset()
            a_rollout_data = rollout(
                env,
                env_idx,
                policy,
                return_observations=return_episode_data,
                render_callback=render_frame if max_episodes_rendered > 0 else None,
            )
            for key in rollout_data.keys():
                rollout_data[key].append(a_rollout_data[key])
        for key in rollout_data.keys():
            if "image" not in key:
                rollout_data[key] = torch.cat(rollout_data[key])
        # Figure out where in each rollout sequence the first done condition was encountered (results after
        # this won't be included).
        # Extend metrics.
        sum_rewards.extend(rollout_data['reward'].tolist())
        max_rewards.extend(rollout_data['reward'].tolist())
        batch_successes = rollout_data['success']
        all_successes.extend(batch_successes)
        observation_predicted_images.extend(rollout_data['observation_predicted_images'][0])

        # FIXME: episode_data is either None or it doesn't exist
        if return_episode_data:
            this_episode_data = _compile_episode_data(
                rollout_data,
                done_indices,
                start_episode_index=batch_ix * num_envs,
                start_data_index=(0 if episode_data is None else (episode_data["index"][-1].item() + 1)),
                fps=env.unwrapped.metadata["render_fps"],
            )
            if episode_data is None:
                episode_data = this_episode_data
            else:
                # Some sanity checks to make sure we are correctly compiling the data.
                assert episode_data["episode_index"][-1] + 1 == this_episode_data["episode_index"][0]
                assert episode_data["index"][-1] + 1 == this_episode_data["index"][0]
                # Concatenate the episode data.
                episode_data = {k: torch.cat([episode_data[k], this_episode_data[k]]) for k in episode_data}

        # Maybe render video for visualization.

        if max_episodes_rendered > 0 and len(ep_frames) > 0:
            for env_id in ep_frames.keys():
                a_ep_frames = ep_frames[env_id]
                if env_id >= max_episodes_rendered:
                    break

                videos_dir.mkdir(parents=True, exist_ok=True)
                video_path = videos_dir / f"eval_episode_{env_id}.mp4"
                video_paths.append(str(video_path))
                thread = threading.Thread(
                    target=write_video,
                    args=(
                        str(video_path),
                        a_ep_frames,  # + 1 to capture the last observation
                        20, ## TODO: set fps
                    ),
                )
                thread.start()
                threads.append(thread)

    # Wait till all video rendering threads are done.
    for thread in threads:
        thread.join()
    # Compile eval info.
    info = {
        "per_episode": [
            {
                "episode_ix": i,
                "sum_reward": sum_reward,
                "max_reward": max_reward,
                "video_path": video_paths[i] if i < len(video_paths) else None,
                "observation_predicted_images": observation_predicted_images[i] if i < 10 else None, ## TODO: limit image number
                "success": success,
            }
            for i, (sum_reward, max_reward, success) in enumerate(
                zip(
                    sum_rewards,
                    max_rewards,
                    all_successes,
                    strict=True,
                )
            )
        ],
        "aggregated": {
            "avg_sum_reward": float(np.nanmean(sum_rewards)),
            "avg_max_reward": float(np.nanmean(max_rewards)),
            "pc_success": float(np.nanmean(all_successes) * 100),
            "eval_s": time.time() - start,
            "eval_ep_s": (time.time() - start) / 1,
        },
    }

    if return_episode_data:
        info["episodes"] = episode_data

    if max_episodes_rendered > 0:
        info["video_paths"] = video_paths

    return info


def _compile_episode_data(
    rollout_data: dict, done_indices: Tensor, start_episode_index: int, start_data_index: int, fps: float
) -> dict:
    """Convenience function for `eval_policy(return_episode_data=True)`

    Compiles all the rollout data into a Hugging Face dataset.

    Similar logic is implemented when datasets are pushed to hub (see: `push_to_hub`).
    """
    ep_dicts = []
    total_frames = 0
    for ep_ix in range(rollout_data["action"].shape[0]):
        # + 2 to include the first done frame and the last observation frame.
        num_frames = done_indices[ep_ix].item() + 2
        total_frames += num_frames

        # Here we do `num_frames - 1` as we don't want to include the last observation frame just yet.
        ep_dict = {
            "action": rollout_data["action"][ep_ix, : num_frames - 1],
            "episode_index": torch.tensor([start_episode_index + ep_ix] * (num_frames - 1)),
            "frame_index": torch.arange(0, num_frames - 1, 1),
            "timestamp": torch.arange(0, num_frames - 1, 1) / fps,
            "next.done": rollout_data["done"][ep_ix, : num_frames - 1],
            "next.success": rollout_data["success"][ep_ix, : num_frames - 1],
            "next.reward": rollout_data["reward"][ep_ix, : num_frames - 1].type(torch.float32),
        }

        # For the last observation frame, all other keys will just be copy padded.
        for k in ep_dict:
            ep_dict[k] = torch.cat([ep_dict[k], ep_dict[k][-1:]])

        for key in rollout_data["observation"]:
            ep_dict[key] = rollout_data["observation"][key][ep_ix, :num_frames]

        ep_dicts.append(ep_dict)

    data_dict = {}
    for key in ep_dicts[0]:
        data_dict[key] = torch.cat([x[key] for x in ep_dicts])

    data_dict["index"] = torch.arange(start_data_index, start_data_index + total_frames, 1)

    return data_dict


@parser.wrap()
def eval_main(cfg: EvalPipelineConfig):
    logging.info(pformat(asdict(cfg)))

    # Check device is available
    device = get_safe_torch_device(cfg.policy.device, log=True)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    set_seed(cfg.seed)

    logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")

    logging.info("Making environment.")
    env = make_env(cfg.env, n_envs=cfg.eval.batch_size, use_async_envs=cfg.eval.use_async_envs)

    logging.info("Making policy.")

    policy = make_policy(
        cfg=cfg.policy,
        env_cfg=cfg.env,
    )
    policy.eval()

    with torch.no_grad(), torch.autocast(device_type=device.type) if cfg.policy.use_amp else nullcontext():
        info = eval_policy(
            env,
            policy,
            cfg.eval.n_episodes,
            max_episodes_rendered=10,
            videos_dir=Path(cfg.output_dir) / "videos",
            start_seed=cfg.seed,
        )
    print(info["aggregated"])

    # Save info
    with open(Path(cfg.output_dir) / "eval_info.json", "w") as f:
        json.dump(info, f, indent=2)

    env.close()

    logging.info("End of eval")


if __name__ == "__main__":
    init_logging()
    eval_main()
