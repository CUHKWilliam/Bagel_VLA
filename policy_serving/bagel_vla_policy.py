import logging
import time
from typing import Dict, Optional, Tuple, List
import numpy as np
import torch
from typing_extensions import override
import websockets.sync.client

from lerobot.policies.pi0.modeling_pi0 import PI0Policy

import base_policy as _base_policy
import msgpack_numpy
import pickle
import os


class BagelVLAPolicy(_base_policy.BasePolicy):
    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        super().__init__()

        logging.info(f"Loading VLA policy from {checkpoint_path}")
        self.policy = PI0Policy.from_pretrained(checkpoint_path, dataset_stats=pickle.load(open(os.path.join(checkpoint_path, "../dataset_stats.pkl"), 'rb')))

        self.policy.to(device)
        self.policy.eval()
        self.policy.reset()
        
        self.device = device
        self.num_prior_actions = 4 
        self.prior = [[], []]  # (actions, observations)
        
    @override
    def infer(self, obs: Dict) -> Dict:
        cam_high = obs["observation.images.cam_high"]
        cam_left_wrist = obs["observation.images.cam_left_wrist"]
        cam_right_wrist = obs["observation.images.cam_right_wrist"]
        state = obs["observation.state"]
        task_description = obs["task"]
        
        observation = {
            "observation.state": torch.from_numpy(state.copy()).to(torch.float32).to(self.device).unsqueeze(0),
            "observation.images.cam_high": torch.from_numpy(cam_high / 255.0)
                                                .permute(2, 0, 1)   # (C, H, W)
                                                .to(torch.float32)
                                                .to(self.device).unsqueeze(0),
            "observation.images.cam_left_wrist": torch.from_numpy(cam_left_wrist / 255.0)
                                                .permute(2, 0, 1)   # (H, W, C) -> (C, H, W)
                                                .to(torch.float32)
                                                .to(self.device).unsqueeze(0),
            "observation.images.cam_right_wrist": torch.from_numpy(cam_right_wrist / 255.0)
                                                .permute(2, 0, 1)   # (C, H, W)
                                                .to(torch.float32)
                                                .to(self.device).unsqueeze(0),
            "task": [task_description],
        }
        
        # print(f"\n input obs:")
        # print(f"  state: {observation['observation.state'].shape}")
        # print(f"  cam_high: {observation['observation.images.cam_high'].shape}")
        # print(f"  cam_left_wrist: {observation['observation.images.cam_left_wrist'].shape}")
        # print(f"  cam_right_wrist: {observation['observation.images.cam_right_wrist'].shape}")
        # print(f"  task: {task_description}")
        
        with torch.inference_mode():
            if len(self.prior[0]) < self.num_prior_actions:
                action_tensor = torch.zeros(1, observation['observation.state'].shape[1]).to(self.device)
                predict_image = None
                
                self.prior[0].append(action_tensor)
                self.prior[1].append(observation)
            else:
                if len(self.prior[1]) == self.num_prior_actions:
                    self.prior[1].append(observation)
                action_tensor, predict_image = self.policy.select_action(observation, prior=self.prior)
                
                if len(self.prior[0]) >= self.num_prior_actions:
                    self.prior[0].pop(0)
                    self.prior[1].pop(0)
                self.prior[0].append(action_tensor)
                self.prior[1].append(observation)
        
        action_numpy = action_tensor.cpu().numpy()[0]
        
        result = {
            "actions": action_numpy.tolist(), 
            "action_tensor_shape": list(action_tensor.shape),
            "predict_image_available": predict_image is not None,
        }
        
        # if predict_image is not None:
        #     predict_image_np = predict_image.cpu().numpy()
        #     result["predict_image_shape"] = list(predict_image_np.shape)
            
        return result
    
    @override
    def reset(self) -> None:
        self.policy.reset()
        self.prior = ([], [])
        logging.info("VLA policy reset")
