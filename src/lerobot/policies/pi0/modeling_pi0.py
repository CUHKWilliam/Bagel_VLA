#!/usr/bin/env python

# Copyright 2025 Physical Intelligence and The HuggingFace Inc. team. All rights reserved.
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

"""
π0: A Vision-Language-Action Flow Model for General Robot Control

[Paper](https://www.physicalintelligence.company/download/pi0.pdf)
[Jax code](https://github.com/Physical-Intelligence/openpi)

Designed by Physical Intelligence. Ported from Jax by Hugging Face.
Disclaimer: It is not expected to perform as well as the original implementation.

Install pi0 extra dependencies:
```bash
pip install -e ".[pi0]"
```

Example of finetuning the pi0 pretrained model (`pi0_base` in `openpi`):
```bash
python -m lerobot.scripts.train \
--policy.path=lerobot/pi0 \
--dataset.repo_id=danaaubakirova/koch_test
```

Example of finetuning the pi0 neural network with PaliGemma and expert Gemma
pretrained with VLM default parameters before pi0 finetuning:
```bash
python -m lerobot.scripts.train \
--policy.type=pi0 \
--dataset.repo_id=danaaubakirova/koch_test
```

Example of using the pi0 pretrained model outside LeRobot training framework:
```python
policy = Pi0Policy.from_pretrained("lerobot/pi0")
```

"""
import os
import math
from collections import deque

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn
from transformers import AutoTokenizer

from lerobot.constants import ACTION, OBS_STATE
from lerobot.policies.normalize import Normalize, Unnormalize
from lerobot.policies.pi0.configuration_pi0 import PI0Config
from lerobot.policies.pi0.paligemma_with_expert import (
    PaliGemmaWithExpertConfig,
    PaliGemmaWithExpertModel,
)
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.utils import get_safe_dtype
from lerobot.policies.pi0.modeling.qwen2 import Qwen2Tokenizer
from lerobot.policies.pi0.modeling.bagel.qwen2_navit import NaiveCache
from lerobot.utils.data_utils import add_special_tokens
from lerobot.policies.pi0.modeling.qwen2 import Qwen2Config
from lerobot.policies.pi0.modeling.bagel import (
    BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM, SiglipVisionConfig, SiglipVisionModel
)
from dataclasses import dataclass, field
from transformers import HfArgumentParser
import functools
import yaml
import numpy as np
import torch
import cv2
from PIL import Image
from safetensors.torch import load_file
from .modeling.autoencoder import load_ae
from lerobot.policies.pi0.dataset_base import PackedDataset, SimpleCustomBatch
from copy import deepcopy
import functools
import yaml
import numpy as np
import torch
import cv2
from PIL import Image
from safetensors.torch import load_file

def autocast(data_batch, dtype1, dtype2):
    for key in data_batch.keys():
        value = data_batch[key]
        if isinstance(value, torch.Tensor) and value.dtype == dtype1:
            value = value.type(dtype2)
            data_batch[key] = value
    return data_batch

@dataclass
class DataArguments:
    dataset_config_file: str = field(
        default="data/configs/example.yaml",
        metadata={"help": "YAML file specifying dataset groups, weights, and preprocessing rules."}
    )
    prefetch_factor: int = field(
        default=2,
        metadata={"help": "How many batches each DataLoader worker pre-loads in advance."}
    )
    num_workers: int = field(
        default=1,
        metadata={"help": "Number of background workers for the PyTorch DataLoader."}
    )
    max_num_tokens_per_sample: int = field(
        # default=26384,
        default=3000,
        metadata={"help": "Maximum tokens allowed in one raw sample; longer samples are skipped."}
    )
    max_num_tokens: int = field(
        # default=66864,
        default=6000,
        metadata={"help": "Hard limit on tokens in a packed batch; flush if adding a sample would exceed it."}
    )
    prefer_buffer_before: int = field(
        default=16384,
        metadata={"help": "While batch length is below this, pop from the overflow buffer before new sampling."}
    )
    max_buffer_size: int = field(
        default=50,
        metadata={"help": "Maximum number of oversized samples kept in the overflow buffer."}
    )
    data_seed: int = field(
        default=42,
        metadata={"help": "Seed used when shuffling / sampling data shards to ensure reproducibility."}
    )
@dataclass
class ModelArguments:
    model_path: str = field(
        default="/root/lerobot/weight/BAGEL-7B-MoT",
        metadata={"help": "Path of the pretrained BAGEL model."}
    )
    llm_path: str = field(
        default="hf/Qwen2.5-0.5B-Instruct/",
        metadata={"help": "Path or HuggingFace repo ID of the pretrained Qwen2-style language model."}
    )
    llm_qk_norm: bool = field(
        default=True,
        metadata={"help": "Enable QK LayerNorm (qk_norm) inside the attention blocks."}
    )
    tie_word_embeddings: bool = field(
        default=False,
        metadata={"help": "Share input and output word embeddings (tied embeddings)."}
    )
    layer_module: str = field(
        default="Qwen2MoTDecoderLayer",
        metadata={"help": "Python class name of the decoder layer to instantiate."}
    )
    vae_path: str = field(
        default="ffxvs/vae-flux/ae.safetensors",
        metadata={"help": "Path to the pretrained VAE checkpoint for latent-space image generation."}
    )
    vit_path: str = field(
        default="hf/siglip-so400m-14-980-flash-attn2-navit/",
        metadata={"help": "Path or repo ID of the SigLIP Vision Transformer used for image understanding."}
    )
    max_latent_size: int = field(
        default=64,
        metadata={"help": "Maximum latent grid size (patches per side) for the VAE latent tensor."}
    )
    latent_patch_size: int = field(
        default=2,
        metadata={"help": "Spatial size (in VAE pixels) covered by each latent patch."}
    )
    vit_patch_size: int = field(
        default=14,
        metadata={"help": "Patch size (pixels) for the Vision Transformer encoder."}
    )
    vit_max_num_patch_per_side: int = field(
        default=70,
        metadata={"help": "Maximum number of ViT patches along one image side after cropping / resize."}
    )
    connector_act: str = field(
        default="gelu_pytorch_tanh",
        metadata={"help": "Activation function used in the latent-to-text connector MLP."}
    )
    interpolate_pos: bool = field(
        default=False,
        metadata={"help": "Interpolate positional embeddings when image resolution differs from pre-training."}
    )
    vit_select_layer: int = field(
        default=-2,
        metadata={"help": "Which hidden layer of the ViT to take as the visual feature (negative = from the end)."}
    )
    vit_rope: bool = field(
        default=False,
        metadata={"help": "Replace ViT positional encodings with RoPE."}
    )

    text_cond_dropout_prob: float = field(
        default=0.1,
        metadata={"help": "Probability of dropping text embeddings during training."}
    )
    vae_cond_dropout_prob: float = field(
        default=0.3,
        metadata={"help": "Probability of dropping VAE latent inputs during training."}
    )
    vit_cond_dropout_prob: float = field(
        default=0.3,
        metadata={"help": "Probability of dropping ViT visual features during training."}
    )
@dataclass
class TrainingArguments:
    # --- modality switches ---
    visual_gen: bool = field(
        default=True,
        metadata={"help": "Train image generation branch."}
    )
    visual_und: bool = field(
        default=True,
        metadata={"help": "Train image understanding branch."}
    )
    action_gen: bool = field(
        default=True,
        metadata={"help": "Train to generate action."}
    )

    # --- reporting frequency ---
    timestep_shift: float = field(
        default=1.0,
        metadata={"help": "Shift applied to diffusion timestep indices (for latent prediction)."}
    )
    mse_weight: float = field(
        default=1.0,
        metadata={"help": "Scaling factor for the image-reconstruction MSE loss term."}
    )
    ce_weight: float = field(
        default=1.0,
        metadata={"help": "Scaling factor for the language cross-entropy loss term."}
    )
    ce_loss_reweighting: bool = field(
        default=False,
        metadata={"help": "Reweight CE loss by token importance (provided via ce_loss_weights)."}
    )
    expected_num_tokens: int = field(
        default=32768,
        metadata={"help": "Soft target token count; yield the batch once it reaches or exceeds this size."}
    )

    # --- module freezing ---
    freeze_llm: bool = field(
        default=False,
        metadata={"help": "Keep language-model weights fixed (no gradient updates)."}
    )
    freeze_vit: bool = field(
        default=True,
        metadata={"help": "Keep ViT weights fixed during training."}
    )
    freeze_vae: bool = field(
        default=True,
        metadata={"help": "Keep VAE weights fixed; only predict latents, dont fine-tune encoder/decoder."}
    )
    freeze_und: bool = field(
        default=False,
        metadata={"help": "Freeze the visual understanding connector layers."}
    )
    copy_init_moe: bool = field(
        default=True,
        metadata={"help": "Duplicate initial MoE experts so each has identical initialisation."}
    )
    use_flex: bool = field(
        default=True,
        metadata={"help": "Enable FLEX (flash-ext friendly) packing algorithm for sequence data."}
    )
assert torch.cuda.is_available()
parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
model_args, data_args, training_args, _ = parser.parse_args_into_dataclasses(return_remaining_strings=True)


class DataConfig:
    def __init__(
        self, 
        grouped_datasets, 
        text_cond_dropout_prob=0.1,
        vit_cond_dropout_prob=0.4,
        vae_cond_dropout_prob=0.1,
        vae_image_downsample=16,
        max_latent_size=32,
        vit_patch_size=14,
        max_num_patch_per_side=70,
    ):
        self.grouped_datasets = grouped_datasets
        self.text_cond_dropout_prob = text_cond_dropout_prob
        self.vit_cond_dropout_prob = vit_cond_dropout_prob
        self.vit_patch_size = vit_patch_size
        self.max_num_patch_per_side = max_num_patch_per_side
        self.vae_cond_dropout_prob = vae_cond_dropout_prob
        self.vae_image_downsample = vae_image_downsample
        self.max_latent_size = max_latent_size






def create_sinusoidal_pos_embedding(
    time: torch.tensor, dimension: int, min_period: float, max_period: float, device="cpu"
) -> Tensor:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    dtype = get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    # Compute the outer product
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    pos_emb = torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)
    return pos_emb


def sample_beta(alpha, beta, bsize, device):
    gamma1 = torch.empty((bsize,), device=device).uniform_(0, 1).pow(1 / alpha)
    gamma2 = torch.empty((bsize,), device=device).uniform_(0, 1).pow(1 / beta)
    return gamma1 / (gamma1 + gamma2)


def make_att_2d_masks(pad_masks, att_masks):
    """Copied from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` int[B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: int32[B, N] mask that's 1 where previous tokens cannot depend on
        it and 0 where it shares the same attention mask as the previous token.
    """
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    att_2d_masks = att_2d_masks & pad_2d_masks
    return att_2d_masks


def resize_with_pad(img, width, height, pad_value=-1):
    # assume no-op when width height fits already
    if img.ndim != 4:
        raise ValueError(f"(b,c,h,w) expected, but {img.shape}")

    cur_height, cur_width = img.shape[2:]

    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    resized_img = F.interpolate(
        img, size=(resized_height, resized_width), mode="bilinear", align_corners=False
    )

    pad_height = max(0, int(height - resized_height))
    pad_width = max(0, int(width - resized_width))

    # pad on left and top of image
    padded_img = F.pad(resized_img, (pad_width, 0, pad_height, 0), value=pad_value)
    return padded_img


def pad_vector(vector, new_dim):
    """Can be (batch_size x sequence_length x features_dimension)
    or (batch_size x features_dimension)
    """
    if vector.shape[-1] == new_dim:
        return vector
    shape = list(vector.shape)
    current_dim = shape[-1]
    shape[-1] = new_dim
    new_vector = torch.zeros(*shape, dtype=vector.dtype, device=vector.device)
    new_vector[..., :current_dim] = vector
    return new_vector


def normalize(x, min_val, max_val):
    return (x - min_val) / (max_val - min_val)


def unnormalize(x, min_val, max_val):
    return x * (max_val - min_val) + min_val


def safe_arcsin(value):
    # This ensures that the input stays within
    # [−1,1] to avoid invalid values for arcsin
    return torch.arcsin(torch.clamp(value, -1.0, 1.0))


def aloha_gripper_to_angular(value):
    # Aloha transforms the gripper positions into a linear space. The following code
    # reverses this transformation to be consistent with pi0 which is pretrained in
    # angular space.
    #
    # These values are coming from the Aloha code:
    # PUPPET_GRIPPER_POSITION_OPEN, PUPPET_GRIPPER_POSITION_CLOSED
    value = unnormalize(value, min_val=0.01844, max_val=0.05800)

    # This is the inverse of the angular to linear transformation inside the Interbotix code.
    def linear_to_radian(linear_position, arm_length, horn_radius):
        value = (horn_radius**2 + linear_position**2 - arm_length**2) / (2 * horn_radius * linear_position)
        return safe_arcsin(value)

    # The constants are taken from the Interbotix code.
    value = linear_to_radian(value, arm_length=0.036, horn_radius=0.022)

    # Normalize to [0, 1].
    # The values 0.4 and 1.5 were measured on an actual Trossen robot.
    return normalize(value, min_val=0.4, max_val=1.5)


def aloha_gripper_from_angular(value):
    # Convert from the gripper position used by pi0 to the gripper position that is used by Aloha.
    # Note that the units are still angular but the range is different.

    # The values 0.4 and 1.5 were measured on an actual Trossen robot.
    value = unnormalize(value, min_val=0.4, max_val=1.5)

    # These values are coming from the Aloha code:
    # PUPPET_GRIPPER_JOINT_OPEN, PUPPET_GRIPPER_JOINT_CLOSE
    return normalize(value, min_val=-0.6213, max_val=1.4910)


def aloha_gripper_from_angular_inv(value):
    # Directly inverts the gripper_from_angular function.
    value = unnormalize(value, min_val=-0.6213, max_val=1.4910)
    return normalize(value, min_val=0.4, max_val=1.5)


class PI0Policy(PreTrainedPolicy):
    """Wrapper class around PI0FlowMatching model to train and run inference within LeRobot."""

    config_class = PI0Config
    name = "pi0"

    def __init__(
        self,
        config: PI0Config,
        dataset_stats: dict[str, dict[str, Tensor]] | None = None,
    ):
        """
        Args:
            config: Policy configuration class instance or None, in which case the default instantiation of
                    the configuration class is used.
            dataset_stats: Dataset statistics to be used for normalization. If not passed here, it is expected
                that they will be passed with a call to `load_state_dict` before the policy is used.
        """

        super().__init__(config)
        config.validate_features()
        self.config = config
        self.remove_pi0 = config.remove_pi0 if config.remove_pi0 is not None else False
        self.normalize_inputs = Normalize(config.input_features, config.normalization_mapping, dataset_stats)
        self.normalize_targets = Normalize(
            config.output_features, config.normalization_mapping, dataset_stats
        )
        self.unnormalize_outputs = Unnormalize(
            config.output_features, config.normalization_mapping, dataset_stats
        )
        if not self.remove_pi0:
            self.language_tokenizer = AutoTokenizer.from_pretrained("google/paligemma-3b-pt-224")
        self.model = PI0FlowMatching(config)
        self.reset()

    def reset(self):
        """This should be called whenever the environment is reset."""
        self._action_queue = deque([], maxlen=self.config.n_action_steps)

    def get_optim_params(self) -> dict:
        return self.parameters()

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        """Override the from_pretrained method to display important disclaimer."""
        print(
            "⚠️  DISCLAIMER: The PI0 model is ported from JAX by the Hugging Face team. \n"
            "   It is not expected to perform as well as the original implementation. \n"
            "   Original implementation: https://github.com/Physical-Intelligence/openpi"
        )
        return super().from_pretrained(*args, **kwargs)

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        """Predict a chunk of actions given environment observations."""
        raise NotImplementedError("Currently not implemented for PI0")

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        """Select a single action given environment observations.

        This method wraps `select_actions` in order to return one action at a time for execution in the
        environment. It works by managing the actions in a queue and only calling `select_actions` when the
        queue is empty.
        """
        self.eval()

        if self.config.adapt_to_pi_aloha:
            batch[OBS_STATE] = self._pi_aloha_decode_state(batch[OBS_STATE])

        batch = self.normalize_inputs(batch)

        # Action queue logic for n_action_steps > 1. When the action_queue is depleted, populate it by
        # querying the policy.
        if len(self._action_queue) == 0:
            if not self.remove_pi0:
                images, img_masks = self.prepare_images(batch)
                lang_tokens, lang_masks = self.prepare_language(batch)
            else:
                images, img_masks, lang_tokens, lang_masks = None, None, None, None
            state = self.prepare_state(batch)

            actions, predict_image = self.model.sample_actions(
                images, img_masks, lang_tokens, lang_masks, state, noise=noise, batch=batch, unnormalize_outputs=self.unnormalize_outputs
            )

            # Unpad actions
            original_action_dim = self.config.action_feature.shape[0]
            actions = actions[:, :, :original_action_dim]

            actions = self.unnormalize_outputs({"action": actions})["action"]

            if self.config.adapt_to_pi_aloha:
                actions = self._pi_aloha_encode_actions(actions)

            # `self.model.forward` returns a (batch_size, n_action_steps, action_dim) tensor, but the queue
            # effectively has shape (n_action_steps, batch_size, *), hence the transpose.
            ## TODO:
            self._action_queue.extend(actions.transpose(0, 1))# [:10])
        return self._action_queue.popleft(), predict_image

    def forward(self, batch: dict[str, Tensor], noise=None, time=None) -> tuple[Tensor, dict[str, Tensor]]:
        """Do a full training forward pass to compute the loss"""
        if self.config.adapt_to_pi_aloha:
            batch[OBS_STATE] = self._pi_aloha_decode_state(batch[OBS_STATE])
            batch[ACTION] = self._pi_aloha_encode_actions_inv(batch[ACTION])

        batch = self.normalize_inputs(batch)
        batch = self.normalize_targets(batch)
        state = self.prepare_state(batch)

        if not self.remove_pi0:
            images, img_masks = self.prepare_images(batch)
            lang_tokens, lang_masks = self.prepare_language(batch)
        else:
            images, img_masks, lang_tokens, lang_masks = None, None, None, None

        actions = self.prepare_action(batch)
        actions_is_pad = batch.get("action_is_pad")

        loss_dict = {}
        loss_dict, losses = self.model.forward(images, img_masks, lang_tokens, lang_masks, state, actions, noise, time, unnormalize_outputs=self.unnormalize_outputs, batch=batch)
        
        action_mse = loss_dict['action_mse'].clone()
        
        if actions_is_pad is not None:
            in_episode_bound = ~actions_is_pad
            action_mse = action_mse * in_episode_bound.unsqueeze(-1)

        # Remove padding
        action_mse = action_mse[:, :, : self.config.max_action_dim]

        # For backward pass
        action_mse = action_mse.mean()
        loss_dict['action_mse'] = action_mse.clone().detach()
        # For logging
        losses += action_mse
        
        return losses, loss_dict

    def prepare_images(self, batch):
        """Apply Pi0 preprocessing to the images, like resizing to 224x224 and padding to keep aspect ratio, and
        convert pixel range from [0.0, 1.0] to [-1.0, 1.0] as requested by SigLIP.
        """
        images = []
        img_masks = []

        present_img_keys = [key for key in self.config.image_features if key in batch]
        missing_img_keys = [key for key in self.config.image_features if key not in batch]

        if len(present_img_keys) == 0:
            raise ValueError(
                f"All image features are missing from the batch. At least one expected. (batch: {batch.keys()}) (image_features:{self.config.image_features})"
            )

        # Preprocess image features present in the batch
        for key in present_img_keys:
            img = batch[key]

            if self.config.resize_imgs_with_padding is not None:
                img = resize_with_pad(img, *self.config.resize_imgs_with_padding, pad_value=0)

            # Normalize from range [0,1] to [-1,1] as expected by siglip
            img = img * 2.0 - 1.0

            bsize = img.shape[0]
            device = img.device
            mask = torch.ones(bsize, dtype=torch.bool, device=device)
            images.append(img)
            img_masks.append(mask)

        # Create image features not present in the batch
        # as fully 0 padded images.
        for num_empty_cameras in range(len(missing_img_keys)):
            if num_empty_cameras >= self.config.empty_cameras:
                break
            img = torch.ones_like(img) * -1
            mask = torch.zeros_like(mask)
            images.append(img)
            img_masks.append(mask)

        return images, img_masks

    def prepare_language(self, batch) -> tuple[Tensor, Tensor]:
        """Tokenize the text input"""
        device = batch[OBS_STATE].device
        tasks = batch["task"]

        # PaliGemma prompt has to end with a new line
        tasks = [task if task.endswith("\n") else f"{task}\n" for task in tasks]

        tokenized_prompt = self.language_tokenizer.__call__(
            tasks,
            padding="max_length",
            padding_side="right",
            max_length=self.config.tokenizer_max_length,
            return_tensors="pt",
        )
        lang_tokens = tokenized_prompt["input_ids"].to(device=device)
        lang_masks = tokenized_prompt["attention_mask"].to(device=device, dtype=torch.bool)

        return lang_tokens, lang_masks

    def _pi_aloha_decode_state(self, state):
        # Flip the joints.
        for motor_idx in [1, 2, 8, 9]:
            state[:, motor_idx] *= -1
        # Reverse the gripper transformation that is being applied by the Aloha runtime.
        for motor_idx in [6, 13]:
            state[:, motor_idx] = aloha_gripper_to_angular(state[:, motor_idx])
        return state

    def _pi_aloha_encode_actions(self, actions):
        # Flip the joints.
        for motor_idx in [1, 2, 8, 9]:
            actions[:, :, motor_idx] *= -1
        # Reverse the gripper transformation that is being applied by the Aloha runtime.
        for motor_idx in [6, 13]:
            actions[:, :, motor_idx] = aloha_gripper_from_angular(actions[:, :, motor_idx])
        return actions

    def _pi_aloha_encode_actions_inv(self, actions):
        # Flip the joints again.
        for motor_idx in [1, 2, 8, 9]:
            actions[:, :, motor_idx] *= -1
        # Reverse the gripper transformation that is being applied by the Aloha runtime.
        for motor_idx in [6, 13]:
            actions[:, :, motor_idx] = aloha_gripper_from_angular_inv(actions[:, :, motor_idx])
        return actions

    def prepare_state(self, batch):
        """Pad state"""
        state = pad_vector(batch[OBS_STATE], self.config.max_state_dim)
        return state

    def prepare_action(self, batch):
        """Pad action"""
        actions = pad_vector(batch[ACTION], self.config.max_action_dim)
        return actions


class PI0FlowMatching(nn.Module):
    """
    π0: A Vision-Language-Action Flow Model for General Robot Control

    [Paper](https://www.physicalintelligence.company/download/pi0.pdf)
    [Jax code](https://github.com/Physical-Intelligence/openpi)

    Designed by Physical Intelligence. Ported from Jax by Hugging Face.
    ┌──────────────────────────────┐
    │               actions        │
    │               ▲              │
    │              ┌┴─────┐        │
    │  kv cache    │Gemma │        │
    │  ┌──────────►│Expert│        │
    │  │           │      │        │
    │ ┌┴────────┐  │x 10  │        │
    │ │         │  └▲──▲──┘        │
    │ │PaliGemma│   │  │           │
    │ │         │   │  robot state │
    │ │         │   noise          │
    │ └▲──▲─────┘                  │
    │  │  │                        │
    │  │  image(s)                 │
    │  language tokens             │
    └──────────────────────────────┘
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.merge_bagel = config.merge_bagel if config.merge_bagel is not None else False
        self.pi0_keep_ratio = config.pi0_keep_ratio if config.pi0_keep_ratio is not None else False
        self.remove_pi0 = config.remove_pi0 if config.remove_pi0 is not None else False

        if self.merge_bagel:
            assert self.pi0_keep_ratio is not None
            llm_config = Qwen2Config.from_json_file(os.path.join(model_args.model_path, "llm_config.json"))
            llm_config.layer_module = model_args.layer_module
            llm_config.qk_norm = model_args.llm_qk_norm
            llm_config.tie_word_embeddings = model_args.tie_word_embeddings
            llm_config.freeze_und = training_args.freeze_und
            language_model = Qwen2ForCausalLM(llm_config, visual_gen = training_args.visual_gen)
            if training_args.copy_init_moe:
                language_model.init_moe()
            if training_args.visual_und:  
                vit_config = SiglipVisionConfig.from_json_file(os.path.join(model_args.model_path, "vit_config.json"))
                vit_config.num_hidden_layers = vit_config.num_hidden_layers + 1 + model_args.vit_select_layer
                vit_config.rope = model_args.vit_rope
                vit_model = SiglipVisionModel(vit_config)

            vae_model, vae_config = load_ae(
                local_path=os.path.join(model_args.model_path, "ae.safetensors") 
            )
            self.vae_config = vae_config

            self.bagel_config = BagelConfig(
                visual_gen=training_args.visual_gen,
                visual_und=training_args.visual_und,
                llm_config=llm_config, 
                vit_config=vit_config if training_args.visual_und else None,
                vae_config=vae_config,
                latent_patch_size=model_args.latent_patch_size,
                max_latent_size=model_args.max_latent_size,
                vit_max_num_patch_per_side=model_args.vit_max_num_patch_per_side,
                connector_act=model_args.connector_act,
                interpolate_pos=model_args.interpolate_pos,
                timestep_shift=training_args.timestep_shift,
            )
            self.bagel_config.chunk_size = config.chunk_size
            bagel_model = Bagel(
                language_model, 
                vit_model if training_args.visual_und else None, 
                self.bagel_config,
            )
            self.bagel_model = bagel_model
            if training_args.visual_und:
                bagel_model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config)
            model_state_dict_path = os.path.join(model_args.model_path, "ema.safetensors")
            model_state_dict = load_file(model_state_dict_path, device="cpu")
            msg = bagel_model.load_state_dict(model_state_dict, strict=False)
            print(f"load Bagel: {msg}")

            tokenizer = Qwen2Tokenizer.from_pretrained(model_args.model_path)
            tokenizer, new_token_ids, num_new_tokens = add_special_tokens(tokenizer)
            self.new_token_ids = new_token_ids
            if num_new_tokens > 0:
                bagel_model.language_model.resize_token_embeddings(len(tokenizer))
                bagel_model.config.llm_config.vocab_size = len(tokenizer)
                bagel_model.language_model.config.vocab_size = len(tokenizer)
        
            # TODO: fix bagel
            if training_args.action_gen:
                for name, param in bagel_model.named_parameters():
                    param.requires_grad = True

            if training_args.freeze_vae and training_args.visual_gen:
                for param in vae_model.parameters():
                    param.requires_grad = False
            self.vae_model = vae_model
            # if training_args.freeze_llm:
            #     bagel_model.language_model.eval()
            #     for param in bagel_model.language_model.parameters():
            #         param.requires_grad = False

            if training_args.freeze_vit and training_args.visual_und:
                bagel_model.vit_model.eval()
                for param in bagel_model.vit_model.parameters():
                    param.requires_grad = False
            
            # Setup packed dataloader
            with open(data_args.dataset_config_file, "r") as stream:
                dataset_meta = yaml.safe_load(stream)
            dataset_config = DataConfig(grouped_datasets=dataset_meta)
            if training_args.visual_und:
                dataset_config.vit_patch_size = model_args.vit_patch_size
                dataset_config.max_num_patch_per_side = model_args.vit_max_num_patch_per_side
            vae_image_downsample = model_args.latent_patch_size * vae_config.downsample
            dataset_config.vae_image_downsample = vae_image_downsample
            dataset_config.max_latent_size = model_args.max_latent_size
            dataset_config.text_cond_dropout_prob = model_args.text_cond_dropout_prob
            dataset_config.vae_cond_dropout_prob = model_args.vae_cond_dropout_prob
            dataset_config.vit_cond_dropout_prob = model_args.vit_cond_dropout_prob
            self.dataset = PackedDataset(
                dataset_config,
                tokenizer=tokenizer,
                special_tokens=new_token_ids,
                expected_num_tokens=training_args.expected_num_tokens,
                max_num_tokens_per_sample=data_args.max_num_tokens_per_sample,
                max_num_tokens=data_args.max_num_tokens,
                max_buffer_size=data_args.max_buffer_size,
                prefer_buffer_before=data_args.prefer_buffer_before,
                interpolate_pos=model_args.interpolate_pos,
                use_flex=training_args.use_flex,
                data_status=None,
                action_dim=self.bagel_model.config.action_dim,
                action_horizon = self.config.chunk_size,
                visual_gen=training_args.visual_gen,
            )
        
        paligemma_with_expert_config = PaliGemmaWithExpertConfig(
            freeze_vision_encoder = self.config.freeze_vision_encoder,
            train_expert_only = self.config.train_expert_only,
            attention_implementation = self.config.attention_implementation,
            remove_pi0 = self.remove_pi0
        )
        self.paligemma_with_expert = PaliGemmaWithExpertModel(paligemma_with_expert_config, self.remove_pi0).float().cuda()

        self.state_proj = nn.Linear(self.config.max_state_dim, self.config.proj_width)

        self.action_time_mlp_in = nn.Linear(self.config.proj_width * 2, self.config.proj_width)
        self.action_time_mlp_out = nn.Linear(self.config.proj_width, self.config.proj_width)
        self.set_requires_grad()
        self.action_in_proj = nn.Linear(self.config.max_action_dim, self.config.proj_width)
        self.action_out_proj = nn.Linear(self.config.proj_width, self.config.max_action_dim)


    def set_requires_grad(self):
        for params in self.state_proj.parameters():
            params.requires_grad = self.config.train_state_proj

    def sample_noise(self, shape, device):
        noise = torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
        )
        return noise

    def sample_time(self, bsize, device):
        time_beta = sample_beta(1.5, 1.0, bsize, device)
        time = time_beta * 0.999 + 0.001
        return time.to(dtype=torch.float32, device=device)

    def embed_prefix(
        self, images, img_masks, lang_tokens, lang_masks, batch=None, unnormalize_outputs=None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed images with SigLIP and language tokens with embedding layer to prepare
        for PaliGemma transformer processing.
        """
        self.dtype = self.state_proj.weight.dtype
        self.paligemma_with_expert = self.paligemma_with_expert.to(self.dtype)

        if self.merge_bagel:
            datas = self.dataset(unnormalize_outputs(batch))
            data_batch = SimpleCustomBatch([datas]).cuda(f"cuda:{torch.cuda.current_device()}").to_dict()
            data_batch = autocast(data_batch, torch.float32, self.dtype)
            if training_args.visual_gen:
               with torch.no_grad():
                    data_batch['padded_latent'] = self.vae_model.encode(data_batch.pop('padded_images'))
            if "packed_action_tokens" in data_batch.keys():
                with torch.no_grad():
                    data_batch['packed_action_tokens'] = torch.tensor(data_batch['packed_action_tokens']).to(f"cuda:{torch.cuda.current_device()}")
        else:
            data_batch = None

        if self.remove_pi0:
            return data_batch, None, None, None
        # TODO: avoid list in python and torch.cat ; prefer pre-allocation with torch.empty
        embs = []
        pad_masks = []
        att_masks = []

        # TODO: remove for loop
        for (
            img,
            img_mask,
        ) in zip(images, img_masks, strict=False):
            img_emb = self.paligemma_with_expert.embed_image(img)
            # img_emb = img_emb.to(dtype=torch.bfloat16)

            # Normalize image embeddings
            img_emb_dim = img_emb.shape[-1]
            img_emb = img_emb * torch.tensor(img_emb_dim**0.5, dtype=img_emb.dtype, device=img_emb.device)

            bsize, num_img_embs = img_emb.shape[:2]
            img_mask = img_mask[:, None].expand(bsize, num_img_embs)

            embs.append(img_emb)
            pad_masks.append(img_mask)

            # Create attention masks so that image tokens attend to each other
            att_masks += [0] * num_img_embs

        lang_emb = self.paligemma_with_expert.embed_language_tokens(lang_tokens)

        # Normalize language embeddings
        lang_emb_dim = lang_emb.shape[-1]
        lang_emb = lang_emb * math.sqrt(lang_emb_dim)
        
        embs.append(lang_emb)
        pad_masks.append(lang_masks)

        # full attention between image and language inputs
        num_lang_embs = lang_emb.shape[1]
        att_masks += [0] * num_lang_embs
        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))
        return data_batch, embs, pad_masks, att_masks

    def embed_suffix(self, state, noisy_actions, timestep):
        """Embed state, noisy_actions, timestep to prepare for Expert Gemma processing."""
        embs = []
        pad_masks = []
        att_masks = []

        # Embed state
        state_emb = self.state_proj(state)
        # state_emb = state_emb.to(dtype=torch.bfloat16)
        embs.append(state_emb[:, None, :])
        bsize = state_emb.shape[0]
        dtype = state_emb.dtype
        device = state_emb.device

        state_mask = torch.ones(bsize, 1, dtype=torch.bool, device=device)
        pad_masks.append(state_mask)

        # Set attention masks so that image and language inputs do not attend to state or actions
        att_masks += [1]

        # Embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = create_sinusoidal_pos_embedding(
            timestep, self.config.proj_width, min_period=4e-3, max_period=4.0, device=device
        )
        time_emb = time_emb.type(dtype=dtype)

        # Fuse timestep + action information using an MLP
        action_emb = self.action_in_proj(noisy_actions)

        time_emb = time_emb[:, None, :].expand_as(action_emb)
        action_time_emb = torch.cat([action_emb, time_emb], dim=2)

        action_time_emb = self.action_time_mlp_in(action_time_emb)
        action_time_emb = F.silu(action_time_emb)  # swish == silu
        action_time_emb = self.action_time_mlp_out(action_time_emb)

        # Add to input tokens
        embs.append(action_time_emb)

        bsize, action_time_dim = action_time_emb.shape[:2]
        action_time_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=device)
        pad_masks.append(action_time_mask)

        # Set attention masks so that image, language and state inputs do not attend to action tokens
        att_masks += [1] + ([0] * (self.config.n_action_steps - 1))

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks

    def forward(
        self, images, img_masks, lang_tokens, lang_masks, state, actions, noise=None, time=None, unnormalize_outputs=None, batch=None
    ) -> Tensor:
        """Do a full training forward pass and compute the loss (batch_size x num_steps x num_motors)"""
        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)

        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions
    
        data_batch, prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, batch, unnormalize_outputs
        )
        state = state.to(self.dtype)
        x_t = x_t.to(self.dtype)
        time = time.to(self.dtype)
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(state, x_t, time)
        if self.merge_bagel:
            past_key_values = NaiveCache(self.bagel_model.config.llm_config.num_hidden_layers)
            if self.bagel_model.config.visual_gen:
                visual_gen_complete = np.random.rand() < 0.5
            ret = self.bagel_model(**data_batch, past_key_values=past_key_values, visual_gen_complete=visual_gen_complete)
            sample_lens = data_batch['sample_lens'][:-1]
            max_sample_lens = max(sample_lens)
            bagel_pad_masks = []
            bagel_att_masks = []
            for batch_id in range(len(sample_lens)):
                bagel_pad_mask = torch.from_numpy(np.ones(max_sample_lens)).long().cuda()
                bagel_pad_mask[sample_lens[batch_id]:] = 0
                bagel_pad_masks.append(bagel_pad_mask)
                bagel_att_mask = torch.zeros((max_sample_lens,)).long().cuda()
                bagel_att_masks.append(bagel_att_mask)
            bagel_pad_masks = torch.stack(bagel_pad_masks, dim=0).bool()
            bagel_att_masks = torch.stack(bagel_att_masks, dim=0)

            if not self.remove_pi0:
                prefix_pad_masks = torch.logical_and(torch.rand_like(prefix_pad_masks.float().cuda())< self.pi0_keep_ratio, prefix_pad_masks)
                position_ids =  torch.cumsum(torch.cat([bagel_pad_masks, prefix_pad_masks, suffix_pad_masks], dim=1), dim=1) - 1
                pad_masks = torch.cat([bagel_pad_masks, prefix_pad_masks, suffix_pad_masks], dim=1)
                att_masks = torch.cat([ bagel_att_masks, prefix_att_masks, suffix_att_masks], dim=1)
            else:
                position_ids = torch.cumsum(torch.cat([bagel_pad_masks, suffix_pad_masks], dim=1), dim=1) - 1
                pad_masks = torch.cat([bagel_pad_masks, suffix_pad_masks], dim=1)
                att_masks = torch.cat([bagel_att_masks, suffix_att_masks], dim=1)
        else: 
            pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
            att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
            position_ids =  torch.cumsum(pad_masks, dim=1) - 1

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        

        if self.merge_bagel:
            bagel_kv_cache = ret['past_key_values']
            bagel_sample_lens = data_batch['sample_lens']
        else:
            bagel_kv_cache = None
            bagel_sample_lens = None
        

        (_, suffix_out), _ = self.paligemma_with_expert.forward(
            attention_mask=att_2d_masks.bool(),
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, suffix_embs],
            use_cache=False,
            fill_kv_cache=False,
            bagel_kv_cache=bagel_kv_cache,
            bagel_sample_lens=bagel_sample_lens,
        )
        suffix_out = suffix_out[:, -self.config.n_action_steps :]
        # Original openpi code, upcast attention output
        suffix_out = suffix_out.to(dtype=self.dtype)
        v_t = self.action_out_proj(suffix_out)
        
        action_losses = F.mse_loss(u_t.float(), v_t.float(), reduction="none")
        ## TODO:
        ce = None
        loss_dict = {} 
        loss = torch.tensor(0).float().cuda()
        if self.merge_bagel:
            if ce is not None:
                total_ce_tokens = torch.tensor(len(data_batch['ce_loss_indexes']), device=device)
                if training_args.ce_loss_reweighting:
                    ce = ce * ce_loss_weights
                    total_ce_loss_weights = ce_loss_weights.sum()
                    ce = ce.sum() / total_ce_loss_weights
                else:
                    ce = ce.sum() / total_ce_tokens
                loss_dict["ce"] = ce.detach()
                loss = loss + ce * self.bagel_model.config.ce_weight
            else:
                loss_dict["ce"] = torch.tensor(0).cuda().float()
                total_ce_tokens = torch.tensor(0).cuda().float()

            if self.bagel_model.config.visual_gen:
                total_mse_tokens = torch.tensor(len(data_batch['mse_loss_indexes'])).cuda()
                mse = ret['mse'].clone()
                mse = mse.mean(dim=-1).sum() / total_mse_tokens
                loss_dict["mse"] = mse.detach()
                loss = loss + mse * self.bagel_model.config.mse_weight
            else:
                loss_dict["mse"] = torch.tensor(0).cuda()
                total_mse_tokens = torch.tensor(0).cuda()
            if self.bagel_model.config.visual_gen:
                if not visual_gen_complete:
                    action_losses = action_losses.detach()
            loss_dict['action_mse'] = action_losses
        return loss_dict, loss

    def sample_actions(self, images, img_masks, lang_tokens, lang_masks, state, noise=None, batch=None, unnormalize_outputs=None) -> Tensor:
        """Do a full inference forward and compute the action (batch_size x num_steps x num_motors)"""
        self.dtype = self.state_proj.weight.dtype
        device = torch.cuda.current_device()
        # '''
        if self.merge_bagel:
            new_token_ids = self.new_token_ids
            if isinstance(new_token_ids, dict):
                for k, v in new_token_ids.items():
                    if torch.is_tensor(v):
                        new_token_ids[k] = v.to(device)
            elif torch.is_tensor(new_token_ids):
                new_token_ids = new_token_ids.to(device)

            # prefill
            past_key_values = NaiveCache(self.bagel_model.config.llm_config.num_hidden_layers)
            newlens = [0]
            new_rope = [0]
            observation_images = []
            for key in batch.keys():
                if "images." in key and "observation" in key:
                    image_np = (batch[key][0].detach().cpu().numpy().transpose((1, 2, 0)) * 255).astype(np.uint8)
                    observation_images.append(image_np)
                    image = Image.fromarray(image_np)
                    # add images
                    generation_input, newlens, new_rope = self.bagel_model.prepare_vit_images(
                        curr_kvlens=newlens,
                        curr_rope=new_rope,
                        images=[image],
                        transforms=self.dataset.dataset.vit_transform,
                        new_token_ids=new_token_ids,
                    )

                    for k, v in generation_input.items():
                        if torch.is_tensor(v):
                            generation_input[k] = v.to(device)
                    generation_input = autocast(generation_input, torch.float32, self.dtype)
                    past_key_values = self.bagel_model.forward_cache_update_vit(past_key_values, **generation_input)
            observation_image = cv2.hconcat(observation_images)
            '''
            observation_images = []
            for key in batch.keys():
                if "images." in key and "observation" in key:
                    observation_images.append((batch[key][0].detach().cpu().numpy().transpose((1, 2, 0)) * 255).astype(np.uint8))
            observation_image = cv2.hconcat(observation_images)
            image = Image.fromarray(observation_image)
            generation_input, newlens, new_rope = self.bagel_model.prepare_vit_images(
                curr_kvlens=newlens,
                curr_rope=new_rope,
                images=[image],
                transforms=self.dataset.dataset.vit_transform,
                new_token_ids=new_token_ids,
            )

            for k, v in generation_input.items():
                if torch.is_tensor(v):
                    generation_input[k] = v.to(device)
            generation_input = autocast(generation_input, torch.float32, self.dtype)
            past_key_values = self.bagel_model.forward_cache_update_vit(past_key_values, **generation_input)
            '''
            
            # add text
            prompt = "Instruction:" + batch['task'][0] + "."
            generation_input, newlens, new_rope = self.bagel_model.prepare_prompts(
                curr_kvlens=newlens,
                curr_rope=new_rope, 
                prompts=[prompt],
                tokenizer=self.dataset.tokenizer, 
                new_token_ids=new_token_ids,
            )
            for k, v in generation_input.items():
                if torch.is_tensor(v):
                    generation_input[k] = v.to(device)
            past_key_values = self.bagel_model.forward_cache_update_text(past_key_values, **generation_input)

            # TODO: decode for text generation
            # generation_input = self.prepare_start_tokens(newlens, new_rope, new_token_ids)
            # for k, v in generation_input.items():
            #     if torch.is_tensor(v):
            #         generation_input[k] = v.to(device)
            # unpacked_latent = self.generate_text(
            #     past_key_values=past_key_values,
            #     max_length=max_length,
            #     do_sample=do_sample,
            #     temperature=temperature,
            #     end_token_id=new_token_ids['eos_token_id'],
            #     **generation_input,
            # )
            # output = tokenizer.decode(unpacked_latent[:,0])
            # output = output.split('<|im_end|>')[0].split('<|im_start|>')[1]
            
            import ipdb;ipdb.set_trace()
            if training_args.visual_gen:
                resolution = tuple(observation_image.shape[:2])
                generation_input, newlens, new_rope = self.bagel_model.prepare_vae_latent(
                    curr_kvlens=newlens,
                    curr_rope=new_rope, 
                    image_sizes=[resolution], 
                    new_token_ids=new_token_ids,
                )
                for k, v in generation_input.items():
                    if torch.is_tensor(v):
                        generation_input[k] = v.to(device)
                cfg_past_key_values = NaiveCache(self.bagel_model.config.llm_config.num_hidden_layers)
                cfg_newlens = [0]
                cfg_new_rope = [0]
                generation_input_cfg = self.bagel_model.prepare_vae_latent_cfg(
                    curr_kvlens=cfg_newlens,
                    curr_rope=cfg_new_rope, 
                    image_sizes=[resolution],
                )
                for k, v in generation_input_cfg.items():
                    if torch.is_tensor(v):
                        generation_input_cfg[k] = v.to(device)
                num_timesteps = 24 ## TODO: set timesteps here
                cfg_scale = 1
                cfg_interval = [0., 1.]
                timestep_shift = 1.0
                cfg_renorm_min = 0.0
                unpacked_latent, past_key_values = self.bagel_model.generate_image(
                    past_key_values=past_key_values,
                    num_timesteps=num_timesteps,
                    cfg_text_scale=cfg_scale,
                    cfg_interval=cfg_interval,
                    cfg_renorm_min=cfg_renorm_min,
                    timestep_shift=timestep_shift,
                    cfg_text_past_key_values=cfg_past_key_values,
                    cfg_text_packed_position_ids=generation_input_cfg["cfg_packed_position_ids"],
                    cfg_text_key_values_lens=generation_input_cfg["cfg_key_values_lens"],
                    cfg_text_packed_query_indexes=generation_input_cfg["cfg_packed_query_indexes"],
                    cfg_text_packed_key_value_indexes=generation_input_cfg["cfg_packed_key_value_indexes"],
                    **generation_input,
                )
                image_list = []
                for latent in unpacked_latent:
                    latent = latent.reshape(1, resolution[0]//16, resolution[1]//16, 2, 2, 16)
                    latent = torch.einsum("nhwpqc->nchpwq", latent)
                    latent = latent.reshape(1, 16, resolution[0]//8, resolution[1]//8)
                    image = self.vae_model.decode(latent.to(device))
                    tmpimage = ((image * 0.5 + 0.5).clamp(0, 1)[0].permute(1, 2, 0) * 255).to(torch.uint8).cpu().numpy()
                    tmpimage = Image.fromarray(tmpimage)
                    image_list.append(tmpimage)
                predict_images = image_list
                import ipdb;ipdb.set_trace()
            else:
                predict_images = None
            past_key_values.key_cache = past_key_values.key_unnorm_cache
            bagel_kv_cache = past_key_values
            bagel_sample_lens = [newlens[-1], -1]
        else:
            bagel_kv_cache = None
            bagel_sample_lens = None
            predict_images = None
        # '''

        #################
        '''
        if self.merge_bagel:
            self.bagel_model.train()
            data_batch, prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks, batch, unnormalize_outputs )
            past_key_values = NaiveCache(self.bagel_model.config.llm_config.num_hidden_layers)
            ret = self.bagel_model(**data_batch, past_key_values=past_key_values)
            sample_lens = data_batch['sample_lens'][:-1]
            max_sample_lens = max(sample_lens)
            bagel_pad_masks = []
            bagel_att_masks = []
            for batch_id in range(len(sample_lens)):
                bagel_pad_mask = torch.from_numpy(np.ones(max_sample_lens)).long().cuda()
                bagel_pad_mask[sample_lens[batch_id]:] = 0
                bagel_pad_masks.append(bagel_pad_mask)
                bagel_att_mask = torch.zeros((max_sample_lens,)).long().cuda()
                bagel_att_masks.append(bagel_att_mask)
            bagel_pad_masks = torch.stack(bagel_pad_masks, dim=0).bool()
            bagel_att_masks = torch.stack(bagel_att_masks, dim=0)
            bagel_kv_cache = ret['past_key_values']
            bagel_sample_lens = data_batch['sample_lens']
        else:
            bagel_kv_cache = None
            bagel_sample_lens = None
            predict_images = None 
        '''
        ################


        bsize = 1
        device = "cuda"

        if noise is None:
            actions_shape = (bsize, self.config.n_action_steps, self.config.max_action_dim)
            noise = self.sample_noise(actions_shape, device)
        data_batch, prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, batch, unnormalize_outputs
        )

        if self.merge_bagel:
            # '''
            bagel_pad_masks = []
            bagel_att_masks = []
            batch_id = 0
            max_sample_lens = newlens[-1]
            bagel_pad_mask = torch.from_numpy(np.ones(max_sample_lens)).long().cuda()
            bagel_pad_masks.append(bagel_pad_mask)
            bagel_att_mask = torch.zeros((max_sample_lens,)).long().cuda()
            bagel_att_masks.append(bagel_att_mask)
            bagel_pad_masks = torch.stack(bagel_pad_masks, dim=0)
            bagel_att_masks = bagel_att_masks = torch.stack(bagel_att_masks, dim=0)
            # '''
            if not self.remove_pi0:
               #  bagel_pad_masks = torch.logical_and(torch.rand_like(bagel_pad_masks.float().cuda()) < (1 - self.pi0_keep_ratio), bagel_pad_masks) ##TODO:
                prefix_pad_masks = torch.logical_and(torch.rand_like(prefix_pad_masks.float().cuda()) < self.pi0_keep_ratio, prefix_pad_masks) 

                prefix_position_ids = torch.cumsum(torch.cat([bagel_pad_masks, prefix_pad_masks], dim=1), dim=1) - 1
                prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
                prefix_pad_masks = torch.cat([bagel_pad_masks, prefix_pad_masks], dim=1)
                prefix_att_masks = torch.cat([bagel_att_masks, prefix_att_masks], dim=1)
            else:
                prefix_position_ids = torch.cumsum(bagel_pad_masks, dim=1) - 1
                prefix_pad_masks = bagel_pad_masks
                prefix_att_masks = bagel_att_masks
                prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        else:
            prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
            prefix_offsets = None

        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        
        # Compute image and language key value cache
        _, past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks.bool(),
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=self.config.use_cache,
            bagel_kv_cache=bagel_kv_cache,
            bagel_sample_lens=bagel_sample_lens,
            fill_kv_cache=True,
        )

        dt = -1.0 / self.config.num_steps
        dt = torch.tensor(dt, dtype=torch.float32, device=device)
        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        while time >= -dt / 2:
            expanded_time = time.expand(bsize)
            v_t = self.denoise_step(
                state,
                prefix_pad_masks,
                past_key_values,
                x_t,
                expanded_time,
                bagel_kv_cache,
                bagel_sample_lens,
                prefix_offsets,
            )

            # Euler step
            x_t += dt * v_t
            time += dt
        return x_t, predict_image

    def denoise_step(
        self,
        state,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
        bagel_kv_cache,
        bagel_sample_lens,
        prefix_offsets,
    ):
        """Apply one denoising step of the noise `x_t` at a given timestep."""
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(state, x_t, timestep)

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]
        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)

        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)

        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        if prefix_offsets is None:
            prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]

        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1
        outputs_embeds, _ = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks.bool(),
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=self.config.use_cache,
            fill_kv_cache=False,
            bagel_kv_cache=None,
            bagel_sample_lens=None,
        )
        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.n_action_steps :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        v_t = self.action_out_proj(suffix_out)
        return v_t
