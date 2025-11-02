# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0


import random
import json

import numpy as np
import torch

from lerobot.utils.data_utils import (
    get_flattened_position_ids_interpolate,
    get_flattened_position_ids_extrapolate, 
    len2weight,
    patchify, 
    prepare_attention_mask_per_sample, 
)
from PIL import Image, ImageFile, PngImagePlugin
from torchvision.transforms import InterpolationMode
from torchvision import transforms
from torchvision.transforms import functional as F
import torch
from lerobot.utils.data_utils import pil_img2rgb
import cv2
from lerobot.constants import ACTION, OBS_STATE


Image.MAX_IMAGE_PIXELS = 200000000
ImageFile.LOAD_TRUNCATED_IMAGES = True
MaximumDecompressedSize = 1024
MegaByte = 2 ** 20
PngImagePlugin.MAX_TEXT_CHUNK = MaximumDecompressedSize * MegaByte

def read_frames_decord(video_path, num_frames, sample='rand', fix_start=None, clip=None, min_num_frames=4):
    video_reader = decord.VideoReader(video_path, num_threads=1)
    vlen = len(video_reader)
    fps = video_reader.get_avg_fps()
    duration = vlen / float(fps)
    if clip:
        start, end = clip
        duration = end - start
        vlen = int(duration * fps)
        start_index = int(start * fps)

    t_num_frames = np.random.randint(min_num_frames, num_frames + 1)

    frame_indices = get_frame_indices(
        t_num_frames, vlen, sample=sample, fix_start=fix_start,
        input_fps=fps
    )
    if clip:
        frame_indices = [f + start_index for f in frame_indices]
    frames = video_reader.get_batch(frame_indices).asnumpy()  # (T, H, W, C), np.uint8
    frames = [Image.fromarray(frames[i]) for i in range(frames.shape[0])]
    return frames


def extract_frame_number(filename):
    # Extract the numeric part from the filename using regular expressions
    match = re.search(r'_(\d+).jpg$', filename)
    return int(match.group(1)) if match else -1


def sort_frames(frame_paths):
    # Extract filenames from each path and sort by their numeric part
    return sorted(frame_paths, key=lambda x: extract_frame_number(os.path.basename(x)))


def read_frames_folder(video_path, num_frames, sample='rand', fix_start=None, min_num_frames=4):
    image_list = sort_frames(list(os.listdir(video_path)))
    frames = []
    for image in image_list:
        fp = os.path.join(video_path, image)
        frame = Image.open(fp).convert('RGB')
        frames.append(frame)
    vlen = len(frames)

    t_num_frames = np.random.randint(min_num_frames, num_frames + 1)

    if vlen > t_num_frames:
        frame_indices = get_frame_indices(
            t_num_frames, vlen, sample=sample, fix_start=fix_start
        )
        frames = [frames[i] for i in frame_indices]
    return frames

class FrameSampler:
    def __init__(self, max_num_frames=-1, min_num_frames=8, sample='rand'):
        self.max_num_frames = max_num_frames
        self.min_num_frames = min_num_frames
        self.sample = sample
    
    def __call__(self, file_name):
        fn = read_frames_folder if file_name.endswith('/') else read_frames_decord
        frames = fn(file_name, num_frames=self.max_num_frames, min_num_frames=self.min_num_frames, sample=self.sample)
        return frames

class MaxLongEdgeMinShortEdgeResize(torch.nn.Module):
    """Resize the input image so that its longest side and shortest side are within a specified range,
    ensuring that both sides are divisible by a specified stride.

    Args:
        max_size (int): Maximum size for the longest edge of the image.
        min_size (int): Minimum size for the shortest edge of the image.
        stride (int): Value by which the height and width of the image must be divisible.
        max_pixels (int): Maximum pixels for the full image.
        interpolation (InterpolationMode): Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`. Default is ``InterpolationMode.BILINEAR``.
            If input is Tensor, only ``InterpolationMode.NEAREST``, ``InterpolationMode.NEAREST_EXACT``,
            ``InterpolationMode.BILINEAR``, and ``InterpolationMode.BICUBIC`` are supported.
            The corresponding Pillow integer constants, e.g., ``PIL.Image.BILINEAR`` are also accepted.
        antialias (bool, optional): Whether to apply antialiasing (default is True).
    """

    def __init__(
        self, 
        max_size: int, 
        min_size: int, 
        stride: int, 
        max_pixels: int,
        interpolation=InterpolationMode.BICUBIC, 
        antialias=True
    ):
        super().__init__()
        self.max_size = max_size
        self.min_size = min_size
        self.stride = stride
        self.max_pixels = max_pixels
        self.interpolation = interpolation
        self.antialias = antialias

    def _make_divisible(self, value, stride):
        """Ensure the value is divisible by the stride."""
        return max(stride, int(round(value / stride) * stride))

    def _apply_scale(self, width, height, scale):
        new_width = round(width * scale)
        new_height = round(height * scale)
        new_width = self._make_divisible(new_width, self.stride)
        new_height = self._make_divisible(new_height, self.stride)
        return new_width, new_height

    def forward(self, img, img_num=1):
        """
        Args:
            img (PIL Image): Image to be resized.
            img_num (int): Number of images, used to change max_tokens.
        Returns:
            PIL Image or Tensor: Rescaled image with divisible dimensions.
        """
        if isinstance(img, torch.Tensor):
            height, width = img.shape[-2:]
        else:
            width, height = img.size

        scale = min(self.max_size / max(width, height), 1.0)
        scale = max(scale, self.min_size / min(width, height))
        new_width, new_height = self._apply_scale(width, height, scale)

        # Ensure the number of pixels does not exceed max_pixels
        if new_width * new_height > self.max_pixels / img_num:
            scale = self.max_pixels / img_num / (new_width * new_height)
            new_width, new_height = self._apply_scale(new_width, new_height, scale)

        # Ensure longest edge does not exceed max_size
        if max(new_width, new_height) > self.max_size:
            scale = self.max_size / max(new_width, new_height)
            new_width, new_height = self._apply_scale(new_width, new_height, scale)

        return F.resize(img, (new_height, new_width), self.interpolation, antialias=self.antialias)


class ImageTransform:
    def __init__(
        self, 
        max_image_size, 
        min_image_size, 
        image_stride, 
        max_pixels=14*14*9*1024,
        image_mean=[0.5, 0.5, 0.5], 
        image_std=[0.5, 0.5, 0.5]
    ):
        self.stride = image_stride

        self.resize_transform = MaxLongEdgeMinShortEdgeResize(
            max_size=max_image_size, 
            min_size=min_image_size, 
            stride=image_stride,
            max_pixels=max_pixels,
        )
        self.to_tensor_transform = transforms.ToTensor()
        self.normalize_transform = transforms.Normalize(mean=image_mean, std=image_std, inplace=True)

    def __call__(self, img, img_num=1):
        img = self.resize_transform(img, img_num=img_num)
        img = self.to_tensor_transform(img)
        img = self.normalize_transform(img)
        return img

class InterleavedBaseIterableDataset:
    def __init__(self,  transform, vit_transform, tokenizer):
        self.transform = transform
        self.vit_transform = vit_transform
        self.tokenizer = tokenizer

    def _init_data(self,):
        data = {
            'sequence_plan': [],
            'text_ids_list': [],
            'image_tensor_list': [],
            "action": [],
            'num_tokens': 0,
        }
        return data

    def _add_text(self, data, text, need_loss, enable_cfg=True):
        text_ids = self.tokenizer.encode(text)
        data['num_tokens'] += len(text_ids)
        data['text_ids_list'].append(text_ids)
        data['sequence_plan'].append(
            {
                'type': 'text',
                'enable_cfg': int(enable_cfg),
                'loss': int(need_loss),
                'special_token_loss': 0,
                'special_token_label': None,
            }
        )
        return data
    
    def _add_action(self, data, action, need_loss, enable_cfg=True):
        data['action'].append(action)
        data['sequence_plan'].append(
            {
                "type": "action",
                "enable_cfg": 0,
                "loss": 1,
                "special_token_loss": 1,
            }
        )
        return data

    def _add_image(self, data, image, need_loss, need_vae, need_vit, enable_cfg=True):
        assert need_loss or need_vae or need_vit

        if need_loss:
            data['sequence_plan'].append(
                {
                    'type': 'vae_image', 
                    'enable_cfg': 0, 
                    'loss': 1, 
                    'special_token_loss': 0,
                    'special_token_label': None,
                }
            )
            image_tensor = self.transform(image)
            height, width = image_tensor.shape[1:]
            data['num_tokens'] += width * height // self.transform.stride ** 2
            data['image_tensor_list'].append(image_tensor)
        if need_vae:
            data['sequence_plan'].append(
                    {
                    'type':'vae_image',
                    'enable_cfg':int(enable_cfg),
                    'loss':0,
                    'special_token_loss':0,
                    'special_token_label':None,
                    }
            )
            image_tensor = self.transform(image)
            height, width = image_tensor.shape[1:]
            data['num_tokens'] += width * height // self.transform.stride ** 2
            data['image_tensor_list'].append(image_tensor.clone())

        if need_vit:
            data['sequence_plan'].append(
                {
                    'type': 'vit_image',
                    'enable_cfg': int(enable_cfg), 
                    'loss': 0,
                    'special_token_loss': 0,
                    'special_token_label': None,
                },
            )
            vit_image_tensor = self.vit_transform(image)
            height, width = vit_image_tensor.shape[1:]
            data['num_tokens'] += width * height // self.vit_transform.stride ** 2
            data['image_tensor_list'].append(vit_image_tensor)
        
        return data

    def _add_video(self, data, frames, frame_indexes, need_loss, need_vae, enable_cfg=True):
        assert int(need_loss) + int(need_vae) == 1

        if need_loss:
            for idx, (image, frame_idx) in enumerate(zip(frames, frame_indexes)):
                current_sequence_plan = {
                    'type': 'vae_image', 
                    'enable_cfg': 0, 
                    'loss': 1, 
                    'special_token_loss': 0,
                    'special_token_label': None,
                    'split_start': idx == 0,
                    'split_end': idx == len(frames) - 1,
                }
                if idx < len(frame_indexes) - 1:
                    current_sequence_plan['frame_delta'] = frame_indexes[idx + 1] - frame_idx
                data['sequence_plan'].append(current_sequence_plan)
                image_tensor = self.transform(image)
                height, width = image_tensor.shape[1:]
                data['image_tensor_list'].append(image_tensor)
                data['num_tokens'] += width * height // self.transform.stride ** 2

        elif need_vae:
            for idx, (image, frame_idx) in enumerate(zip(frames, frame_indexes)):
                current_sequence_plan = {
                    'type': 'vae_image', 
                    'enable_cfg': int(enable_cfg), 
                    'loss': 0, 
                    'special_token_loss': 0,
                    'special_token_label': None,
                    'split_start': idx == 0,
                    'split_end': idx == len(frames) - 1,
                }
                if idx < len(frame_indexes) - 1:
                    current_sequence_plan['frame_delta'] = frame_indexes[idx + 1] - frame_idx
                data['sequence_plan'].append(current_sequence_plan)
                image_tensor = self.transform(image)
                height, width = image_tensor.shape[1:]
                data['image_tensor_list'].append(image_tensor)
                data['num_tokens'] += width * height // self.transform.stride ** 2

        return data



class UnifiedEditIterableDataset(InterleavedBaseIterableDataset):
    def __init__(self, transform, vit_transform, tokenizer, action_horizon=5, action_dim=7, visual_gen=True,action_gen=True,):
        super().__init__(transform, vit_transform, tokenizer)
        self.action_horizon, self.action_dim = action_horizon, action_dim
        self.visual_gen = visual_gen
        self.action_gen = action_gen
        self.action_horizon = action_horizon

    def sort_keys(self,sample_keys):
        ## order: wrist first, head second, 3rd view(images) last; left first, right second
        sorted_sample_keys = []
        for key in sample_keys:
            if "left" in key and "wrist" in key:
                sorted_sample_keys.append(key)
        for key in sample_keys:
            if "right" in key and "wrist" in key:
                sorted_sample_keys.append(key)
        for key in sample_keys:
            if "left" in key and "head" in key:
                sorted_sample_keys.append(key)
        for key in sample_keys:
            if "right" in key and "head" in key:
                sorted_sample_keys.append(key)
        for key in sample_keys:
            if "images" in key and "wrist" not in key and "head" not in key:
                sorted_sample_keys.append(key)
 
        for key in sample_keys:
            if key not in sorted_sample_keys:
                sorted_sample_keys.append(key)
        return sorted_sample_keys
          

    def __call__(self, sample):
        batch_size = len(sample['task'])
        datas = []
        # observation_images = []
        data = self._init_data()
        ## For action generation 
        sample_keys = list(sample.keys())
        sorted_sample_keys =self.sort_keys(sample_keys)

        ## TODO: shuffle keys
        sorted_sample_keys = np.random.shuffle(sorted_sample_keys)

        ## For in-context reference (image1, action , image2)-pair
        for key in sorted_sample_keys:
            if "images." in key and "ref" in key:
                ref_action = sample['ref_action']
                for i in range(len(ref_action) - 1):
                    a_ref_action = ref_action[i]
                    current_image = sample[key][0][i]
                    next_image = sample[key][0][i + 1]
                    data = self._add_image(
                        data,
                        pil_img2rgb(Image.fromarray((current_image.detach().cpu().numpy().transpose((1, 2, 0)) * 255).astype(np.uint8))),
                        need_loss=False,
                        need_vae=False,
                        need_vit=True,
                    )
                    data = self._add_action(
                        data,
                        a_ref_action,
                        need_loss=False,
                    )
                    data = self._add_image(
                        data,
                        pil_img2rgb(Image.fromarray((next_image.detach().cpu().numpy().transpose((1, 2, 0)) * 255).astype(np.uint8))),
                        need_loss=False,
                        need_vae=False,
                        need_vit=True,
                    )

        for key in sorted_sample_keys:
            if "images." in key and "observation" in key:
                data = self._add_image(
                    data,
                    pil_img2rgb(Image.fromarray((sample[key][0].detach().cpu().numpy().transpose((1, 2, 0)) * 255).astype(np.uint8))),
                    need_loss=False,
                    need_vae=self.visual_gen,
                    need_vit=True,
                )
        ## For VQA images
        image_cnt = 0
        for key in sorted_sample_keys:
            if "images" in key and "vqa" in key:
                image_cnt += 1
        for image_idx in range(image_cnt):
            key = "vqa.images.image.{}".format(image_idx)
            data = self._add_image(
                data,
                pil_img2rgb(Image.fromarray((sample[key][0].detach().cpu().numpy().transpose((1, 2, 0)) * 255).astype(np.uint8))),
                need_loss=False,
                need_vae=False,
                need_vit=True,
            )
        ## For actionless video
        for key in sorted_sample_keys:
            if "images" in key and "actionless_video" in key:
                data = self._add_image(
                    data,
                    pil_img2rgb(Image.fromarray((sample[key][0].detach().cpu().numpy().transpose((1, 2, 0)) * 255).astype(np.uint8))),
                    need_loss=False,
                    need_vae=True,
                    need_vit=True,
                )

        instruction = sample['task'][0]
        if "action" in sorted_sample_keys:
            instruction = "Task:" + sample['task'][0] + ". Please predict the next observation and the action."
            data = self._add_text(data, instruction, need_loss=False)
        else:
            conv = json.loads(instruction)
            for a_conv in conv:
                if a_conv['role'] == "assistant":
                    a_conv = a_conv['content'][0]['text']
                    data = self._add_text(data, a_conv, need_loss=True)
                elif a_conv['role'] == "user" or a_conv['role'] == 'system':
                    a_conv = a_conv['content'][0]['text']
                    data = self._add_text(data, a_conv, need_loss=False)
                else:
                    raise NotImplementedError

        ## For goal image generation
        next_images = []
        for key in sorted_sample_keys:
            if "images." in key and "next" in key:
                next_images.append((sample[key][0].detach().cpu().numpy().transpose((1, 2, 0)) * 255).astype(np.uint8))
        next_img_num = len(next_images)
        if self.visual_gen:
            if len(next_images) > 0:
                # next_images = next_images[-1] ## TODO: select only one image for now
                next_images = cv2.hconcat(next_images)
                data = self._add_image(
                    data, 
                    pil_img2rgb(Image.fromarray(next_images)),
                    need_loss=True, 
                    need_vae=False, 
                    need_vit=True, 
                )
        ## For action generation
        if self.action_gen:
            if ACTION in sample.keys():
                actions = sample['action'][0]
                data = self._add_action(
                    data,
                    actions,
                    need_loss=True,
                )
        data['ref_num'] = sample['ref_num']
        datas.append(data)
        return datas
    

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


class PackedDataset:
    def __init__(
        self, 
        data_config, 
        tokenizer, 
        special_tokens,
        expected_num_tokens=54768, 
        max_num_tokens_per_sample=16384,
        max_num_tokens=86864,
        prefer_buffer_before=16384,
        max_buffer_size=50,
        interpolate_pos=False,
        use_flex=False,
        data_status=None,
        action_dim=7,
        action_horizon=5,
        visual_gen=True,
    ):
        self.expected_num_tokens = expected_num_tokens
        self.max_num_tokens_per_sample = max_num_tokens_per_sample
        self.prefer_buffer_before = prefer_buffer_before
        self.max_num_tokens = max_num_tokens
        self.max_buffer_size = max_buffer_size
        self.tokenizer = tokenizer
        self.use_flex = use_flex
        for k, v in special_tokens.items():
            setattr(self, k, v)
        self.action_dim, self.action_horizon = action_dim, action_horizon
        self.visual_gen = visual_gen
        self.dataset = self.build_datasets("unified_edit")
        self.data_config = data_config
        self.interpolate_pos = interpolate_pos
        if self.interpolate_pos:
            self.get_flattened_position_ids = get_flattened_position_ids_interpolate
        else:
            self.get_flattened_position_ids = get_flattened_position_ids_extrapolate
        
    def build_datasets(self, dataset_name):
        dataset_args = {
            "image_transform_args": {
                "image_stride": 16,
                # "max_image_size": 1024,
                # "min_image_size": 512
                "max_image_size": 256,
                "min_image_size": 256
            },
            "vit_image_transform_args":{
                "image_stride": 14,
                "max_image_size": 256,
                "min_image_size": 256
                # "max_image_size": 128,
                # "min_image_size": 64
            },
            "is_mandatory": False,
        }
        transform = ImageTransform(**dataset_args.pop('image_transform_args'))
        dataset_args['transform'] = transform
        vit_transform = ImageTransform(**dataset_args.pop('vit_image_transform_args'))
        dataset_args['vit_transform'] = vit_transform

        data = UnifiedEditIterableDataset(transform, vit_transform, self.tokenizer, 
                action_dim = self.action_dim, action_horizon = self.action_horizon, visual_gen=self.visual_gen)
        return data

    def set_epoch(self, seed):
        for dataset in self.grouped_datasets:
            dataset.set_epoch(seed)

    def set_sequence_status(self):
        sequence_status = dict(
            curr                        = 0,
            sample_lens                 = list(),
            packed_position_ids         = list(),
            nested_attention_masks      = list(),
            split_lens                  = list(),
            attn_modes                  = list(),
            packed_text_ids             = list(), 
            packed_text_indexes         = list(),
            packed_label_ids            = list(),
            ce_loss_indexes             = list(),
            ce_loss_weights             = list(),
            vae_image_tensors           = list(), 
            packed_latent_position_ids  = list(),
            vae_latent_shapes           = list(), 
            packed_vae_token_indexes    = list(), 
            packed_timesteps            = list(), 
            mse_loss_indexes            = list(),
            packed_vit_tokens           = list(), 
            vit_token_seqlens           = list(),
            packed_vit_position_ids     = list(),
            packed_vit_token_indexes    = list(), 
            packed_act_token_indexes = list(),
            packed_act_tokens        = list(),
            ref_num                  = list(),
        )
        return sequence_status

    def to_tensor(self, sequence_status):
        data = dict(
            sequence_length=sum(sequence_status['sample_lens']),
            sample_lens=sequence_status['sample_lens'],
            packed_text_ids=torch.tensor(sequence_status['packed_text_ids']),
            packed_text_indexes=torch.tensor(sequence_status['packed_text_indexes']),
            packed_position_ids=torch.tensor(sequence_status['packed_position_ids']),
        )
        if not self.use_flex:
            data['nested_attention_masks'] = sequence_status['nested_attention_masks']
        else:
            sequence_len = data['sequence_length']
            pad_len = self.max_num_tokens - sequence_len
            data['split_lens'] = sequence_status['split_lens'] + [pad_len]
            data['attn_modes'] = sequence_status['attn_modes'] + ['causal']
            data['sample_lens'] += [pad_len]

        # if the model has a convnet vae (e.g., as visual tokenizer)
        if len(sequence_status['vae_image_tensors']) > 0:
            image_tensors = sequence_status.pop('vae_image_tensors')
            image_sizes = [item.shape for item in image_tensors]
            max_image_size = [max(item) for item in list(zip(*image_sizes))]
            padded_images = torch.zeros(size=(len(image_tensors), *max_image_size))
            for i, image_tensor in enumerate(image_tensors):
                padded_images[i, :, :image_tensor.shape[1], :image_tensor.shape[2]] = image_tensor

            data['padded_images'] = padded_images
            data['patchified_vae_latent_shapes'] = sequence_status['vae_latent_shapes']
            data['packed_latent_position_ids'] = torch.cat(sequence_status['packed_latent_position_ids'], dim=0)
            data['packed_vae_token_indexes'] = torch.tensor(sequence_status['packed_vae_token_indexes'])

        # if the model has a vit (e.g., as visual tokenizer)
        if len(sequence_status['packed_vit_tokens']) > 0:
            data['packed_vit_tokens'] = torch.cat(sequence_status['packed_vit_tokens'], dim=0)
            data['packed_vit_position_ids'] = torch.cat(sequence_status['packed_vit_position_ids'], dim=0)
            data['packed_vit_token_indexes'] = torch.tensor(sequence_status['packed_vit_token_indexes'])
            data['vit_token_seqlens'] = torch.tensor(sequence_status['vit_token_seqlens'])
        # if the model is required to perform visual generation
        if len(sequence_status['packed_timesteps']) > 0:
            data['packed_timesteps'] = torch.tensor(sequence_status['packed_timesteps'])
            data['mse_loss_indexes'] = torch.tensor(sequence_status['mse_loss_indexes'])

        # if the model is required to perform text generation
        if len(sequence_status['packed_label_ids']) > 0:
            data['packed_label_ids'] = torch.tensor(sequence_status['packed_label_ids'])
            data['ce_loss_indexes'] = torch.tensor(sequence_status['ce_loss_indexes'])
            data['ce_loss_weights'] = torch.tensor(sequence_status['ce_loss_weights'])

        if len(sequence_status['packed_act_tokens']) > 0:
            data['packed_act_tokens'] = torch.cat(sequence_status['packed_act_tokens'], dim=0)
            data['packed_act_token_indexes'] = torch.tensor(sequence_status['packed_act_token_indexes'])
        return data

    def __call__(self, batch_dataloader, tokenize_action, use_ref=True):
        dl_iter = iter(batch_dataloader)
        batch_data_indexes = []
        sequence_status = self.set_sequence_status()
        buffer = []
        while True:
            while True:
                try:
                    batch = next(dl_iter)
                except StopIteration:
                    dl_iter = iter(dataloader)
                    batch = next(dl_iter)
                if "action" in batch.keys():
                    actions = batch["action"]
                    act_ids = tokenize_action(actions)
                    batch['action'] = act_ids
                if "ref_action" in batch.keys() and use_ref:
                    ref_actions = batch['ref_action'][0]
                    ref_act_ids = []
                    for ref_action in ref_actions:
                        ref_act_ids.append(tokenize_action(ref_action.unsqueeze(0).unsqueeze(0))[0])
                    batch['ref_action'] = ref_act_ids
                sample = self.dataset(batch)[0]
                num_tokens = sample['num_tokens'] + 2 * len(sample['sequence_plan'])
                if num_tokens < self.max_num_tokens_per_sample:
                    break
                else:
                    print(f"skip a sample with length {num_tokens}")
                    continue
            if sequence_status['curr'] + num_tokens > self.max_num_tokens:
                print(f"Yielding data with length {sum(sequence_status['sample_lens'])}")
                data = self.to_tensor(sequence_status)
                yield data
                sequence_status = self.set_sequence_status()
                batch_data_indexes = []
            sequence_status = self.pack_sequence(sample, sequence_status)
            continue
        return sequence_status

    def pack_sequence(self, sample, sequence_status):
        image_tensor_list = sample['image_tensor_list']
        text_ids_list = sample['text_ids_list']
        sequence_plan = sample['sequence_plan']

        split_lens, attn_modes = list(), list()
        curr = sequence_status['curr']
        curr_rope_id = 0
        sample_lens = 0
        for item in sequence_plan:
            split_start = item.get('split_start', True)
            if split_start:
                curr_split_len = 0

            if item['type'] == 'text':
                text_ids = text_ids_list.pop(0)
                # if item['enable_cfg'] == 1 and random.random() < self.data_config.text_cond_dropout_prob:
                #     continue

                shifted_text_ids = [self.bos_token_id] + text_ids
                sequence_status['packed_text_ids'].extend(shifted_text_ids)
                sequence_status['packed_text_indexes'].extend(range(curr, curr + len(shifted_text_ids)))
                if item['loss'] == 1:
                    sequence_status['ce_loss_indexes'].extend(range(curr, curr + len(shifted_text_ids)))
                    sequence_status['ce_loss_weights'].extend(
                        [len2weight(len(shifted_text_ids))] * len(shifted_text_ids)
                    )
                    sequence_status['packed_label_ids'].extend(text_ids + [self.eos_token_id])
                curr += len(shifted_text_ids)
                curr_split_len += len(shifted_text_ids)

                # add a <|im_end|> token
                sequence_status['packed_text_ids'].append(self.eos_token_id)
                sequence_status['packed_text_indexes'].append(curr)
                if item['special_token_loss'] == 1: # <|im_end|> may have loss
                    sequence_status['ce_loss_indexes'].append(curr)
                    sequence_status['ce_loss_weights'].append(1.0)
                    sequence_status['packed_label_ids'].append(item['special_token_label'])
                curr += 1
                curr_split_len += 1

                # update sequence status
                attn_modes.append("causal")
                sequence_status['packed_position_ids'].extend(range(curr_rope_id, curr_rope_id + curr_split_len))
                curr_rope_id += curr_split_len
            
            elif item['type'] == 'vit_image':
                image_tensor = image_tensor_list.pop(0)

                # add a <|startofimage|> token
                sequence_status['packed_text_ids'].append(self.start_of_image)
                sequence_status['packed_text_indexes'].append(curr)
                curr += 1
                curr_split_len += 1

                # preprocess image
                vit_tokens = patchify(image_tensor, self.data_config.vit_patch_size)
                num_img_tokens = vit_tokens.shape[0]
                sequence_status['packed_vit_token_indexes'].extend(range(curr, curr + num_img_tokens))
                curr += num_img_tokens
                curr_split_len += num_img_tokens

                sequence_status['packed_vit_tokens'].append(vit_tokens)
                sequence_status['vit_token_seqlens'].append(num_img_tokens)
                sequence_status['packed_vit_position_ids'].append(
                    self.get_flattened_position_ids(
                        image_tensor.size(1), image_tensor.size(2),
                        self.data_config.vit_patch_size, 
                        max_num_patches_per_side=self.data_config.max_num_patch_per_side
                    )
                )
                

                # add a <|endofimage|> token
                sequence_status['packed_text_ids'].append(self.end_of_image)
                sequence_status['packed_text_indexes'].append(curr)
                if item['special_token_loss'] == 1: # <|endofimage|> may have loss
                    sequence_status['ce_loss_indexes'].append(curr)
                    sequence_status['ce_loss_weights'].append(1.0)
                    sequence_status['packed_label_ids'].append(item['special_token_label'])
                curr += 1
                curr_split_len += 1

                # update sequence status
                attn_modes.append("full")
                sequence_status['packed_position_ids'].extend([curr_rope_id] * curr_split_len)
                curr_rope_id += 1

            elif item['type'] == 'action':
                text_ids = sample['action'].pop(0).tolist()
                shifted_text_ids = [self.boa_token_id] + text_ids
                sequence_status['packed_text_ids'].extend(shifted_text_ids)
                sequence_status['packed_text_indexes'].extend(range(curr, curr + len(shifted_text_ids)))
                if item['loss'] == 1:
                    sequence_status['ce_loss_indexes'].extend(range(curr, curr + len(shifted_text_ids)))
                    sequence_status['ce_loss_weights'].extend(
                        [len2weight(len(shifted_text_ids))] * len(shifted_text_ids)
                    )
                    sequence_status['packed_label_ids'].extend(text_ids + [self.eoa_token_id])
                curr += len(shifted_text_ids)
                curr_split_len += len(shifted_text_ids)

                sequence_status['packed_text_ids'].append(self.eoa_token_id)
                sequence_status['packed_text_indexes'].append(curr)
                if item['special_token_loss'] == 1:
                    sequence_status['ce_loss_indexes'].append(curr)
                    sequence_status['ce_loss_weights'].append(1.0)
                    sequence_status['packed_label_ids'].append(self.eoa_token_id)
                curr += 1
                curr_split_len += 1

                # update sequence status
                attn_modes.append("causal")
                sequence_status['packed_position_ids'].extend(range(curr_rope_id, curr_rope_id + curr_split_len))
                curr_rope_id += 1

            elif item['type'] == 'vae_image':
                image_tensor = image_tensor_list.pop(0)
                # if item['enable_cfg'] == 1 and random.random() < self.data_config.vae_cond_dropout_prob:
                #     # FIXME fix vae dropout in video2video setting.
                #     curr_rope_id += 1
                #     continue

                # add a <|startofimage|> token
                sequence_status['packed_text_ids'].append(self.start_of_image)
                sequence_status['packed_text_indexes'].append(curr)
                curr += 1
                curr_split_len += 1

                # preprocess image
                sequence_status['vae_image_tensors'].append(image_tensor)
                sequence_status['packed_latent_position_ids'].append(
                    self.get_flattened_position_ids(
                        image_tensor.size(1), image_tensor.size(2),
                        self.data_config.vae_image_downsample, 
                        max_num_patches_per_side=self.data_config.max_latent_size
                    )
                )
                H, W = image_tensor.shape[1:]
                h = H // self.data_config.vae_image_downsample
                w = W // self.data_config.vae_image_downsample
                sequence_status['vae_latent_shapes'].append((h, w))

                num_img_tokens = w * h
                sequence_status['packed_vae_token_indexes'].extend(range(curr, curr + num_img_tokens))
                if item['loss'] == 1:
                    sequence_status['mse_loss_indexes'].extend(range(curr, curr + num_img_tokens))
                    if split_start:
                        timestep = np.random.randn()
                else:
                    timestep = float('-inf')

                sequence_status['packed_timesteps'].extend([timestep] * num_img_tokens)
                curr += num_img_tokens
                curr_split_len += num_img_tokens

                # add a <|endofimage|> token
                sequence_status['packed_text_ids'].append(self.end_of_image)
                sequence_status['packed_text_indexes'].append(curr)
                # <|endofimage|> may have loss
                if item['special_token_loss'] == 1:
                    sequence_status['ce_loss_indexes'].append(curr)
                    sequence_status['ce_loss_weights'].append(1.0)
                    sequence_status['packed_label_ids'].append(item['special_token_label'])
                curr += 1
                curr_split_len += 1

                # update sequence status
                if split_start:
                    if item['loss'] == 1 and 'frame_delta' not in item.keys():
                        attn_modes.append("noise")
                    else:
                        attn_modes.append("full")
                sequence_status['packed_position_ids'].extend([curr_rope_id] * (num_img_tokens + 2))
                if 'frame_delta' in item.keys():
                    curr_rope_id += item['frame_delta']
                elif item['loss'] == 0:
                    curr_rope_id += 1

            if item.get('split_end', True):
                split_lens.append(curr_split_len)
                sample_lens += curr_split_len
        sequence_status['curr'] = curr
        sequence_status['sample_lens'].append(sample_lens)
        # prepare attention mask
        if not self.use_flex:
            sequence_status['nested_attention_masks'].append(
                prepare_attention_mask_per_sample(split_lens, attn_modes)
            )
        else:
            sequence_status['split_lens'].extend(split_lens)
            sequence_status['attn_modes'].extend(attn_modes)
        sequence_status['ref_num'].append(sample['ref_num'])
        return sequence_status


class SimpleCustomBatch:
    def __init__(self, batch):
        data = batch[0]
        self.sequence_length = data["sequence_length"]
        self.sample_lens = data["sample_lens"]
        self.packed_text_ids = data["packed_text_ids"]
        self.packed_text_indexes = data["packed_text_indexes"]
        self.packed_position_ids = data["packed_position_ids"]

        self.use_flex = "nested_attention_masks" not in data.keys()

        if self.use_flex:
            self.split_lens = data["split_lens"]
            self.attn_modes = data["attn_modes"]
        else:
            self.nested_attention_masks = data["nested_attention_masks"]

        if "padded_images" in data.keys():
            self.padded_images = data["padded_images"]
            self.patchified_vae_latent_shapes = data["patchified_vae_latent_shapes"]
            self.packed_latent_position_ids = data["packed_latent_position_ids"]
            self.packed_vae_token_indexes = data["packed_vae_token_indexes"]

        if "packed_vit_tokens" in data.keys():
            self.packed_vit_tokens = data["packed_vit_tokens"]
            self.packed_vit_position_ids = data["packed_vit_position_ids"]
            self.packed_vit_token_indexes = data["packed_vit_token_indexes"]
            self.vit_token_seqlens = data["vit_token_seqlens"]

        if "packed_timesteps" in data.keys():
            self.packed_timesteps = data["packed_timesteps"]
            self.mse_loss_indexes = data["mse_loss_indexes"]

        if "packed_label_ids" in data.keys():
            self.packed_label_ids = data["packed_label_ids"]
            self.ce_loss_indexes = data["ce_loss_indexes"]
            self.ce_loss_weights = data["ce_loss_weights"]

        if "packed_act_tokens" in data.keys():
            self.packed_act_tokens = data["packed_act_tokens"]
            self.packed_act_token_indexes = data["packed_act_token_indexes"]
            self.act_ce_loss_indexes = data['act_ce_loss_indexes']
            self.act_ce_loss_weights = data['act_ce_loss_weights']

    def pin_memory(self):
        self.packed_text_ids = self.packed_text_ids.pin_memory()
        self.packed_text_indexes = self.packed_text_indexes.pin_memory()
        self.packed_position_ids = self.packed_position_ids.pin_memory()

        if not self.use_flex:
            self.nested_attention_masks = [item.pin_memory() for item in self.nested_attention_masks]

        if hasattr(self, 'padded_images'):
            self.padded_images = self.padded_images.pin_memory()
            self.packed_vae_token_indexes = self.packed_vae_token_indexes.pin_memory()
            self.packed_latent_position_ids = self.packed_latent_position_ids.pin_memory()

        if hasattr(self, 'packed_timesteps'):
            self.packed_timesteps = self.packed_timesteps.pin_memory()
            self.mse_loss_indexes = self.mse_loss_indexes.pin_memory()

        if hasattr(self, 'packed_vit_tokens'):
            self.packed_vit_tokens = self.packed_vit_tokens.pin_memory()
            self.packed_vit_position_ids = self.packed_vit_position_ids.pin_memory()
            self.packed_vit_token_indexes = self.packed_vit_token_indexes.pin_memory()
            self.vit_token_seqlens = self.vit_token_seqlens.pin_memory()

        if hasattr(self, 'packed_act_tokens'):
            self.packed_act_tokens = self.packed_act_tokens.pin_memory()
            self.packed_act_token_indexes = self.packed_act_token_indexes.pin_memory()
            self.act_ce_loss_indexes = self.act_ce_loss_indexes.pin_memory()
            self.act_ce_loss_weights = self.act_ce_loss_weights.pin_memory()

        if hasattr(self, 'packed_label_ids'):
            self.packed_label_ids = self.packed_label_ids.pin_memory()
            self.ce_loss_indexes = self.ce_loss_indexes.pin_memory()
            self.ce_loss_weights = self.ce_loss_weights.pin_memory()

        return self

    def cuda(self, device):
        self.packed_text_ids = self.packed_text_ids.to(device)
        self.packed_text_indexes = self.packed_text_indexes.to(device)
        self.packed_position_ids = self.packed_position_ids.to(device)

        if not self.use_flex:
            self.nested_attention_masks = [item.to(device) for item in self.nested_attention_masks]

        if hasattr(self, 'padded_images'):
            self.padded_images = self.padded_images.to(device)
            self.packed_vae_token_indexes = self.packed_vae_token_indexes.to(device)
            self.packed_latent_position_ids = self.packed_latent_position_ids.to(device)

        if hasattr(self, 'packed_timesteps'):
            self.packed_timesteps = self.packed_timesteps.to(device)
            self.mse_loss_indexes = self.mse_loss_indexes.to(device)

        if hasattr(self, 'packed_vit_tokens'):
            self.packed_vit_tokens = self.packed_vit_tokens.to(device)
            self.packed_vit_position_ids = self.packed_vit_position_ids.to(device)
            self.packed_vit_token_indexes = self.packed_vit_token_indexes.to(device)
            self.vit_token_seqlens = self.vit_token_seqlens.to(device)

        if hasattr(self, 'packed_act_token'):
            self.packed_act_tokens = self.packed_act_tokens.to(device)
            self.packed_act_token_indexes = self.packed_act_token_indexes.to(device)
            self.act_ce_loss_indexes = self.act_ce_loss_indexes.to(device)
            self.act_ce_loss_weights = self.act_ce_loss_weights.to(device)

        if hasattr(self, 'packed_label_ids'):
            self.packed_label_ids = self.packed_label_ids.to(device)
            self.ce_loss_indexes = self.ce_loss_indexes.to(device)
            self.ce_loss_weights = self.ce_loss_weights.to(device)
        
        return self

    def to_dict(self):
        data = dict(
            sequence_length = self.sequence_length,
            sample_lens = self.sample_lens,
            packed_text_ids = self.packed_text_ids,
            packed_text_indexes = self.packed_text_indexes,
            packed_position_ids = self.packed_position_ids,
        )

        if not self.use_flex:
            data['nested_attention_masks'] = self.nested_attention_masks
        else:
            data['split_lens'] = self.split_lens
            data['attn_modes'] = self.attn_modes

        if hasattr(self, 'padded_images'):
            data['padded_images'] = self.padded_images
            data['patchified_vae_latent_shapes'] = self.patchified_vae_latent_shapes
            data['packed_latent_position_ids'] = self.packed_latent_position_ids
            data['packed_vae_token_indexes'] = self.packed_vae_token_indexes

        if hasattr(self, 'packed_vit_tokens'):
            data['packed_vit_tokens'] = self.packed_vit_tokens
            data['packed_vit_position_ids'] = self.packed_vit_position_ids
            data['packed_vit_token_indexes'] = self.packed_vit_token_indexes
            data['vit_token_seqlens'] = self.vit_token_seqlens
        
        if hasattr(self, 'packed_timesteps'):
            data['packed_timesteps'] = self.packed_timesteps
            data['mse_loss_indexes'] = self.mse_loss_indexes

        if hasattr(self, 'packed_label_ids'):
            data['packed_label_ids'] = self.packed_label_ids
            data['ce_loss_indexes'] = self.ce_loss_indexes
            data['ce_loss_weights'] = self.ce_loss_weights

        if hasattr(self, 'packed_act_tokens'):
            data['packed_act_tokens'] = self.packed_act_tokens
            data['packed_act_token_indexes'] = self.packed_act_token_indexes
        return data


def collate_wrapper():
    def collate_fn(batch):
        return SimpleCustomBatch(batch)
    return collate_fn
