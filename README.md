---
license: cc-by-nc-sa-4.0
language:
- en
- zh
size_categories:
- n>1T
tags:
- robotics
- real-world
- dual-arm
- whole body control
- manipulation

extra_gated_prompt: >-
  ### GALAXEA COMMUNITY LICENSE AGREEMENT: All the data and code
  within this repo are under [CC BY-NC-SA
  4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).
extra_gated_fields:
  First Name: text
  Last Name: text
  Email: text
  Country: country
  Affiliation: text
  Phone: text
  Job title:
    type: select
    options:
    - Student
    - Research Graduate
    - AI researcher
    - AI developer/engineer
    - Reporter
    - Other
  geo: ip_location
  By clicking Submit below I accept the terms of the license and acknowledge that the information I provide will be collected stored processed and shared in accordance with the Galaxea Privacy Policy: checkbox
extra_gated_description: >-
  The information you provide will be collected, stored, processed and shared in
  accordance with the Galaxea Privacy Policy.
extra_gated_button_content: Submit
---

# 🚀 Galaxea Open-World Dataset
[![Project Page](https://img.shields.io/badge/Project%20Page-000000?style=for-the-badge&logo=github)](https://opengalaxea.github.io/G0/)
[![Paper](https://img.shields.io/badge/Paper-8A2BE2?style=for-the-badge&logo=arxiv)](https://arxiv.org/abs/2509.00576)
[![Videos](https://img.shields.io/badge/Videos-FF0000?style=for-the-badge&logo=youtube)](https://opengalaxea.github.io/G0/)
[![Visualizer](https://img.shields.io/badge/Visualizer-FF8C00?style=for-the-badge&logo=airplayvideo)](https://opengalaxea.github.io/G0/visualizer/index.html)
[![Modelscope](https://img.shields.io/badge/Modelscope-1890FF?style=for-the-badge&logo=alibabacloud)](https://www.modelscope.cn/organization/Galaxea)
[![Twitter](https://img.shields.io/badge/Twitter-FF6B35?style=for-the-badge&logo=x)](https://x.com/Galaxea_x)
[![Linkedin](https://img.shields.io/badge/Linkedin-5865F2?style=for-the-badge&logo=linkedin)](https://www.linkedin.com/company/galaxeadynamics/posts/?feedView=all&viewAsMember=true)
[![Discord](https://img.shields.io/badge/Discord-1890FF?style=for-the-badge&logo=discord)](https://discord.gg/hB6BuUWZZA)

## Key features
- **500+ hours** of real-world mobile manipulation data.
- All data collected using **one uniform robotic embodiment** for consistency.
- Fine-grained **subtask language annotations**.
- Covers **residential**, **kitchen**, **retail**, and **office** settings.
- Dataset in **RLDS** and **LeRobot** format.

## Dataset Structure
**For convenience, we divided the 500 hours of data into four equal parts by time. We also provide a small sample dataset for quick start.**
```
rlds
├── part1_r1_lite
│   ├── 1.0.0
│   │   ├── dataset_info.json
│   │   ├── features.json
│   │   ├── merge_dataset_large_r1_lite-train.tfrecord-00000-of-02048
│   │   ├── ...
│   │   ├── merge_dataset_large_r1_lite-train.tfrecord-02047-of-02048
├── part2_r1_lite
├── part3_r1_lite
├── part4_r1_lite
├── sample
│   ├── 1.0.0
│   │   ├── merge_dataset_large_r1_lite-train.tfrecord-00000-of-01024
│   │   ├── ...
│   │   ├── merge_dataset_large_r1_lite-train.tfrecord-01023-of-01024
```

## RLDS Dataset Schema

```
OpenGalaxeaDataset = {
    "episode_metadata": {
        "file_path": tf.Text,  # path to the original data file
    },
    "steps": {
        "is_first": tf.Scalar(dtype=bool),  # true on first step of the episode
        "is_last": tf.Scalar(dtype=bool),  # true on last step of the episode

        "language_instruction": tf.Text,  # language instruction, format: "high level"@"low level chinese"@"low level english"
        "observation": {
            "base_velocity": tf.Tensor(3, dtype=float32),   # robot base velocity
            "gripper_state_left": tf.Tensor(1, dtype=float32),  # left gripper state, 0-close and 100-open
            "gripper_state_right": tf.Tensor(1, dtype=float32), # right gripper state, 0-close and 100-open
            "depth_camera_wrist_left": tf.Tensor(224, 224, 1, dtype=uint16),  # wrist camera depth left viewpoint, unit: mm
            "depth_camera_wrist_right": tf.Tensor(224, 224, 1, dtype=uint16),  # wrist camera depth right viewpoint, unit: mm
            "image_camera_head": tf.Tensor(224, 224, 3, dtype=uint8), # head camera RGB viewpoint
            "image_camera_wrist_left": tf.Tensor(224, 224, 3, dtype=uint8), # wrist camera RGB left viewpoint
            "image_camera_wrist_right": tf.Tensor(224, 224, 3, dtype=uint8), # wrist camera RGB right viewpoint
            "joint_position_arm_left": tf.Tensor(6, dtype=float32), # joint positions of the left arm
            "joint_position_arm_right": tf.Tensor(6, dtype=float32), # joint positions of the right arm
            "joint_position_torso": tf.Tensor(4, dtype=float32), # joint positions of the torso
            "joint_velocity_arm_left": tf.Tensor(6, dtype=float32), # joint velocities of the left arm
            "joint_velocity_arm_right": tf.Tensor(6, dtype=float32), # joint velocities of the right arm
            "last_action": tf.Tensor(26, dtype=float32), # history of the last action
        },
        # action dimensions:
        # 26 = 6 (left arm) + 1 (left gripper) + 6 (right arm) + 1 (right gripper) + 6 (torso) + 6 (base)
        "action": tf.Tensor(26, dtype=float32), 
        "segment_idx": tf.Scalar(dtype=int32),  # index of the segment in the episode
        "variant_idx": tf.Scalar(dtype=int32), 
    },
}
```
#### Lerobot Dataset Schema
A detailed lerobot format Galaxea Open-World Dataset schema can be seen at [lerobot_info.json](https://huggingface.co/datasets/OpenGalaxea/Galaxea-Open-World-Dataset/blob/main/lerobot_info.json).

## RLDS Example

We provide an example script to load our RLDS dataset and transform some episodes into mp4 video format (head camera).

```python
import tensorflow_datasets as tfds
import tyro
import os
import imageio
from tqdm import tqdm

def main(
    dataset_name: str, 
    data_dir: str, 
    output_dir: str = "extracted_videos",
    num_trajs: int = 10
):
    ds = tfds.load(dataset_name, split='train', data_dir=data_dir)
    print(f"Successfully loaded dataset: {dataset_name}")

    os.makedirs(output_dir, exist_ok=True)
    print(f"Videos will be saved to: {output_dir}")

    for i, episode in enumerate(tqdm(ds.take(num_trajs), total=num_trajs, desc="Exporting videos")):
        head_frames = []
        
        for step in episode['steps']:
            head_rgb_image = step['observation']['image_camera_head'].numpy()
            head_frames.append(head_rgb_image)
            instruction = step['language_instruction'].numpy().decode('utf-8')

        video_path = os.path.join(output_dir, f"traj_{i}_head_rgb.mp4")
        try:
            imageio.mimsave(video_path, head_frames, fps=15)
            print(f"Saved video for episode {i} to {video_path} with instruction: '{instruction}'")
        except Exception as e:
            print(f"Error saving video for episode {i}: {e}")

if __name__ == '__main__':
    tyro.cli(main)
```
## 📜 Citation

All the data and code within this repo are under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). If you use our dataset or models, please cite:

```bibtex
@article{galaxea2025,
  title={Galaxea G0: Open-World Dataset and Dual-System VLA Model},
  author={Galaxea Team},
  journal={arXiv preprint arXiv:2509.00576},
  year={2025}
}
