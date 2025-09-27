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
from lerobot.datasets.factory import make_dataset


@parser.wrap()
def download_data(cfg: TrainPipelineConfig) -> None:
    cfg.type = "pi0"
    cfg.resume = True
    # cfg.validate()
    dataset = make_dataset(cfg)
    

if __name__ == "__main__":
    download_data()
