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
from typing import Any

from lerobot.utils.utils import format_big_number
import torch

class AverageMeter:
    """
    Computes and stores the average and current value
    Adapted from https://github.com/pytorch/examples/blob/main/imagenet/main.py
    """

    def __init__(self, name: str, fmt: str = ":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0.0

    def update(self, val: float, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name}:{avg" + self.fmt + "}"
        return fmtstr.format(**self.__dict__)


class MetricsTracker:
    """
    A helper class to track and log metrics over time.

    Usage pattern:

    ```python
    # initialize, potentially with non-zero initial step (e.g. if resuming run)
    metrics = {"loss": AverageMeter("loss", ":.3f")}
    train_metrics = MetricsTracker(cfg, dataset, metrics, initial_step=step)

    # update metrics derived from step (samples, episodes, epochs) at each training step
    train_metrics.step()

    # update various metrics
    loss = policy.forward(batch)
    train_metrics.loss = loss

    # display current metrics
    logging.info(train_metrics)

    # export for wandb
    wandb.log(train_metrics.to_dict())

    # reset averages after logging
    train_metrics.reset_averages()
    ```
    """

    __keys__ = [
        "_num_frames",
        "_avg_samples_per_ep",
        "metrics",
        "steps",
        "samples",
        "episodes",
        "epochs",
        "tokens",
        "_num_episodes",
        "accelerator",
    ]

    def __init__(
        self,
        num_frames: int,
        num_episodes: int,
        metrics: dict[str, AverageMeter],
        accelerator,
        initial_tokens: int = 0,
        initial_step: int = 0
    ):
        self.__dict__.update(dict.fromkeys(self.__keys__))
        self._num_frames = num_frames
        self.metrics = metrics
        self.tokens = initial_tokens
        self.steps = initial_step
        self._num_episodes = num_episodes
        self.accelerator=accelerator

    def __getattr__(self, name: str) -> int | dict[str, AverageMeter] | AverageMeter | Any:
        if name in self.__dict__:
            return self.__dict__[name]
        elif name in self.metrics:
            return self.metrics[name]
        else:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def __setattr__(self, name: str, value: Any) -> None:
        if name in self.__dict__:
            super().__setattr__(name, value)
        elif name in self.metrics:
            self.metrics[name].update(value)
        else:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def step(self, num_token) -> None:
        """
        Updates metrics that depend on 'step' for one step.
        """
            
        add_tokens = num_token
        add_steps = 1
        add_epochs = self._num_episodes / self._num_frames * add_steps

        add_step_all_proc = self.accelerator.gather(torch.tensor(add_steps).cuda())
        add_steps = add_step_all_proc.sum().detach().cpu().item()

        add_tokens_all_proc = self.accelerator.gather(torch.tensor(add_tokens).cuda())
        add_tokens = add_tokens_all_proc.sum().detach().cpu().item()
        
        self.tokens += add_tokens
        self.steps += add_steps

        self.epochs = self._num_episodes / self._num_frames * self.steps
        '''
        for k in self.metrics.keys():
            if isinstance(self.metrics[k], AverageMeter):
                v_all_proc = self.accelerator.gather(torch.tensor(self.metrics[k].avg).float().cuda())
                self.metrics[k].avg = v_all_proc.mean().detach().cpu().item() 
                v_all_proc = self.accelerator.gather(torch.tensor(self.metrics[k].sum).float().cuda())
                self.metrics[k].sum = v_all_proc.sum().detach().cpu().item()
                v_all_proc = self.accelerator.gather(torch.tensor(self.metrics[k].count).float().cuda())
                self.metrics[k].count = v_all_proc.sum().detach().cpu().item()
                v_all_proc = self.accelerator.gather(torch.tensor(self.metrics[k].val).float().cuda())
                self.metrics[k].val = v_all_proc.mean().detach().cpu().item()
                if torch.isnan(torch.tensor(self.metrics[k].val)).any() or torch.isinf(torch.tensor(self.metrics[k].val)).any():
                    print(v_all_proc, k)
                    import ipdb;ipdb.set_trace()
        '''
        return self.tokens, self.steps
    def __str__(self) -> str:
        
        display_list = [
            f"step:{format_big_number(self.steps)}",
            f"epch:{self.epochs:.2f}",
            f"tok:{format_big_number(self.tokens)}",
            *[str(m) for m in self.metrics.values()],
        ]
        return " ".join(display_list)

    def to_dict(self, use_avg: bool = True) -> dict[str, int | float]:
        """
        Returns the current metric values (or averages if `use_avg=True`) as a dict.
        """
        return {
            "steps": self.steps,
            "epochs": self.epochs,
            "tokens": self.tokens,
            **{k: m.avg if use_avg else m.val for k, m in self.metrics.items()},
        }

    def reset_averages(self) -> None:
        """Resets average meters."""
        for m in self.metrics.values():
            m.reset()
