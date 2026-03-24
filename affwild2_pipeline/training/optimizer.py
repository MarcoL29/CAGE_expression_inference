from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import torch


def build_optimizer(model: torch.nn.Module, lr: float, weight_decay: float = 0.01) -> torch.optim.Optimizer:
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def build_cosine_annealing_scheduler(
    optimizer: torch.optim.Optimizer,
    t_max: int,
) -> torch.optim.lr_scheduler.CosineAnnealingLR:
    """CosineAnnealingLR matching the repository's MaxViT training style.

    In `models/AffectNet8_Maxvit_Combined/train.py`, the scheduler is constructed as:
      CosineAnnealingLR(optimizer, T_max=BATCHSIZE * NUM_EPOCHS)
    and `.step()` is called **every iteration**.
    """
    t_max = int(max(1, t_max))
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max)
