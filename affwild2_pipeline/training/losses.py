from __future__ import annotations

import torch
import torch.nn as nn


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean squared error over valid elements.

    pred/target: (B, T, 2)
    mask: (B, T) bool
    """
    if pred.shape != target.shape:
        raise ValueError("pred and target must have the same shape")
    if mask.ndim != 2:
        raise ValueError("mask must have shape (B, T)")
    mask_f = mask.to(dtype=pred.dtype).unsqueeze(-1)  # (B, T, 1)
    diff2 = (pred - target) ** 2
    num = (diff2 * mask_f).sum()
    den = mask_f.sum().clamp_min(1.0) * pred.shape[-1]
    return num / den


class CCCLoss(nn.Module):
    """Concordance correlation coefficient loss for VA regression.

    Computes 1 - mean CCC over the last dimension (valence, arousal).
    Mask is applied over (B, T).
    """

    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if pred.shape != target.shape:
            raise ValueError("pred and target must have the same shape")
        if mask.ndim != 2:
            raise ValueError("mask must have shape (B, T)")

        mask_f = mask.to(dtype=pred.dtype).unsqueeze(-1)  # (B,T,1)
        pred = pred * mask_f
        target = target * mask_f

        # flatten over B,T
        pred = pred.reshape(-1, pred.shape[-1])
        target = target.reshape(-1, target.shape[-1])
        mask_f = mask_f.reshape(-1, 1)

        denom = mask_f.sum(dim=0).clamp_min(1.0)
        mu_x = (pred * mask_f).sum(dim=0) / denom
        mu_y = (target * mask_f).sum(dim=0) / denom

        vx = ((pred - mu_x) ** 2 * mask_f).sum(dim=0) / denom
        vy = ((target - mu_y) ** 2 * mask_f).sum(dim=0) / denom
        cov = (((pred - mu_x) * (target - mu_y)) * mask_f).sum(dim=0) / denom

        ccc = (2 * cov) / (vx + vy + (mu_x - mu_y) ** 2 + self.eps)
        return 1.0 - ccc.mean()
