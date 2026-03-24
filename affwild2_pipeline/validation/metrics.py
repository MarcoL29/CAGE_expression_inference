from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


def _pearsonr(x: np.ndarray, y: np.ndarray, eps: float = 1e-8) -> float:
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    x = x - x.mean()
    y = y - y.mean()
    denom = (np.sqrt((x * x).sum()) * np.sqrt((y * y).sum())) + eps
    return float((x * y).sum() / denom)


def _ccc(x: np.ndarray, y: np.ndarray, eps: float = 1e-8) -> float:
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    mx, my = x.mean(), y.mean()
    vx, vy = x.var(), y.var()
    cov = np.mean((x - mx) * (y - my))
    return float((2 * cov) / (vx + vy + (mx - my) ** 2 + eps))


def compute_va_metrics(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    """pred/target: (N,2) ; mask: (N,) bool"""
    if pred.shape != target.shape:
        raise ValueError("pred and target must have same shape")
    if pred.shape[1] != 2:
        raise ValueError("Expected last dim == 2 (valence, arousal)")

    m = mask.astype(bool)
    pred = pred[m]
    target = target[m]
    if pred.shape[0] == 0:
        return {"ccc_v": 0.0, "ccc_a": 0.0, "ccc_mean": 0.0, "pcc_v": 0.0, "pcc_a": 0.0, "rmse_v": 0.0, "rmse_a": 0.0}

    pv, pa = pred[:, 0], pred[:, 1]
    tv, ta = target[:, 0], target[:, 1]

    ccc_v = _ccc(pv, tv)
    ccc_a = _ccc(pa, ta)
    pcc_v = _pearsonr(pv, tv)
    pcc_a = _pearsonr(pa, ta)
    rmse_v = float(np.sqrt(np.mean((pv - tv) ** 2)))
    rmse_a = float(np.sqrt(np.mean((pa - ta) ** 2)))

    return {
        "ccc_v": ccc_v,
        "ccc_a": ccc_a,
        "ccc_mean": float((ccc_v + ccc_a) / 2.0),
        "pcc_v": pcc_v,
        "pcc_a": pcc_a,
        "rmse_v": rmse_v,
        "rmse_a": rmse_a,
    }
