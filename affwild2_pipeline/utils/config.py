from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


def load_json(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


@dataclass(frozen=True)
class TrainConfig:
    annotation_root: str
    frames_root: str
    run_name: str = "affwild2_temporal_vit"

    # Model / backbone
    backbone_weights: str | None = "DEFAULT"  # torchvision MaxViT-T weights: 'DEFAULT' or null

    image_size: int = 112
    seq_len: int = 32
    train_stride: int = 16
    val_stride: int = 8

    batch_size: int = 8
    num_workers: int = 4
    pin_memory: bool = True

    epochs: int = 20
    lr: float = 1e-4
    weight_decay: float = 0.01
    warmup_epochs: int = 0
    grad_clip_norm: float = 0.0

    amp: bool = True
    seed: int = 1337

    log_every: int = 50
    output_dir: str = "affwild2_pipeline/training"

    strict_alignment: bool = False

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "TrainConfig":
        return TrainConfig(**d)


def load_train_config(path: str | Path, overrides: Optional[Dict[str, Any]] = None) -> TrainConfig:
    data = load_json(path)
    if overrides:
        data = {**data, **overrides}
    return TrainConfig.from_dict(data)
