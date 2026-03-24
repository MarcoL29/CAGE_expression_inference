from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torchvision.models as tv_models


def _infer_maxvit_embed_dim(model: nn.Module) -> int:
    """Infer the feature dimension produced by MaxViT before its classification head."""
    classifier = getattr(model, "classifier", None)
    if classifier is None:
        raise ValueError("Expected torchvision MaxViT model to have a .classifier")

    last_linear: Optional[nn.Linear] = None
    for m in classifier.modules():
        if isinstance(m, nn.Linear):
            last_linear = m
    if last_linear is None:
        raise ValueError("Could not infer MaxViT embed_dim from classifier")
    return int(last_linear.in_features)


class MaxViTFrameEncoder(nn.Module):
    """Frame encoder using torchvision's MaxViT-T.

    Returns a per-image embedding of shape (B, D) by replacing the classifier head
    with global pooling + flatten.
    """

    def __init__(self, weights: str | None = "DEFAULT") -> None:
        super().__init__()
        w = weights
        ckpt_path: Path | None = None
        if isinstance(w, str):
            if w.lower() == "none":
                w = None
            else:
                p = Path(w)
                if p.is_file():
                    ckpt_path = p
                    # When loading a checkpoint, don't also request torchvision weights.
                    w = None

        backbone = tv_models.maxvit_t(weights=w)
        embed_dim = _infer_maxvit_embed_dim(backbone)

        backbone.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

        if ckpt_path is not None:
            state = torch.load(str(ckpt_path), map_location="cpu")
            # Support both raw state_dict saves and our checkpoint payload format.
            if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
                state = state["model"]
            if not isinstance(state, dict):
                raise ValueError(f"Unsupported checkpoint format at {ckpt_path}")
            backbone.load_state_dict(state, strict=False)

        self.backbone = backbone
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class TemporalMaxViT(nn.Module):
    """Temporal MaxViT: MaxViT-T frame encoder + temporal transformer over frame embeddings.

    Input: frames (B, T, C, H, W)
    Output: va (B, T, 2)
    """

    def __init__(
        self,
        image_size: int = 112,
        backbone_weights: str | None = None,
        temporal_depth: int = 4,
        temporal_heads: int = 4,
        max_seq_len: int = 64,
        dropout: float = 0.1,
        head_dropout: float = 0.3,
    ) -> None:
        super().__init__()

        # Match the repository's AffectNet scripts: torchvision.models.maxvit_t
        # `image_size` is kept for API compatibility; resizing is handled by the dataset transforms.
        _ = image_size
        weights = backbone_weights if backbone_weights is not None else "DEFAULT"
        self.frame_encoder = MaxViTFrameEncoder(weights=weights)
        embed_dim = int(self.frame_encoder.embed_dim)

        self.temporal_pos = nn.Parameter(torch.zeros(1, max_seq_len, embed_dim))
        nn.init.trunc_normal_(self.temporal_pos, std=0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=temporal_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.temporal = nn.TransformerEncoder(enc_layer, num_layers=temporal_depth)

        # LayerNorm -> Linear -> Tanh -> Dropout -> Linear
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.Tanh(),
            nn.Dropout(p=float(head_dropout)),
            nn.Linear(embed_dim, 2, bias=False),
        )

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        if frames.ndim != 5:
            raise ValueError("frames must have shape (B, T, C, H, W)")
        b, t, c, h, w = frames.shape

        x = frames.reshape(b * t, c, h, w)
        feats = self.frame_encoder(x)  # (B*T, D)
        feats = feats.reshape(b, t, -1)  # (B, T, D)

        if t > self.temporal_pos.shape[1]:
            raise ValueError(f"seq_len={t} exceeds max_seq_len={self.temporal_pos.shape[1]}")

        feats = feats + self.temporal_pos[:, :t]
        feats = self.temporal(feats)
        out = self.head(feats)  # (B, T, 2)
        return out
