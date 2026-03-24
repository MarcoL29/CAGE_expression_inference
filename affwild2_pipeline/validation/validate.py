from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from affwild2_pipeline.data.affwild2_va_dataset import AffWild2VADataset
from affwild2_pipeline.data.transforms import build_val_transform
from affwild2_pipeline.models.temporal_maxvit import TemporalMaxViT
from affwild2_pipeline.training.checkpointing import load_checkpoint
from affwild2_pipeline.utils.config import load_train_config, load_json, save_json
from affwild2_pipeline.utils.logging import setup_logging
from affwild2_pipeline.validation.metrics import compute_va_metrics


def _aggregate_predictions(
    dataset: AffWild2VADataset,
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    amp: bool,
):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    # For each video, maintain per-frame sum and counts
    sum_pred: Dict[str, np.ndarray] = {}
    cnt_pred: Dict[str, np.ndarray] = {}
    gt: Dict[str, np.ndarray] = {}
    mask_gt: Dict[str, np.ndarray] = {}

    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="validate", leave=False):
            frames = batch["frames"].to(device)
            meta = batch["meta"]

            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(amp and device.type == "cuda")):
                pred = model(frames).detach().cpu().numpy()  # (B,T,2)

            b, t, _ = pred.shape
            for i in range(b):
                video_id = meta["video_id"][i]
                start = int(meta["start"][i])
                record = dataset.get_video_record(video_id)

                if video_id not in sum_pred:
                    n = record.va.shape[0]
                    sum_pred[video_id] = np.zeros((n, 2), dtype=np.float32)
                    cnt_pred[video_id] = np.zeros((n, 1), dtype=np.float32)
                    gt[video_id] = record.va.cpu().numpy().astype(np.float32)
                    mask_gt[video_id] = record.valid.cpu().numpy().astype(bool)

                end = min(start + t, sum_pred[video_id].shape[0])
                pred_slice = pred[i, : (end - start)]
                sum_pred[video_id][start:end] += pred_slice
                cnt_pred[video_id][start:end] += 1.0

    # finalize
    all_pred = []
    all_gt = []
    all_mask = []
    rows = []
    for video_id in sorted(sum_pred.keys()):
        p = sum_pred[video_id] / np.clip(cnt_pred[video_id], 1.0, None)
        y = gt[video_id]
        m = mask_gt[video_id]
        all_pred.append(p)
        all_gt.append(y)
        all_mask.append(m)
        for idx in range(p.shape[0]):
            rows.append((video_id, idx + 1, float(p[idx, 0]), float(p[idx, 1]), float(y[idx, 0]), float(y[idx, 1]), bool(m[idx])))

    pred_all = np.concatenate(all_pred, axis=0)
    gt_all = np.concatenate(all_gt, axis=0)
    mask_all = np.concatenate(all_mask, axis=0)
    return pred_all, gt_all, mask_all, rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to training config JSON")
    ap.add_argument("--checkpoint", required=True, help="Path to checkpoint .pt")
    ap.add_argument("--device", default="cuda", help="cuda or cpu")
    args = ap.parse_args()

    cfg = load_train_config(args.config)
    out_dir = Path("affwild2_pipeline/validation/results") / cfg.run_name
    logger = setup_logging(out_dir, name="validate")

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    logger.info(f"Device: {device}")

    ds = AffWild2VADataset(
        annotation_root=cfg.annotation_root,
        frames_root=cfg.frames_root,
        split="validate",
        seq_len=cfg.seq_len,
        stride=cfg.val_stride,
        transform=build_val_transform(cfg.image_size),
        image_size=cfg.image_size,
        drop_last=False,
        strict_alignment=cfg.strict_alignment,
    )

    model = TemporalMaxViT(
        image_size=cfg.image_size,
        max_seq_len=cfg.seq_len,
        backbone_weights=cfg.backbone_weights,
    )
    model.to(device)
    load_checkpoint(args.checkpoint, model=model, map_location=str(device))

    pred, target, mask, rows = _aggregate_predictions(
        dataset=ds,
        model=model,
        device=device,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        amp=cfg.amp,
    )

    metrics = compute_va_metrics(pred, target, mask)
    logger.info(f"CCC mean: {metrics['ccc_mean']:.4f} (V={metrics['ccc_v']:.4f}, A={metrics['ccc_a']:.4f})")

    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.json"
    pred_path = out_dir / "predictions.csv"

    with pred_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["video_id", "frame_idx", "pred_valence", "pred_arousal", "gt_valence", "gt_arousal", "valid"])
        for r in rows:
            w.writerow(r)

    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"Wrote: {pred_path}")
    logger.info(f"Wrote: {metrics_path}")


if __name__ == "__main__":
    main()
