from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from affwild2_pipeline.data.affwild2_va_dataset import AffWild2VADataset
from affwild2_pipeline.data.transforms import build_train_transform, build_val_transform
from affwild2_pipeline.models.temporal_maxvit import TemporalMaxViT
from affwild2_pipeline.training.checkpointing import save_checkpoint
from affwild2_pipeline.training.losses import CCCLoss, masked_mse
from affwild2_pipeline.training.optimizer import build_cosine_annealing_scheduler, build_optimizer
from affwild2_pipeline.utils.config import load_json, load_train_config
from affwild2_pipeline.utils.logging import setup_logging
from affwild2_pipeline.utils.seed import seed_everything
from affwild2_pipeline.validation.metrics import compute_va_metrics


def _collate(batch):
    frames = torch.stack([b["frames"] for b in batch], dim=0)
    targets = torch.stack([b["targets"] for b in batch], dim=0)
    mask = torch.stack([b["mask"] for b in batch], dim=0)
    meta = {
        "video_id": [b["meta"]["video_id"] for b in batch],
        "start": torch.tensor([b["meta"]["start"] for b in batch], dtype=torch.int64),
    }
    return {"frames": frames, "targets": targets, "mask": mask, "meta": meta}


@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
    amp: bool,
    ccc_loss: CCCLoss,
) -> Tuple[Dict[str, float], float]:
    model.eval()
    amp_enabled = bool(amp and device.type == "cuda")
    preds = []
    gts = []
    masks = []
    loss_sum = 0.0
    for batch in tqdm(loader, desc="val", leave=False):
        frames = batch["frames"].to(device)
        targets_t = batch["targets"].to(device)
        mask_t = batch["mask"].to(device)

        targets = targets_t.detach().cpu().numpy()
        mask = mask_t.detach().cpu().numpy().astype(bool)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled):
            pred_t = model(frames)
            pred = pred_t.detach().cpu().numpy()

            # Mirror `models/AffectNet8_Maxvit_VA/train.py` loss structure:
            # 3*MSE(v) + 3*MSE(a) + CCC(v) + CCC(a)
            loss_mse_v = masked_mse(pred_t[..., 0:1], targets_t[..., 0:1], mask_t)
            loss_mse_a = masked_mse(pred_t[..., 1:2], targets_t[..., 1:2], mask_t)
            loss_ccc_v = ccc_loss(pred_t[..., 0:1], targets_t[..., 0:1], mask_t)
            loss_ccc_a = ccc_loss(pred_t[..., 1:2], targets_t[..., 1:2], mask_t)
            loss = 3.0 * loss_mse_v + 3.0 * loss_mse_a + loss_ccc_v + loss_ccc_a
            loss_sum += float(loss.detach().cpu())

        preds.append(pred.reshape(-1, 2))
        gts.append(targets.reshape(-1, 2))
        masks.append(mask.reshape(-1))

    pred_all = np.concatenate(preds, axis=0)
    gt_all = np.concatenate(gts, axis=0)
    mask_all = np.concatenate(masks, axis=0).astype(bool)
    metrics = compute_va_metrics(pred_all, gt_all, mask_all)
    loss_mean = loss_sum / max(1, len(loader))
    return metrics, float(loss_mean)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to config JSON")
    ap.add_argument("--device", default="cuda", help="cuda or cpu")
    args = ap.parse_args()

    cfg = load_train_config(args.config)
    seed_everything(cfg.seed)

    output_dir = Path(cfg.output_dir)
    run_dir = output_dir / "runs" / cfg.run_name
    ckpt_dir = output_dir / "checkpoints"
    logger = setup_logging(run_dir, name="train")

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    logger.info(f"Device: {device}")
    logger.info(f"Run dir: {run_dir}")

    amp_enabled = bool(cfg.amp and device.type == "cuda")

    train_ds = AffWild2VADataset(
        annotation_root=cfg.annotation_root,
        frames_root=cfg.frames_root,
        split="train",
        seq_len=cfg.seq_len,
        stride=cfg.train_stride,
        transform=build_train_transform(cfg.image_size),
        image_size=cfg.image_size,
        drop_last=True,
        strict_alignment=cfg.strict_alignment,
    )
    val_ds = AffWild2VADataset(
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

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        drop_last=True,
        collate_fn=_collate,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        drop_last=False,
        collate_fn=_collate,
    )

    model = TemporalMaxViT(
        image_size=cfg.image_size,
        max_seq_len=cfg.seq_len,
        backbone_weights=cfg.backbone_weights,
    )
    model.to(device)

    optimizer = build_optimizer(model, lr=cfg.lr, weight_decay=cfg.weight_decay)
    # Match `models/AffectNet8_Maxvit_Combined/train.py` behavior:
    # CosineAnnealingLR(T_max=BATCHSIZE*NUM_EPOCHS) stepped every iteration.
    scheduler = build_cosine_annealing_scheduler(optimizer, t_max=cfg.batch_size * cfg.epochs)

    ccc_loss = CCCLoss()

    # Mirror reference style: always create GradScaler; disable it when not using CUDA AMP.
    scaler = torch.amp.GradScaler() if amp_enabled else torch.amp.GradScaler(enabled=False)
    best = -1e9

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        for step, batch in enumerate(tqdm(train_loader, desc=f"epoch {epoch}/{cfg.epochs}"), start=1):
            frames = batch["frames"].to(device, non_blocking=True)
            targets = batch["targets"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled):
                pred = model(frames)
                loss_mse_v = masked_mse(pred[..., 0:1], targets[..., 0:1], mask)
                loss_mse_a = masked_mse(pred[..., 1:2], targets[..., 1:2], mask)
                loss_ccc_v = ccc_loss(pred[..., 0:1], targets[..., 0:1], mask)
                loss_ccc_a = ccc_loss(pred[..., 1:2], targets[..., 1:2], mask)
                loss = 3.0 * loss_mse_v + 3.0 * loss_mse_a + loss_ccc_v + loss_ccc_a

            scaler.scale(loss).backward()
            if cfg.grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
            scheduler.step()
            scaler.step(optimizer)
            scaler.update()

            train_loss_sum += float(loss.detach().cpu())
            if step % cfg.log_every == 0:
                logger.info(
                    f"epoch={epoch} step={step}/{len(train_loader)} loss={train_loss_sum/step:.4f} lr={scheduler.get_last_lr()[0]:.2e}"
                )

        metrics, val_loss = evaluate(model, val_loader, device, amp=cfg.amp, ccc_loss=ccc_loss)
        train_loss = train_loss_sum / max(1, len(train_loader))
        logger.info(
            f"epoch={epoch} train_loss={train_loss:.4f} val_loss={val_loss:.4f} lr={scheduler.get_last_lr()[0]:.2e}"
        )

        score = float(metrics["ccc_mean"])
        logger.info(
            f"epoch={epoch} val ccc_mean={metrics['ccc_mean']:.4f} (v={metrics['ccc_v']:.4f}, a={metrics['ccc_a']:.4f}) rmse(v,a)=({metrics['rmse_v']:.4f},{metrics['rmse_a']:.4f})"
        )

        save_checkpoint(
            ckpt_dir / "last.pt",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            best_score=max(best, score),
            config=load_json(args.config),
        )
        if score > best:
            best = score
            save_checkpoint(
                ckpt_dir / "best.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_score=best,
                config=load_json(args.config),
            )
            logger.info(f"New best checkpoint saved (ccc_mean={best:.4f})")


if __name__ == "__main__":
    main()
