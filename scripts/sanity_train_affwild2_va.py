from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Allow running this script directly without installing the package or setting PYTHONPATH.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from affwild2_pipeline.data.affwild2_va_dataset import AffWild2VADataset
from affwild2_pipeline.data.transforms import build_train_transform, build_val_transform
from affwild2_pipeline.models.temporal_maxvit import TemporalMaxViT
from affwild2_pipeline.training.losses import CCCLoss, masked_mse
from affwild2_pipeline.training.optimizer import build_cosine_annealing_scheduler, build_optimizer
from affwild2_pipeline.utils.config import load_train_config
from affwild2_pipeline.utils.seed import seed_everything


def _collate(batch):
    frames = torch.stack([b["frames"] for b in batch], dim=0)
    targets = torch.stack([b["targets"] for b in batch], dim=0)
    mask = torch.stack([b["mask"] for b in batch], dim=0)
    meta = {
        "video_id": [b["meta"]["video_id"] for b in batch],
        "start": torch.tensor([b["meta"]["start"] for b in batch], dtype=torch.int64),
    }
    return {"frames": frames, "targets": targets, "mask": mask, "meta": meta}


def _va_loss(pred: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor, ccc_loss: CCCLoss) -> torch.Tensor:
    # Mirror AffectNet VA loss structure used in this repo.
    loss_mse_v = masked_mse(pred[..., 0:1], targets[..., 0:1], mask)
    loss_mse_a = masked_mse(pred[..., 1:2], targets[..., 1:2], mask)
    loss_ccc_v = ccc_loss(pred[..., 0:1], targets[..., 0:1], mask)
    loss_ccc_a = ccc_loss(pred[..., 1:2], targets[..., 1:2], mask)
    return 3.0 * loss_mse_v + 3.0 * loss_mse_a + loss_ccc_v + loss_ccc_a


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sanity training run for Aff-Wild2 VA temporal MaxViT")
    p.add_argument(
        "--config",
        type=str,
        default=str(_REPO_ROOT / "affwild2_pipeline" / "training" / "configs" / "maxvit_t_224.json"),
        help="Path to affwild2_pipeline train config JSON",
    )
    p.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    p.add_argument("--steps", type=int, default=5, help="Number of train iterations to run")
    p.add_argument("--val_batches", type=int, default=1, help="Number of validation batches to run")

    p.add_argument("--batch_size", type=int, default=2, help="Override batch size for sanity")
    p.add_argument("--seq_len", type=int, default=8, help="Override seq_len for sanity")
    p.add_argument("--image_size", type=int, default=None, help="Override image_size (defaults to config)")
    p.add_argument("--train_stride", type=int, default=None)
    p.add_argument("--val_stride", type=int, default=None)

    p.add_argument(
        "--backbone_weights",
        type=str,
        default="none",
        help="Override backbone_weights. Use 'DEFAULT' to use torchvision weights, 'none' for random init, or a path to a .pt",
    )
    p.add_argument("--no_amp", action="store_true", help="Disable AMP even on CUDA")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    overrides: dict[str, object] = {
        "batch_size": int(args.batch_size),
        "seq_len": int(args.seq_len),
        "backbone_weights": args.backbone_weights,
        "num_workers": 0,
        "pin_memory": bool(torch.cuda.is_available()),
        "strict_alignment": False,
    }
    if args.image_size is not None:
        overrides["image_size"] = int(args.image_size)
    if args.train_stride is not None:
        overrides["train_stride"] = int(args.train_stride)
    if args.val_stride is not None:
        overrides["val_stride"] = int(args.val_stride)

    cfg = load_train_config(args.config, overrides=overrides)
    seed_everything(cfg.seed)

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    amp_enabled = bool((not args.no_amp) and cfg.amp and device.type == "cuda")

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
    ).to(device)

    # Fail fast with a clear error if MaxViT doesn't accept the configured image size.
    try:
        with torch.no_grad():
            dummy = torch.zeros(
                1,
                cfg.seq_len,
                3,
                cfg.image_size,
                cfg.image_size,
                device=device,
                dtype=torch.float32,
            )
            _ = model(dummy)
    except RuntimeError as e:
        msg = str(e)
        if "torchvision" in msg.lower() or "maxvit" in msg.lower() or "reshape" in msg.lower():
            raise SystemExit(
                "TemporalMaxViT (torchvision maxvit_t) failed a preflight forward pass. "
                "This is usually caused by an unsupported image_size for MaxViT. "
                "Try running with `--image_size 224` (or use the config `affwild2_pipeline/training/configs/maxvit_t_224.json`).\n"
                f"Original error: {type(e).__name__}: {msg.splitlines()[-1]}"
            )
        raise

    optimizer = build_optimizer(model, lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = build_cosine_annealing_scheduler(optimizer, t_max=cfg.batch_size * max(1, args.steps))

    ccc_loss = CCCLoss()
    scaler = torch.amp.GradScaler(enabled=amp_enabled)

    model.train()

    print("--- sanity train ---")
    print(f"device={device} amp={amp_enabled}")
    print(f"batch_size={cfg.batch_size} seq_len={cfg.seq_len} image_size={cfg.image_size}")
    print(f"train_windows={len(train_ds)} val_windows={len(val_ds)}")

    train_iter = iter(train_loader)
    for step in range(1, args.steps + 1):
        batch = next(train_iter)
        frames = batch["frames"].to(device, non_blocking=True)
        targets = batch["targets"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled):
            pred = model(frames)
            loss = _va_loss(pred, targets, mask, ccc_loss)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        with torch.no_grad():
            valid_ratio = float(mask.float().mean().detach().cpu())
        print(
            f"step={step}/{args.steps} loss={float(loss.detach().cpu()):.4f} "
            f"valid_ratio={valid_ratio:.3f} lr={scheduler.get_last_lr()[0]:.2e} "
            f"pred_shape={tuple(pred.shape)}"
        )

    if args.val_batches > 0:
        print("--- sanity val ---")
        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for i, batch in enumerate(tqdm(val_loader, desc="val", total=min(args.val_batches, len(val_loader)))):
                if i >= args.val_batches:
                    break
                frames = batch["frames"].to(device, non_blocking=True)
                targets = batch["targets"].to(device, non_blocking=True)
                mask = batch["mask"].to(device, non_blocking=True)
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled):
                    pred = model(frames)
                    loss = _va_loss(pred, targets, mask, ccc_loss)
                val_loss_sum += float(loss.detach().cpu())
        print(f"val_loss_mean={val_loss_sum / max(1, args.val_batches):.4f}")


if __name__ == "__main__":
    main()
