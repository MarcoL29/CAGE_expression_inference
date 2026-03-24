import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class AffectNetDataset(Dataset):
    def __init__(self, df: pd.DataFrame, image_root: str, transform):
        self.df = df.reset_index(drop=True)
        self.image_root = image_root
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        number = self.df["number"].iloc[idx]
        path = os.path.join(self.image_root, f"{number}.jpg")
        img = Image.open(path).convert("RGB")
        x = self.transform(img)
        va = torch.tensor(self.df.iloc[idx, 2:4].values, dtype=torch.float32)
        return x, va


def build_maxvit_combined(num_classes: int) -> nn.Module:
    model = models.maxvit_t(weights=None)
    block_channels = model.classifier[3].in_features
    model.classifier = nn.Sequential(
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.LayerNorm(block_channels),
        nn.Linear(block_channels, block_channels),
        nn.Tanh(),
        nn.Linear(block_channels, num_classes + 2, bias=False),
    )
    return model


def ccc_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = y_true.astype(np.float64)
    y_pred = y_pred.astype(np.float64)
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    cov = np.mean((y_true - mean_true) * (y_pred - mean_pred))
    return (2.0 * cov) / (var_true + var_pred + (mean_true - mean_pred) ** 2 + 1e-12)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate AffectNet MaxViT_combined checkpoint on valence/arousal regression."
    )
    p.add_argument(
        "--annotations_csv",
        type=str,
        default="C:/Users/marco/Documents/AI.EVENT/Datasets/Other_Datasets/AffectNet/val_set_annotation_without_lnd.csv",
        help="Path to AffectNet annotation CSV (e.g. val_set_annotation_without_lnd.csv)",
    )
    p.add_argument(
        "--image_root",
        type=str,
        default="C:/Users/marco/Documents/AI.EVENT/Datasets/Other_Datasets/AffectNet/val_set/val_set/images/",
        help="Folder containing images named <number>.jpg",
    )
    p.add_argument(
        "--checkpoint",
        type=str,
        default="./model_affectnet_maxvit_combined.pt",
        help="Path to AffectNet model state_dict (model.pt)",
    )
    p.add_argument(
        "--num_classes",
        type=int,
        default=8,
        choices=[7, 8],
        help="How many discrete classes the checkpoint was trained with.",
    )
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--gpu", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.annotations_csv)

    tfm = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    ds = AffectNetDataset(df, args.image_root, tfm)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    device = (
        torch.device("cuda" if args.gpu is None else f"cuda:{args.gpu}")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    model = build_maxvit_combined(num_classes=args.num_classes)
    state = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()

    all_true: list[np.ndarray] = []
    all_pred: list[np.ndarray] = []

    with torch.no_grad():
        for x, va in tqdm(loader, desc="eval affectnet"):
            x = x.to(device, non_blocking=True)
            va = va.to(device, non_blocking=True)
            out = model(x)
            out_reg = out[:, args.num_classes :]
            all_true.append(va.detach().cpu().numpy())
            all_pred.append(out_reg.detach().cpu().numpy())

    y_true = np.concatenate(all_true, axis=0)
    y_pred = np.concatenate(all_pred, axis=0)
    err = y_pred - y_true

    mse = np.mean(err**2, axis=0)
    mae = np.mean(np.abs(err), axis=0)
    rmse = np.sqrt(mse)
    ccc = np.array([
        ccc_np(y_true[:, 0], y_pred[:, 0]),
        ccc_np(y_true[:, 1], y_pred[:, 1]),
    ])

    print("\n--- AffectNet V/A evaluation ---")
    print(f"samples: {y_true.shape[0]}")
    print(f"MAE  (val, aro): {mae[0]:.4f}, {mae[1]:.4f} | mean {mae.mean():.4f}")
    print(f"MSE  (val, aro): {mse[0]:.4f}, {mse[1]:.4f} | mean {mse.mean():.4f}")
    print(f"RMSE (val, aro): {rmse[0]:.4f}, {rmse[1]:.4f} | mean {rmse.mean():.4f}")
    print(f"CCC  (val, aro): {ccc[0]:.4f}, {ccc[1]:.4f} | mean {ccc.mean():.4f}")


if __name__ == "__main__":
    main()
