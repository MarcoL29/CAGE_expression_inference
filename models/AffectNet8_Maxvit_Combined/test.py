import os
import math
import pandas as pd
import numpy as np
from typing import Tuple, Dict, List

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

# ========= Paths (reuse your existing ones) =========
IMAGE_FOLDER_TEST = "C:/Users/marco/Documents/Datasets/AffectNet/val_set/val_set/images/"
valid_annotations_path = "C:/Users/marco/Documents/Datasets/AffectNet/val_set_annotation_without_lnd.csv"
CHECKPOINT_PATH = "./model.pt"
SAVE_PRED_CSV = "./val_cls_reg_predictions.csv"

# ========= Eval params =========
BATCHSIZE = 128
NUM_WORKERS = 0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 8

# ========= Transforms (match your validation) =========
transform_valid = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ========= Dataset =========
class ValDataset(Dataset):
    """
    Expects CSV with at least columns: ['number','exp','valence','arousal'] in that order (like your training).
    Returns (image, class_idx, labels[valence,arousal], number)
    """
    def __init__(self, dataframe: pd.DataFrame, root_dir: str, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_id = int(row["number"])
        img_path = os.path.join(self.root_dir, f"{img_id}.jpg")
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Missing image: {img_path}")
        image = Image.open(img_path).convert("RGB")

        cls = int(row["exp"])
        labels = torch.tensor(row.iloc[2:4].values, dtype=torch.float32)  # valence, arousal

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(cls, dtype=torch.long), labels, img_id

# ========= Metrics =========
@torch.no_grad()
def concordance_cc(pred: torch.Tensor, true: torch.Tensor) -> float:
    """
    CCC across all samples. pred/true shape [N].
    """
    x = pred.detach().float().cpu()
    y = true.detach().float().cpu()
    mu_x = x.mean()
    mu_y = y.mean()
    vx = x.var(unbiased=False)
    vy = y.var(unbiased=False)
    cov = ((x - mu_x) * (y - mu_y)).mean()
    ccc = (2 * cov) / (vx + vy + (mu_x - mu_y) ** 2 + 1e-8)
    return float(ccc.item())

@torch.no_grad()
def mse(pred: torch.Tensor, true: torch.Tensor) -> float:
    return float(torch.mean((pred - true) ** 2).item())

@torch.no_grad()
def mae(pred: torch.Tensor, true: torch.Tensor) -> float:
    return float(torch.mean(torch.abs(pred - true)).item())

def confusion_matrix_numpy(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm

def precision_recall_f1_from_cm(cm: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    with np.errstate(divide='ignore', invalid='ignore'):
        precision = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
        recall    = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
        f1        = np.where(precision + recall > 0, 2 * precision * recall / (precision + recall), 0.0)
    return precision, recall, f1

# ========= Model (match your training head: 10 outputs) =========
def build_model() -> nn.Module:
    model = models.maxvit_t(weights=None)  # weights loaded from checkpoint next
    block_channels = model.classifier[3].in_features
    model.classifier = nn.Sequential(
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.LayerNorm(block_channels),
        nn.Linear(block_channels, block_channels),
        nn.Tanh(),
        nn.Linear(block_channels, 10, bias=False),  # 8 classes + 2 regression
    )
    return model

# ========= Evaluation =========
def main():
    print("==> Loading validation annotations...")
    valid_df = pd.read_csv(valid_annotations_path)

    print("==> Building dataset and dataloader...")
    val_dataset = ValDataset(valid_df, IMAGE_FOLDER_TEST, transform=transform_valid)
    val_loader = DataLoader(
        val_dataset, batch_size=BATCHSIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True
    )

    print("==> Building model and loading weights...")
    model = build_model().to(DEVICE)
    state = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()

    all_ids: List[int] = []
    all_true_cls: List[int] = []
    all_pred_cls: List[int] = []
    all_true_val: List[float] = []
    all_true_aro: List[float] = []
    all_pred_val: List[float] = []
    all_pred_aro: List[float] = []
    all_probs: List[np.ndarray] = []

    autocast_kwargs = {}
    if DEVICE.type == "cuda":
        autocast_kwargs = dict(device_type="cuda", dtype=torch.float16)

    print("==> Running inference on validation set...")
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Valid inference")
        for images, cls_true, labels, img_ids in pbar:
            images = images.to(DEVICE, non_blocking=True)
            cls_true = cls_true.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)  # [B,2] (val, aro)

            ctx = torch.autocast(**autocast_kwargs) if autocast_kwargs else torch.cuda.amp.autocast(enabled=False)
            with ctx:
                outputs = model(images)                # [B,10]
                logits_cls = outputs[:, :NUM_CLASSES] # [B,8]
                reg = outputs[:, NUM_CLASSES:]        # [B,2] -> (val_pred, aro_pred)

            probs = torch.softmax(logits_cls.float(), dim=1)  # [B,8] in fp32 for stability
            preds = torch.argmax(probs, dim=1)                # [B]

            # Accumulate
            all_ids.extend([int(i) for i in img_ids])
            all_true_cls.extend(cls_true.cpu().tolist())
            all_pred_cls.extend(preds.cpu().tolist())
            all_true_val.extend(labels[:, 0].float().cpu().tolist())
            all_true_aro.extend(labels[:, 1].float().cpu().tolist())
            all_pred_val.extend(reg[:, 0].float().cpu().tolist())
            all_pred_aro.extend(reg[:, 1].float().cpu().tolist())
            all_probs.extend(probs.cpu().numpy())

    # ----- Classification metrics -----
    y_true = np.array(all_true_cls, dtype=np.int64)
    y_pred = np.array(all_pred_cls, dtype=np.int64)
    cm = confusion_matrix_numpy(y_true, y_pred, NUM_CLASSES)
    acc = (y_true == y_pred).mean() * 100.0
    per_class_acc = np.divide(np.diag(cm), cm.sum(axis=1, where=cm.sum(axis=1)!=0, initial=0), out=np.zeros(NUM_CLASSES), where=cm.sum(axis=1)!=0) * 100.0
    precision, recall, f1 = precision_recall_f1_from_cm(cm)
    macro_p = float(np.mean(precision) * 100.0)
    macro_r = float(np.mean(recall) * 100.0)
    macro_f1 = float(np.mean(f1) * 100.0)

    # ----- Regression metrics -----
    vt = torch.tensor(all_true_val)
    at = torch.tensor(all_true_aro)
    vp = torch.tensor(all_pred_val)
    ap = torch.tensor(all_pred_aro)

    val_mse = mse(vp, vt); val_mae = mae(vp, vt); val_ccc = concordance_cc(vp, vt)
    aro_mse = mse(ap, at); aro_mae = mae(ap, at); aro_ccc = concordance_cc(ap, at)

    # ----- Print summary -----
    print("\n===== Classification (8-way) =====")
    print(f"Accuracy: {acc:.2f}%")
    print(f"Macro Precision: {macro_p:.2f}% | Macro Recall: {macro_r:.2f}% | Macro F1: {macro_f1:.2f}%")
    print("Per-class accuracy (%): " + ", ".join([f"C{i}: {a:.2f}" for i, a in enumerate(per_class_acc)]))
    print("\nConfusion Matrix (rows=true, cols=pred):")
    with np.printoptions(linewidth=120):
        print(cm)

    print("\n===== Regression =====")
    print(f"Valence -> MSE: {val_mse:.6f} | MAE: {val_mae:.6f} | CCC: {val_ccc:.6f}")
    print(f"Arousal -> MSE: {aro_mse:.6f} | MAE: {aro_mae:.6f} | CCC: {aro_ccc:.6f}")

    # ----- Save CSV -----
    print(f"\n==> Saving predictions to {SAVE_PRED_CSV}")
    prob_cols = {f"prob_c{i}": col for i, col in enumerate(np.array(all_probs).T)}
    # Build dataframe row-wise
    df_rows = []
    for i in range(len(all_ids)):
        row = {
            "number": all_ids[i],
            "exp_true": int(all_true_cls[i]),
            "exp_pred": int(all_pred_cls[i]),
            "val_true": float(all_true_val[i]),
            "aro_true": float(all_true_aro[i]),
            "val_pred": float(all_pred_val[i]),
            "aro_pred": float(all_pred_aro[i]),
        }
        # add probs for this sample
        probs_i = all_probs[i]
        for c in range(NUM_CLASSES):
            row[f"prob_c{c}"] = float(probs_i[c])
        df_rows.append(row)

    out_df = pd.DataFrame(df_rows).sort_values("number")
    out_df.to_csv(SAVE_PRED_CSV, index=False)

if __name__ == "__main__":
    main()
