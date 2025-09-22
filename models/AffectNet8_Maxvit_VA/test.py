import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
from tqdm import tqdm
from sklearn.model_selection import train_test_split


# ====== Paths (reuse your existing ones) ======
IMAGE_FOLDER_TEST = "C:/Users/marco/Documents/Datasets/AffectNet/val_set/val_set/images/"
VALID_ANNOTATIONS_PATH = "C:/Users/marco/Documents/Datasets/AffectNet/val_set_annotation_without_lnd.csv"

# ====== Eval params ======
BATCHSIZE = 128
NUM_WORKERS = 0
CHECKPOINT_PATH = "./model.pt"     # path where you saved best model
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_PRED_CSV = "./val_predictions.csv"

# ====== Data ======
class ValDataset(Dataset):
    """
    Validation dataset that also returns the image ID (number) so we can save predictions.
    Assumes CSV columns (at least): ['number', 'exp', 'valence', 'arousal'] in that order.
    """
    def __init__(self, dataframe, root_dir, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_id = self.df.loc[idx, "number"]
        img_path = os.path.join(self.root_dir, f"{img_id}.jpg")
        image = Image.open(img_path).convert("RGB")

        # Column indices follow your training code: class @ 1, valence @ 2, arousal @ 3
        valence = torch.tensor(self.df.iloc[idx, 2], dtype=torch.float32)
        arousal = torch.tensor(self.df.iloc[idx, 3], dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, valence, arousal, int(img_id)

transform_valid = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

# ====== Metrics ======
@torch.no_grad()
def concordance_cc(pred: torch.Tensor, true: torch.Tensor) -> float:
    """
    Concordance Correlation Coefficient (CCC) computed over all samples.
    pred, true: shape [N], float tensors on any device.
    Returns float.
    """
    # convert to float64 on CPU for numerical stability
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
def mae(pred: torch.Tensor, true: torch.Tensor) -> float:
    return float(torch.mean(torch.abs(pred - true)).item())

@torch.no_grad()
def mse(pred: torch.Tensor, true: torch.Tensor) -> float:
    return float(torch.mean((pred - true) ** 2).item())

@torch.no_grad()
def rmse(pred: torch.Tensor, true: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean((pred - true) ** 2)).item())

@torch.no_grad()
def pcc(pred: torch.Tensor, true: torch.Tensor) -> float:
    """
    Pearson correlation coefficient between pred and true.
    """
    x = pred.detach().float().cpu()
    y = true.detach().float().cpu()
    vx = x - x.mean()
    vy = y - y.mean()
    corr = (vx * vy).sum() / (torch.sqrt((vx ** 2).sum()) * torch.sqrt((vy ** 2).sum()) + 1e-8)
    return float(corr.item())

# ====== Model ======
def build_model():
    # Start from torchvision's MaxViT-T backbone
    model = models.maxvit_t(weights=None)  # we will load our fine-tuned weights next
    # Match the exact classifier you used for final training (2 outputs)
    block_channels = model.classifier[3].in_features
    model.classifier = nn.Sequential(
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.LayerNorm(block_channels),
        nn.Linear(block_channels, block_channels),
        nn.Tanh(),
        nn.Dropout(0.3),
        nn.Linear(block_channels, 2, bias=False),
    )
    return model

def main():
    print("==> Loading validation annotations...")
    valid_df = pd.read_csv(VALID_ANNOTATIONS_PATH)

    df_val, df_test = train_test_split(valid_df, test_size=0.5, random_state=42)
    df_val  = df_val.reset_index().drop('index', axis=1)
    df_test = df_test.reset_index().drop('index', axis=1)

    print("==> Building dataset and dataloader...")
    val_dataset = ValDataset(df_test, IMAGE_FOLDER_TEST, transform=transform_valid)
    val_loader = DataLoader(
        val_dataset, batch_size=BATCHSIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True
    )

    print("==> Building model and loading weights...")
    model = build_model().to(DEVICE)
    state = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()

    all_ids = []
    all_val_true = []
    all_aro_true = []
    all_val_pred = []
    all_aro_pred = []

    print("==> Running inference on validation set...")
    autocast_kwargs = {}
    if DEVICE.type == "cuda":
        autocast_kwargs = dict(device_type="cuda", dtype=torch.float16)

    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Valid inference")
        for images, v_true, a_true, img_ids in pbar:
            images = images.to(DEVICE, non_blocking=True)
            v_true = v_true.to(DEVICE, non_blocking=True)
            a_true = a_true.to(DEVICE, non_blocking=True)

            # Mixed precision on CUDA; plain on CPU
            ctx = torch.autocast(**autocast_kwargs) if autocast_kwargs else torch.cuda.amp.autocast(enabled=False)
            with ctx:
                outputs = model(images)           # [B, 2]
                v_pred = outputs[:, 0].float()    # promote to fp32 for metrics
                a_pred = outputs[:, 1].float()

            all_ids.extend([int(i) for i in img_ids])
            all_val_true.append(v_true.float().cpu())
            all_aro_true.append(a_true.float().cpu())
            all_val_pred.append(v_pred.cpu())
            all_aro_pred.append(a_pred.cpu())

    # Concatenate
    val_true = torch.cat(all_val_true, dim=0)
    aro_true = torch.cat(all_aro_true, dim=0)
    val_pred = torch.cat(all_val_pred, dim=0)
    aro_pred = torch.cat(all_aro_pred, dim=0)

    # Metrics
    val_mse = mse(val_pred, val_true)
    aro_mse = mse(aro_pred, aro_true)
    val_mae = mae(val_pred, val_true)
    aro_mae = mae(aro_pred, aro_true)
    val_rmse = rmse(val_pred, val_true)
    aro_rmse = rmse(aro_pred, aro_true)
    val_pcc = pcc(val_pred, val_true)
    aro_pcc = pcc(aro_pred, aro_true)
    val_ccc = concordance_cc(val_pred, val_true)
    aro_ccc = concordance_cc(aro_pred, aro_true)

    print("\n==== Validation Metrics ====")
    print(f"Valence  - MSE: {val_mse:.6f} | RMSE: {val_rmse:.6f} | MAE: {val_mae:.6f} | PCC: {val_pcc:.6f} | CCC: {val_ccc:.6f}")
    print(f"Arousal  - MSE: {aro_mse:.6f} | RMSE: {aro_rmse:.6f} | MAE: {aro_mae:.6f} | PCC: {aro_pcc:.6f} | CCC: {aro_ccc:.6f}")

    # Save per-sample predictions
    print(f"\n==> Saving predictions to {SAVE_PRED_CSV}")
    out_df = pd.DataFrame({
        "number": all_ids,
        "val_true": val_true.numpy(),
        "aro_true": aro_true.numpy(),
        "val_pred": val_pred.numpy(),
        "aro_pred": aro_pred.numpy(),
    }).sort_values("number")
    out_df.to_csv(SAVE_PRED_CSV, index=False)

if __name__ == "__main__":
    main()
