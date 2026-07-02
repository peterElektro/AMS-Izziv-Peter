import numpy as np
import nibabel as nib
import torch
from pathlib import Path
import re

from models.stunet import STUNetLitePlus
from inference.sliding_window import sliding_window_inference


# -------------------------
# Load STU-Net model
# -------------------------
def load_model(model_path, device):
    model = STUNetLitePlus(in_channels=1, out_channels=1)
    ckpt = torch.load(model_path, map_location=device)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict) and "model_state" in ckpt:
        state_dict = ckpt["model_state"]
    else:
        state_dict = ckpt

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


# -------------------------
# Dice
# -------------------------
def dice(a, b):
    a = a.astype(bool)
    b = b.astype(bool)
    inter = (a & b).sum()
    return 2 * inter / (a.sum() + b.sum() + 1e-8)


# -------------------------
# Paths (Docker)
# -------------------------
model_path = Path("outputs/stunet/model_best.pth")
img_dir = Path("nnUNet_raw/Dataset501_ImageCAS/imagesTs")
gt_dir  = Path("nnUNet_raw/Dataset501_ImageCAS/labelsTs")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_model(model_path, device)

# -------------------------
# Collect images
# -------------------------
files = sorted(list(img_dir.glob("*.nii.gz")))
thr_list = np.linspace(0.05, 0.95, 19)
results = {thr: [] for thr in thr_list}

print(f"Found {len(files)} images in {img_dir}")

# -------------------------
# Main loop
# -------------------------
for f in files:
    print(f"\n=== Processing {f.name} ===")

    img = nib.load(str(f)).get_fdata().astype(np.float32)
    img = (img - img.mean()) / (img.std() + 1e-8)

    if img.ndim == 3:
        img = img[np.newaxis, ...]

    # STU-Net inference
    probs = sliding_window_inference(
        img,
        model,
        patch_size=(128, 128, 128),
        overlap=0.5,
        device=device
    )

    # Softmax
    exp = np.exp(probs - probs.max(axis=0, keepdims=True))
    soft = exp / exp.sum(axis=0, keepdims=True)

    # Foreground probability
    prob_fg = soft[0] if soft.shape[0] == 1 else soft[1]

    # -------------------------
    # Find GT file
    # -------------------------
    base = f.name.split("_")[0]  # "181_0000.nii.gz" → "181"
    candidates = list(gt_dir.glob(f"{base}*.nii.gz"))

    if not candidates:
        print(f"[WARNING] No GT for {f.name}")
        continue

    gt_path = candidates[0]
    gt = nib.load(str(gt_path)).get_fdata()

    if prob_fg.shape != gt.shape:
        print(f"[WARNING] Shape mismatch: {prob_fg.shape} vs {gt.shape}")
        continue

    # -------------------------
    # Threshold sweep
    # -------------------------
    for thr in thr_list:
        pred_bin = (prob_fg >= thr).astype(np.uint8)
        results[thr].append(dice(pred_bin, gt))


# -------------------------
# Summary
# -------------------------
best_thr, best_mean = None, -1

print("\n=== THRESHOLD SWEEP RESULTS ===")
for thr in thr_list:
    vals = results[thr]
    if not vals:
        continue
    mean = float(np.mean(vals))
    print(f"thr {thr:.2f} -> mean Dice {mean:.4f} (n={len(vals)})")
    if mean > best_mean:
        best_mean = mean
        best_thr = thr

print(f"\nBest threshold: {best_thr:.2f} (mean Dice {best_mean:.4f})")
