# sweep_all.py
import os
import csv
import numpy as np
import nibabel as nib
import torch
from pathlib import Path
from run_inference import load_model, sliding_window_inference

# --- CONFIGURE HERE ---
INPUT_DIR  = Path(r"D:\izzivAMS\AMS-Izziv-Peter\data\nnunet_raw\Dataset501_ImageCAS\imagesVal")
GT_DIR     = Path(r"D:\izzivAMS\AMS-Izziv-Peter\data\nnunet_raw\Dataset501_ImageCAS\labelsVal")
MODEL_PATH = r"D:\izzivAMS\AMS-Izziv-Peter\outputs\stunet\model_final.pth"
OUT_DIR    = Path(r"D:\izzivAMS\AMS-Izziv-Peter\outputs\stunet\sweep_results")
PATCH_SIZE = (128,128,128)
OVERLAP    = 0.5
BATCH_SIZE = 1
THR_LIST   = np.linspace(0.05,0.95,19)  # thresholds to test
# -----------------------

OUT_DIR.mkdir(parents=True, exist_ok=True)
prob_dir = OUT_DIR / "prob_maps"
prob_dir.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

model = load_model(MODEL_PATH, "models.stunet", device)
model.eval()

def softmax_from_logits(logits):
    exp = np.exp(logits - logits.max(axis=0, keepdims=True))
    soft = exp / exp.sum(axis=0, keepdims=True)
    return soft

def dice(a,b):
    a = a.astype(bool); b = b.astype(bool)
    inter = (a & b).sum()
    return 2*inter / (a.sum() + b.sum() + 1e-8)

# collect prob maps and matching GT
pairs = []
for img_path in sorted(INPUT_DIR.glob("*.nii*")):
    stem = img_path.stem
    # find GT by prefix match
    candidates = [g for g in GT_DIR.iterdir() if g.name.startswith(stem)]
    if not candidates:
        # try numeric prefix
        import re
        m = re.match(r"^(\d+)", stem)
        if m:
            prefix = m.group(1)
            candidates = [g for g in GT_DIR.iterdir() if g.name.startswith(prefix)]
    if not candidates:
        print("No GT for", img_path.name, "skipping")
        continue
    gt_path = candidates[0]
    print("Processing", img_path.name, "->", gt_path.name)
    img = nib.load(str(img_path)).get_fdata().astype(np.float32)
    img_affine = nib.load(str(img_path)).affine
    img = (img - img.mean())/(img.std()+1e-8)
    if img.ndim == 3:
        img = img[np.newaxis,...]
    probs = sliding_window_inference(img, model, device, patch_size=PATCH_SIZE, overlap=OVERLAP, batch_size=BATCH_SIZE)
    if isinstance(probs, torch.Tensor):
        probs = probs.detach().cpu().numpy()
    # compute softmax if multi-channel logits
    if probs.shape[0] > 1:
        soft = softmax_from_logits(probs)
        prob_fg = soft[1]
    else:
        prob_fg = probs[0]
    # save prob map
    out_prob = prob_dir / (img_path.stem + "_prob_fg.nii.gz")
    nib.save(nib.Nifti1Image(prob_fg.astype(np.float32), img_affine), str(out_prob))
    pairs.append((out_prob, gt_path))
    print("Saved prob map", out_prob.name)

# threshold sweep
results = []
for thr in THR_LIST:
    thr_scores = []
    for prob_path, gt_path in pairs:
        prob = nib.load(str(prob_path)).get_fdata()
        gt = nib.load(str(gt_path)).get_fdata()
        if prob.shape != gt.shape:
            print("Shape mismatch", prob_path.name, gt_path.name, "skipping")
            continue
        pred = (prob >= thr).astype(np.uint8)
        thr_scores.append(dice(pred, gt))
    if thr_scores:
        mean = float(np.mean(thr_scores))
        std  = float(np.std(thr_scores))
        results.append((float(thr), mean, std, len(thr_scores)))
        print(f"thr {thr:.2f} -> mean {mean:.4f} std {std:.4f} n {len(thr_scores)}")

# save CSV and best threshold
csv_path = OUT_DIR / "threshold_sweep_results.csv"
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["threshold","mean_dice","std_dice","n"])
    for row in results:
        writer.writerow(row)

if results:
    best = max(results, key=lambda x: x[1])
    best_thr, best_mean = best[0], best[1]
    print("\nBest threshold:", best_thr, "mean Dice:", best_mean)
    with open(OUT_DIR / "best_threshold.txt", "w") as f:
        f.write(f"{best_thr},{best_mean}\n")
else:
    print("No results computed. Check pairs and shapes.")
