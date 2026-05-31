# eval_dice.py (robustna različica)
import os
import numpy as np
import nibabel as nib
from pathlib import Path

pred_dir = Path(r"D:\izzivAMS\AMS-Izziv-Peter\outputs\stunet\predictions")
gt_dir   = Path(r"D:\izzivAMS\AMS-Izziv-Peter\data\nnunet_raw\Dataset501_ImageCAS\labelsVal")

def dice(a,b):
    a = a.astype(bool); b = b.astype(bool)
    inter = (a & b).sum()
    return 2*inter / (a.sum() + b.sum() + 1e-8)

def find_gt_for_pred(pred_name):
    # pred_name like "161.img.nii_pred.nii.gz"
    base = pred_name.replace("_pred.nii.gz","")
    candidates = []

    # 1) try replacing .img with .label
    if ".img." in base:
        candidates.append(base.replace(".img.", ".label."))
        candidates.append(base.replace(".img.", "."))
    # 2) try base as-is and with common suffixes
    candidates.append(base)
    candidates.append(base + ".nii.gz")
    candidates.append(base + ".nii")
    # 3) if base starts with digits, try case_XXXX patterns
    import re
    m = re.match(r"^(\d+)", base)
    if m:
        num = int(m.group(1))
        candidates.append(f"case_{num:04d}.nii.gz")
        candidates.append(f"case_{num:04d}.nii")
    # 4) also try replacing ".img" suffix if present
    if base.endswith(".img"):
        candidates.append(base[:-4] + ".label")
    # search in gt_dir
    for c in candidates:
        for f in gt_dir.iterdir():
            if f.name == c or f.name.startswith(c):
                return f
    # fallback: try any GT that starts with the numeric id
    if m:
        prefix = m.group(1)
        for f in gt_dir.iterdir():
            if f.name.startswith(prefix):
                return f
    return None

scores = []
missing = []
for p in sorted(pred_dir.iterdir()):
    if not p.name.endswith("_pred.nii.gz"):
        continue
    gt_path = find_gt_for_pred(p.name)
    if gt_path is None:
        missing.append(p.name)
        continue
    pred = nib.load(str(p)).get_fdata()
    gt   = nib.load(str(gt_path)).get_fdata()
    if pred.shape != gt.shape:
        print(f"Shape mismatch: {p.name} vs {gt_path.name} -> pred {pred.shape}, gt {gt.shape} (skipping)")
        missing.append(p.name)
        continue
    d = dice(pred, gt)
    print(f"{p.name}  ->  {gt_path.name}  :  Dice={d:.4f}")
    scores.append(d)

arr = np.array(scores)
print("\nEvaluated:", len(arr))
print("Missing or skipped:", len(missing))
if missing:
    print("Examples missing/skipped:", missing[:10])
if len(arr):
    print("Dice mean: {:.4f}".format(arr.mean()))
    print("Dice std : {:.4f}".format(arr.std()))
    print("Dice min : {:.4f}".format(arr.min()))
    print("Dice max : {:.4f}".format(arr.max()))
else:
    print("No valid predictions found to evaluate.")
