# check_pred_values.py
import nibabel as nib
import numpy as np
from pathlib import Path

p = Path(r"D:\izzivAMS\AMS-Izziv-Peter\outputs\stunet\predictions")
preds = sorted([f for f in p.iterdir() if f.name.endswith("_pred.nii.gz")])
if not preds:
    print("Ni predikcij.")
    raise SystemExit
f = preds[0]
arr = nib.load(str(f)).get_fdata()
print("File:", f.name)
print("Shape:", arr.shape)
print("Unique values (up to 20):", np.unique(arr).tolist()[:20])
print("Min,Max,Mean:", float(arr.min()), float(arr.max()), float(arr.mean()))
# count nonzero voxels
print("Nonzero voxels:", int((arr!=0).sum()), " / ", arr.size)
