import nibabel as nib, numpy as np, pathlib
p = pathlib.Path(r"D:\izzivAMS\AMS-Izziv-Peter\outputs\stunet\predictions_thresh04\161.img.nii_pred.nii.gz")
arr = nib.load(str(p)).get_fdata()
print("File:", p.name)
print("Shape:", arr.shape)
print("Unique values:", np.unique(arr)[:20])
print("Min/Max/Mean:", float(arr.min()), float(arr.max()), float(arr.mean()))
print("Nonzero voxels:", int((arr!=0).sum()), "/", arr.size)
PY
