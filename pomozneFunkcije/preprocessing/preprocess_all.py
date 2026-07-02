# preprocess_all.py
import nibabel as nib, numpy as np, pathlib
from scipy.ndimage import zoom

SRC_IMG = pathlib.Path(r"D:\izzivAMS\AMS-Izziv-Peter\data\nnunet_raw\Dataset501_ImageCAS\imagesTr")
DST_IMG = pathlib.Path(r"D:\izzivAMS\AMS-Izziv-Peter\data_preproc\Dataset501_ImageCAS\imagesTr")
DST_IMG.mkdir(parents=True, exist_ok=True)

# Optional: target spacing (set to None to skip resampling)
TARGET_SPACING = None  # e.g., (1.0,1.0,1.0) or None

def zscore(a):
    a = a.astype(np.float32)
    m = a.mean()
    s = a.std()
    return (a - m) / (s + 1e-8)

for f in sorted(SRC_IMG.glob("*.nii*")):
    img = nib.load(str(f))
    data = img.get_fdata()
    if TARGET_SPACING is not None:
        # naive resample using zoom factor (only if header has pixdim)
        hdr = img.header
        orig_sp = tuple(hdr.get_zooms()[:3])
        zoom_f = tuple(o/t for o,t in zip(orig_sp, TARGET_SPACING))
        data = zoom(data, zoom_f, order=1)
    data = zscore(data)
    out = DST_IMG / f.name
    nib.save(nib.Nifti1Image(data.astype(np.float32), img.affine, img.header), str(out))
    print("Saved", out.name)
