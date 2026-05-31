# remap_labels.py
import nibabel as nib, numpy as np, pathlib
src = pathlib.Path(r"D:\izzivAMS\AMS-Izziv-Peter\data\nnunet_raw\Dataset501_ImageCAS\labelsTr")
dst = pathlib.Path(r"D:\izzivAMS\AMS-Izziv-Peter\data_preproc\Dataset501_ImageCAS\labelsTr")
dst.mkdir(parents=True, exist_ok=True)
for f in sorted(src.glob("*.nii*")):
    img = nib.load(str(f))
    a = img.get_fdata()
    a = (a > 0).astype(np.uint8)   # map any nonzero to 1
    out = dst / f.name
    nib.save(nib.Nifti1Image(a, img.affine, img.header), str(out))
    print("Saved", out.name)
