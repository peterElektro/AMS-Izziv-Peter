import nibabel as nib
import numpy as np
from pathlib import Path


def threshold_volume(pred_path, out_path, thr=0.5):
    """Naloži predikcijo, naredi binarno masko in shrani."""
    pred = nib.load(str(pred_path)).get_fdata().astype(np.float32)

    mask = (pred > thr).astype(np.uint8)

    print(f"\n=== {pred_path.name} ===")
    print("Foreground voxels:", int(mask.sum()))
    print("Threshold:", thr)

    ref = nib.load(str(pred_path))
    mask_img = nib.Nifti1Image(mask, ref.affine, ref.header)
    nib.save(mask_img, str(out_path))


if __name__ == "__main__":
    pred_dir = Path("predictions_batch")
    out_dir = Path("masks_batch")
    out_dir.mkdir(exist_ok=True)

    threshold = 0.5

    for case_id in range(181, 201):
        pred_path = pred_dir / f"prediction_{case_id}.nii.gz"
        out_path = out_dir / f"mask_{case_id}.nii.gz"

        if not pred_path.exists():
            print(f"[WARNING] Missing prediction for case {case_id}")
            continue

        threshold_volume(pred_path, out_path, thr=threshold)

    print("\nAll masks saved to:", out_dir)