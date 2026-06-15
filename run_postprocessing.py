import nibabel as nib
import numpy as np
from pathlib import Path
from scipy.ndimage import label, binary_closing, generate_binary_structure


def keep_largest_component(mask):
    """Obdrži največjo povezano komponento."""
    struct = generate_binary_structure(3, 2)
    labeled, num = label(mask, structure=struct)

    if num == 0:
        return mask

    sizes = [(labeled == i).sum() for i in range(1, num + 1)]
    largest = np.argmax(sizes) + 1

    return (labeled == largest).astype(np.uint8)


def postprocess_mask(mask_path, out_path, closing=True):
    """Naloži masko, očisti komponente, opcijsko zapre luknje."""
    mask = nib.load(str(mask_path)).get_fdata().astype(np.uint8)

    print(f"\n=== {mask_path.name} ===")
    print("Initial voxels:", int(mask.sum()))

    # 1) Obdrži največjo komponento
    mask_clean = keep_largest_component(mask)
    print("After largest component:", int(mask_clean.sum()))

    # 2) Opcijsko morphological closing
    if closing:
        struct = generate_binary_structure(3, 1)
        mask_clean = binary_closing(mask_clean, structure=struct).astype(np.uint8)
        print("After closing:", int(mask_clean.sum()))

    # Shrani rezultat
    ref = nib.load(str(mask_path))
    out_img = nib.Nifti1Image(mask_clean, ref.affine, ref.header)
    nib.save(out_img, str(out_path))


if __name__ == "__main__":
    mask_dir = Path("masks_batch")
    out_dir = Path("masks_clean")
    out_dir.mkdir(exist_ok=True)

    for case_id in range(181, 201):
        mask_path = mask_dir / f"mask_{case_id}.nii.gz"
        out_path = out_dir / f"clean_{case_id}.nii.gz"

        if not mask_path.exists():
            print(f"[WARNING] Missing mask for case {case_id}")
            continue

        postprocess_mask(mask_path, out_path, closing=True)

    print("\nAll cleaned masks saved to:", out_dir)
