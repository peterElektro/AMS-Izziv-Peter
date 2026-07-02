from pathlib import Path

BASE = Path("/workspace/nnUNet_raw/Dataset501_ImageCAS")

def rename_images(folder, ids):
    for cid in ids:
        # iščemo karkoli, kar se začne s številko (npr. 181.img.nii.gz)
        candidates = list((BASE / folder).glob(f"{cid}*"))
        if len(candidates) == 0:
            print(f"[WARNING] Missing image for {cid} in {folder}")
            continue

        src = candidates[0]
        dst = BASE / folder / f"{cid:03d}_0000.nii.gz"
        src.rename(dst)
        print(f"Renamed image: {src.name} -> {dst.name}")


def rename_labels(folder, ids):
    for cid in ids:
        candidates = list((BASE / folder).glob(f"{cid}*"))
        if len(candidates) == 0:
            print(f"[WARNING] Missing label for {cid} in {folder}")
            continue

        src = candidates[0]
        dst = BASE / folder / f"{cid:03d}.nii.gz"
        src.rename(dst)
        print(f"Renamed label: {src.name} -> {dst.name}")


# === Train: 1–160 ===
rename_images("imagesTr", range(1, 161))
rename_labels("labelsTr", range(1, 161))

# === Val: 161–180 ===
rename_images("imagesVal", range(161, 181))
rename_labels("labelsVal", range(161, 181))

# === Test: 181–200 (brez labelov) ===
rename_images("imagesTs", range(181, 201))

print("\nAll renaming done.")
