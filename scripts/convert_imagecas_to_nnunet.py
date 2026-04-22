#!/usr/bin/env python3
import os
import argparse
import json
import shutil
from pathlib import Path
import nibabel as nib

def log(msg):
    print(f"[convert] {msg}")

def validate_nifti(path):
    try:
        img = nib.load(str(path))
        _ = img.get_fdata()
        return True
    except Exception as e:
        log(f"ERROR: Invalid NIfTI file {path}: {e}")
        return False

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def load_split(split_file):
    with open(split_file, "r") as f:
        split = json.load(f)
    return split["train"], split["val"], split.get("test", [])

def convert_case(case_id, raw_dir, out_img_dir, out_lbl_dir, is_train=True):
    img_path = raw_dir / f"{case_id}.img.nii.gz"
    lbl_path = raw_dir / f"{case_id}.label.nii.gz"

    if not img_path.exists():
        log(f"WARNING: Missing image for case {case_id}")
        return False
    if is_train and not lbl_path.exists():
        log(f"WARNING: Missing label for case {case_id}")
        return False

    # Validate NIfTI
    if not validate_nifti(img_path):
        return False
    if is_train and not validate_nifti(lbl_path):
        return False

    # nnU-Net naming convention
    # imagesTr: case_0001_0000.nii.gz
    # labelsTr: case_0001.nii.gz
    out_img_name = f"case_{case_id:04d}_0000.nii.gz"
    out_lbl_name = f"case_{case_id:04d}.nii.gz"

    shutil.copy(img_path, out_img_dir / out_img_name)
    if is_train:
        shutil.copy(lbl_path, out_lbl_dir / out_lbl_name)

    return True

def main():
    parser = argparse.ArgumentParser(description="Convert ImageCAS → nnU-Net format")
    parser.add_argument("--raw_dir", type=str, required=True,
                        help="Path to data/raw/ directory containing *.img.nii.gz and *.label.nii.gz")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Path to data/nnunet_raw/ where DatasetXXX_ImageCAS will be created")
    parser.add_argument("--dataset_id", type=int, default=501,
                        help="nnU-Net dataset ID (e.g., 501)")
    parser.add_argument("--split_file", type=str, required=True,
                        help="JSON file containing Split-1 train/val/test lists")
    parser.add_argument("--include_test", action="store_true",
                        help="Also convert test set into imagesTs/")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_root = Path(args.output_dir)
    dataset_name = f"Dataset{args.dataset_id}_ImageCAS"

    out_dataset_dir = out_root / dataset_name
    out_imagesTr = out_dataset_dir / "imagesTr"
    out_labelsTr = out_dataset_dir / "labelsTr"
    out_imagesTs = out_dataset_dir / "imagesTs"

    log(f"Creating nnU-Net dataset at: {out_dataset_dir}")

    ensure_dir(out_imagesTr)
    ensure_dir(out_labelsTr)
    if args.include_test:
        ensure_dir(out_imagesTs)

    # Load split
    train_ids, val_ids, test_ids = load_split(args.split_file)

    log(f"Train cases: {len(train_ids)}")
    log(f"Val cases:   {len(val_ids)}")
    log(f"Test cases:  {len(test_ids)}")

    # Convert train + val (both go to imagesTr/labelsTr)
    for cid in train_ids + val_ids:
        ok = convert_case(cid, raw_dir, out_imagesTr, out_labelsTr, is_train=True)
        if not ok:
            log(f"Failed to convert case {cid}")

    # Convert test (images only)
    if args.include_test:
        for cid in test_ids:
            img_path = raw_dir / f"{cid}.img.nii.gz"
            if not img_path.exists():
                log(f"WARNING: Missing test image for case {cid}")
                continue
            if not validate_nifti(img_path):
                continue
            out_img_name = f"case_{cid:04d}_0000.nii.gz"
            shutil.copy(img_path, out_imagesTs / out_img_name)

    # Create dataset.json (required by nnU-Net)
    dataset_json = {
        "name": "ImageCAS",
        "description": "Coronary artery segmentation dataset",
        "tensorImageSize": "3D",
        "reference": "ImageCAS Challenge",
        "licence": "CC-BY-NC",
        "release": "1.0",
        "modality": {
            "0": "CT"
        },
        "labels": {
            "0": "background",
            "1": "artery"
        },
        "numTraining": len(train_ids) + len(val_ids),
        "numTest": len(test_ids),
        "training": [
            {"image": f"./imagesTr/case_{cid:04d}_0000.nii.gz",
             "label": f"./labelsTr/case_{cid:04d}.nii.gz"}
            for cid in train_ids + val_ids
        ],
        "test": [
            f"./imagesTs/case_{cid:04d}_0000.nii.gz"
            for cid in test_ids
        ]
    }

    with open(out_dataset_dir / "dataset.json", "w") as f:
        json.dump(dataset_json, f, indent=4)

    log("Conversion complete.")

if __name__ == "__main__":
    main()
