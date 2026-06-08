#!/usr/bin/env python3
"""
run_test.py
Evaluacija modela na celotnih 3D volumnih z uporabo sliding-window inferenc.
Shrani:
  - predikcije (.nii.gz)
  - CSV z Dice rezultati
"""

import argparse
from pathlib import Path
import numpy as np
import nibabel as nib
import torch
import csv

from inference.sliding_window import sliding_window_inference
from run_inference import load_model, postprocess_prediction


# -------------------------
# Dice metric
# -------------------------
def dice_score(pred, gt):
    """
    pred, gt: numpy arrays (H, W, D), binary {0,1}
    """
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    intersection = np.logical_and(pred, gt).sum()
    denom = pred.sum() + gt.sum()

    if denom == 0:
        return 1.0 if intersection == 0 else 0.0

    return 2.0 * intersection / denom


# -------------------------
# Process single case
# -------------------------
def evaluate_case(img_path, gt_path, model, device,
                  patch_size, overlap, aggregation, threshold, no_cc):

    # Load image
    img = nib.load(str(img_path))
    vol = img.get_fdata().astype(np.float32)

    # Normalization
    vol = (vol - vol.mean()) / (vol.std() + 1e-8)

    # Channel-first
    if vol.ndim == 3:
        vol = vol[np.newaxis, ...]

    # Sliding-window inference
    probs = sliding_window_inference(
        volume=vol,
        model=model,
        patch_size=patch_size,
        overlap=overlap,
        device=device
    )

    # Postprocess → binary mask
    pred = postprocess_prediction(
        probs,
        method=aggregation,
        threshold=threshold,
        keep_largest=not no_cc
    )

    # Load GT
    gt = nib.load(str(gt_path)).get_fdata().astype(np.uint8)

    # Compute Dice
    dice = dice_score(pred, gt)

    return pred, dice, img.affine, img.header


# -------------------------
# CLI
# -------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--images_dir", required=True, help="Directory with validation images")
    p.add_argument("--labels_dir", required=True, help="Directory with GT masks")
    p.add_argument("--model_path", required=True)
    p.add_argument("--model_module", default="models.stunet")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--patch_size", nargs=3, type=int, default=[128,128,128])
    p.add_argument("--overlap", type=float, default=0.5)
    p.add_argument("--aggregation", choices=["argmax","threshold"], default="argmax")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--no_cc", action="store_true")
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    images_dir = Path(args.images_dir)
    labels_dir = Path(args.labels_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")

    # Load model
    model = load_model(args.model_path, args.model_module, device)

    # Collect files
    images = sorted([f for f in images_dir.iterdir() if f.name.endswith(".nii.gz")])
    labels = sorted([f for f in labels_dir.iterdir() if f.name.endswith(".nii.gz")])

    if len(images) != len(labels):
        raise RuntimeError("Mismatch between number of images and labels.")

    # CSV output
    csv_path = out_dir / "results.csv"
    csv_file = open(csv_path, "w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow(["case", "dice"])

    # Evaluate each case
    for img_path, gt_path in zip(images, labels):
        print(f"Evaluating {img_path.name}...")

        pred, dice, affine, header = evaluate_case(
            img_path=img_path,
            gt_path=gt_path,
            model=model,
            device=device,
            patch_size=tuple(args.patch_size),
            overlap=args.overlap,
            aggregation=args.aggregation,
            threshold=args.threshold,
            no_cc=args.no_cc
        )

        # Save prediction
        out_pred = out_dir / f"{img_path.stem}_pred.nii.gz"
        nib.save(nib.Nifti1Image(pred.astype(np.uint8), affine, header), str(out_pred))

        # Write CSV
        writer.writerow([img_path.name, dice])
        print(f"Dice = {dice:.4f}")

    csv_file.close()
    print(f"\nResults saved to {csv_path}")


if __name__ == "__main__":
    main()
