import nibabel as nib
import numpy as np
from pathlib import Path
from scipy.spatial.distance import cdist
import csv


def dice_score(pred, gt):
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    return 2 * inter / (pred.sum() + gt.sum() + 1e-8)


def hd95(pred, gt):
    pred_pts = np.argwhere(pred > 0)
    gt_pts = np.argwhere(gt > 0)

    if len(pred_pts) == 0 or len(gt_pts) == 0:
        return np.inf

    d1 = cdist(pred_pts, gt_pts).min(axis=1)
    d2 = cdist(gt_pts, pred_pts).min(axis=1)

    return np.percentile(np.hstack([d1, d2]), 95)


def evaluate_case(pred_path, gt_path):
    pred = nib.load(str(pred_path)).get_fdata().astype(np.uint8)
    gt = nib.load(str(gt_path)).get_fdata().astype(np.uint8)

    dsc = dice_score(pred, gt)
    hd = hd95(pred, gt)

    return dsc, hd


if __name__ == "__main__":
    pred_dir = Path("/workspace/outputs/stunet/predictions")
    gt_dir = Path("/workspace/nnUNet_raw/Dataset501_ImageCAS/labelsTs")

    results = []
    csv_path = "eval_results_stunet.csv"

    for case_id in range(181, 201):

        # STU-Net predikcije: 181_0000.nii_pred.nii.gz
        pred_candidates = list(pred_dir.glob(f"{case_id}_*.nii.gz"))
        if len(pred_candidates) == 0:
            print(f"[WARNING] Missing prediction for case {case_id}")
            continue
        pred_path = pred_candidates[0]

        # GT: 181.nii.gz
        gt_candidates = list(gt_dir.glob(f"{case_id}.nii.gz"))
        if len(gt_candidates) == 0:
            print(f"[WARNING] Missing GT for case {case_id}")
            continue
        gt_path = gt_candidates[0]

        print(f"\n=== Evaluating case {case_id} ===")
        dsc, hd = evaluate_case(pred_path, gt_path)
        print(f"Dice: {dsc:.4f}")
        print(f"HD95: {hd:.2f}")

        results.append([case_id, dsc, hd])

    # Save CSV
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Case", "Dice", "HD95"])
        writer.writerows(results)

    # Summary
    dices = [r[1] for r in results]
    hd95s = [r[2] for r in results]

    print("\n=== SUMMARY ===")
    print(f"Mean Dice: {np.mean(dices):.4f}")
    print(f"Mean HD95: {np.mean(hd95s):.2f}")
    print(f"Results saved to {csv_path}")
