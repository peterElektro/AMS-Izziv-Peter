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
    """Compute 95th percentile Hausdorff distance."""
    pred_pts = np.argwhere(pred > 0)
    gt_pts = np.argwhere(gt > 0)

    if len(pred_pts) == 0 or len(gt_pts) == 0:
        return np.inf

    d1 = cdist(pred_pts, gt_pts).min(axis=1)
    d2 = cdist(gt_pts, pred_pts).min(axis=1)

    return np.percentile(np.hstack([d1, d2]), 95)


def evaluate_case(clean_path, gt_path):
    pred = nib.load(str(clean_path)).get_fdata().astype(np.uint8)
    gt = nib.load(str(gt_path)).get_fdata().astype(np.uint8)

    dsc = dice_score(pred, gt)
    hd = hd95(pred, gt)

    return dsc, hd


if __name__ == "__main__":
    clean_dir = Path("nnunet_preds")
    gt_dir = Path("data_preproc/Dataset501_ImageCAS/labelsTs")
    


    results = []
    csv_path = "eval_results_nnunet.csv"

    for case_id in range(181, 201):
        clean_path = clean_dir / f"{case_id}.nii.gz"

        # GT lahko ima različne konvencije imen → vzamemo prvi match
        candidates = list(gt_dir.glob(f"{case_id}*.nii.gz"))
        if len(candidates) == 0:
            print(f"[WARNING] Missing GT for case {case_id}")
            continue

        gt_path = candidates[0]

        print(f"\n=== Evaluating case {case_id} ===")
        dsc, hd = evaluate_case(clean_path, gt_path)
        print(f"Dice: {dsc:.4f}")
        print(f"HD95: {hd:.2f}")

        results.append([case_id, dsc, hd])

    # Shrani CSV
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Case", "Dice", "HD95"])
        writer.writerows(results)

    # Povprečja
    dices = [r[1] for r in results]
    hd95s = [r[2] for r in results]

    print("\n=== SUMMARY ===")
    print(f"Mean Dice: {np.mean(dices):.4f}")
    print(f"Mean HD95: {np.mean(hd95s):.2f}")
    print(f"Results saved to {csv_path}")
