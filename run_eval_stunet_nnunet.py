import nibabel as nib
import numpy as np
from pathlib import Path
from scipy.spatial.distance import cdist
import csv
import re


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


# -------------------------
# Robustno iskanje predikcij
# -------------------------
def find_pred_file(pred_dir, case_id):
    pred_dir = Path(pred_dir)
    cid = str(case_id)

    candidates = []

    for f in pred_dir.iterdir():
        name = f.name.lower()

        if not name.endswith(".nii.gz"):
            continue

        # vsebuje case ID (181, 0181, case_0181, ...)
        if cid not in name and f"0{cid}" not in name and f"{cid.zfill(4)}" not in name:
            continue

        # STU-Net stil
        if "pred" in name:
            candidates.append(f)
            continue

        # nnU-Net stil
        if name.startswith(cid):
            candidates.append(f)
            continue

        # nnU-Net case_0181.nii.gz
        if name.startswith(f"case_{cid.zfill(4)}"):
            candidates.append(f)
            continue

    if not candidates:
        return None

    return sorted(candidates, key=lambda x: len(x.name), reverse=True)[0]


# -------------------------
# POPRAVLJENO robustno iskanje GT
# -------------------------
def find_gt_file(gt_dir, case_id):
    gt_dir = Path(gt_dir)
    cid = str(case_id)

    candidates = []

    for f in gt_dir.iterdir():
        name = f.name.lower()

        if not name.endswith(".nii.gz"):
            continue

        # najpogostejši formati:
        # 181.nii.gz
        # 181.label.nii.gz
        # case_0181.nii.gz
        # case_0181_label.nii.gz
        # case_0181_0000.nii.gz

        if name.startswith(cid):
            candidates.append(f)
            continue

        if f"{cid}.label" in name:
            candidates.append(f)
            continue

        if name.startswith(f"case_{cid.zfill(4)}"):
            candidates.append(f)
            continue

    if not candidates:
        return None

    return sorted(candidates, key=lambda x: len(x.name), reverse=True)[0]


def evaluate_case(pred_path, gt_path):
    pred = nib.load(str(pred_path)).get_fdata().astype(np.uint8)
    gt   = nib.load(str(gt_path)).get_fdata().astype(np.uint8)
    return dice_score(pred, gt), hd95(pred, gt)


def eval_model(pred_dir, gt_dir, csv_out):
    results = []

    for case_id in range(181, 201):
        pred_path = find_pred_file(pred_dir, case_id)
        gt_path   = find_gt_file(gt_dir, case_id)

        if pred_path is None:
            print(f"[WARNING] Missing prediction for case {case_id}")
            continue
        if gt_path is None:
            print(f"[WARNING] Missing GT for case {case_id}")
            continue

        print(f"\n=== Evaluating case {case_id} ===")
        print(f"Pred: {pred_path.name}")
        print(f"GT:   {gt_path.name}")

        dsc, hd = evaluate_case(pred_path, gt_path)
        print(f"Dice: {dsc:.4f}")
        print(f"HD95: {hd:.2f}")

        results.append([case_id, dsc, hd])

    with open(csv_out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Case", "Dice", "HD95"])
        w.writerows(results)

    dices = [r[1] for r in results]
    hd95s = [r[2] for r in results]

    print("\n=== SUMMARY ===")
    print(f"Mean Dice: {np.mean(dices):.4f}")
    print(f"Mean HD95: {np.mean(hd95s):.2f}")
    print(f"Results saved to {csv_out}")


if __name__ == "__main__":
    eval_model(
        pred_dir="outputs/stunet/predictions",
        gt_dir="data_preproc/Dataset501_ImageCAS/labelsTs",
        csv_out="eval_results_stunet_test.csv"
    )

    eval_model(
        pred_dir="nnunet_preds",
        gt_dir="data_preproc/Dataset501_ImageCAS/labelsTs",
        csv_out="eval_results_nnunet_test.csv"
    )
