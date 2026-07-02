import nibabel as nib
import numpy as np
from pathlib import Path
from scipy.ndimage import distance_transform_edt, binary_erosion
import csv


# -------------------------
# Dice
# -------------------------
def dice_score(pred, gt):
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    return 2 * inter / (pred.sum() + gt.sum() + 1e-8)


# -------------------------
# HITRI HD95 (surface-based, O(N))
# -------------------------
def hd95(pred, gt):
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    if pred.sum() == 0 or gt.sum() == 0:
        return np.inf

    pred_surface = pred ^ binary_erosion(pred)
    gt_surface   = gt ^ binary_erosion(gt)

    dt_pred = distance_transform_edt(~pred_surface)
    dt_gt   = distance_transform_edt(~gt_surface)

    d_pred_to_gt = dt_gt[pred_surface]
    d_gt_to_pred = dt_pred[gt_surface]

    all_d = np.hstack([d_pred_to_gt, d_gt_to_pred])
    return np.percentile(all_d, 95)


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

        if cid in name or f"0{cid}" in name or f"{cid.zfill(4)}" in name:
            candidates.append(f)

    if not candidates:
        return None

    return sorted(candidates, key=lambda x: len(x.name), reverse=True)[0]


# -------------------------
# Robustno iskanje GT
# -------------------------
def find_gt_file(gt_dir, case_id):
    gt_dir = Path(gt_dir)
    cid = str(case_id)

    candidates = []

    for f in gt_dir.iterdir():
        name = f.name.lower()

        if not name.endswith(".nii.gz"):
            continue

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


# -------------------------
# Eval enega primera
# -------------------------
def evaluate_case(pred_path, gt_path):
    pred = nib.load(str(pred_path)).get_fdata().astype(np.uint8)
    gt   = nib.load(str(gt_path)).get_fdata().astype(np.uint8)

    print("  pred shape:", pred.shape)
    print("  gt   shape:", gt.shape)

    return dice_score(pred, gt), hd95(pred, gt)


# -------------------------
# Eval modela
# -------------------------
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


# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    # STU-Net eval
    eval_model(
        pred_dir="outputs/stunet/predictions_batch",
        gt_dir="nnUNet_raw/Dataset501_ImageCAS/labelsTs",
        csv_out="eval_results_stunet_fast.csv"
    )

    # nnU-Net eval (če želiš)
    eval_model(
        pred_dir="nnunet_preds",
        gt_dir="nnUNet_raw/Dataset501_ImageCAS/labelsTs",
        csv_out="eval_results_nnunet_fast.csv"
    )
