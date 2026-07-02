import torch
import nibabel as nib
import numpy as np
from pathlib import Path

from models.stunet import STUNetLitePlus
from inference.sliding_window import sliding_window_inference


def load_model(model_path, device):
    model = STUNetLitePlus(in_channels=1, out_channels=1)
    ckpt = torch.load(model_path, map_location=device)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict) and "model_state" in ckpt:
        state_dict = ckpt["model_state"]
    else:
        state_dict = ckpt

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def load_and_normalize(img_path):
    img = nib.load(str(img_path)).get_fdata().astype(np.float32)
    img_norm = (img - img.mean()) / (img.std() + 1e-8)
    img_norm = np.expand_dims(img_norm, axis=0)  # (1, D, H, W)
    return img_norm


def run_prediction(model, img_path, out_path, device):
    volume_np = load_and_normalize(img_path)

    # raw logits from STU-Net (shape: (1, D, H, W))
    pred = sliding_window_inference(
        volume_np,
        model,
        patch_size=(128,128,128),
        overlap=0.5,
        device=device
    )

    # ----- POPRAVEK: STU-Net je 1-kanalni → sigmoid + threshold -----
    #pred = torch.tensor(pred)          # pretvori v tensor
    pred = torch.from_numpy(pred)

    #pred = torch.sigmoid(pred)         # sigmoid aktivacija
    pred = pred[0].numpy()             # foreground kanal
    pred_bin = (pred >= 0.45).astype(np.uint8)   # binarna maska
    # ---------------------------------------------------------------

    print("\n===== PREDICTION STATS =====")
    print("min:", float(pred.min()))
    print("max:", float(pred.max()))
    print("mean:", float(pred.mean()))
    print("foreground voxels:", int(pred_bin.sum()))
    print("============================\n")

    ref = nib.load(str(img_path))
    pred_img = nib.Nifti1Image(pred_bin.astype(np.uint8), ref.affine, ref.header)
    nib.save(pred_img, str(out_path))


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_path = "outputs/stunet_long/model_final.pth"
    input_dir = Path("nnUNet_raw/Dataset501_ImageCAS/imagesTs")
    output_dir = Path("outputs/stunet/predictions_batch")
    output_dir.mkdir(exist_ok=True)

    print(f"Loading model: {model_path}")
    model = load_model(model_path, device)

    for case_id in range(181, 201):
        matches = list(input_dir.glob(f"{case_id}_*.nii.gz"))
        if len(matches) == 0:
            print(f"[ERROR] No input file for case {case_id}")
            continue

        img_path = matches[0]
        out_path = output_dir / f"prediction_{case_id}.nii.gz"

        print(f"\n=== Running prediction for case {case_id} ===")
        run_prediction(model, img_path, out_path, device)
        print(f"Saved: {out_path}")
