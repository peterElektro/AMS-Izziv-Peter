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
    img_norm = np.expand_dims(img_norm, axis=0)
    return img_norm


def run_prediction(model, img_path, out_path, device):
    volume_np = load_and_normalize(img_path)

    pred = sliding_window_inference(
        volume_np,
        model,
        patch_size=(128,128,128),
        overlap=0.5,
        device=device
    )

    pred = pred[0]  # (1, D, H, W) → (D, H, W)

    print("\n===== PREDICTION STATS =====")
    print("min:", float(pred.min()))
    print("max:", float(pred.max()))
    print("mean:", float(pred.mean()))
    print("\nthreshold analysis:")
    for t in [0.1, 0.2, 0.3, 0.4, 0.5]:
        print(f"> {t:.1f} : {(pred > t).sum()} voxels")
    print("============================\n")

    ref = nib.load(str(img_path))
    pred_img = nib.Nifti1Image(pred.astype(np.float32), ref.affine, ref.header)
    nib.save(pred_img, str(out_path))


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_path = "outputs/stunet_long_training/model_best.pth"
    input_dir = Path("data_preproc/Dataset501_ImageCAS/imagesTs")
    output_dir = Path("predictions_batch")
    output_dir.mkdir(exist_ok=True)

    print(f"Loading model: {model_path}")
    model = load_model(model_path, device)

    for case_id in range(181, 201):
        img_path = input_dir / f"{case_id}.img.nii.gz"
        out_path = output_dir / f"prediction_{case_id}.nii.gz"

        print(f"\n=== Running prediction for case {case_id} ===")
        run_prediction(model, img_path, out_path, device)
        print(f"Saved: {out_path}")
