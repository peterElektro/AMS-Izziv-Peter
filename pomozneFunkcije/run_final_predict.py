import torch
import nibabel as nib
import numpy as np
from pathlib import Path

from models.stunet import STUNetLitePlus
from inference.sliding_window import sliding_window_inference


# ---------------------------------------------------------
# 1) Robustno nalaganje modela (tvoj checkpoint format)
# ---------------------------------------------------------
def load_model(model_path, device):
    model = STUNetLitePlus(in_channels=1, out_channels=1)
    ckpt = torch.load(model_path, map_location=device)

    # 1) Če checkpoint vsebuje "model_state_dict"
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]

    # 2) Če checkpoint vsebuje "model_state"
    elif isinstance(ckpt, dict) and "model_state" in ckpt:
        state_dict = ckpt["model_state"]

    # 3) Če je čisti state_dict
    else:
        state_dict = ckpt

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


# ---------------------------------------------------------
# 2) Priprava volumna
# ---------------------------------------------------------
def load_and_normalize(img_path):
    img = nib.load(str(img_path)).get_fdata().astype(np.float32)

    # normalizacija
    img_norm = (img - img.mean()) / (img.std() + 1e-8)

    # pretvorba v (C, D, H, W)
    img_norm = np.expand_dims(img_norm, axis=0)

    return img_norm


# ---------------------------------------------------------
# 3) Glavna predikcija
# ---------------------------------------------------------
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

    # -------------------------------------------------
    # 🔥 DEBUG: KLJUČNI DEL (DODANO)
    # -------------------------------------------------
    print("\n===== PREDICTION STATS =====")
    print("min:", float(pred.min()))
    print("max:", float(pred.max()))
    print("mean:", float(pred.mean()))

    print("\nthreshold analysis:")
    for t in [0.1, 0.2, 0.3, 0.4, 0.5]:
        print(f"> {t:.1f} : {(pred > t).sum()} voxels")

    print("============================\n")

    # -------------------------------------------------
    # save NIfTI
    # -------------------------------------------------
    ref = nib.load(str(img_path))
    pred_img = nib.Nifti1Image(pred.astype(np.float32), ref.affine, ref.header)
    nib.save(pred_img, str(out_path))

# ---------------------------------------------------------
# 4) MAIN
# ---------------------------------------------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_path = "outputs/stunet_long_training/model_best.pth"
    img_path   = "data_preproc/Dataset501_ImageCAS/imagesTs/182.img.nii.gz"
    out_path   = "data_preproc/prediction_182.nii.gz"

    print(f"Loading model: {model_path}")
    model = load_model(model_path, device)

    print("Running inference...")
    run_prediction(model, img_path, out_path, device)

    print(f"Saved prediction to: {out_path}")
