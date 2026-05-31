# debug_model_out.py
import numpy as np
import nibabel as nib
import torch
from pathlib import Path
from run_inference import load_model, sliding_window_inference

img_path = Path(r"D:\izzivAMS\AMS-Izziv-Peter\data\nnunet_raw\Dataset501_ImageCAS\imagesVal\161.img.nii.gz")
model_path = r"D:\izzivAMS\AMS-Izziv-Peter\outputs\stunet\model_final.pth"

device = torch.device("cpu")
model = load_model(model_path, "models.stunet", device)

img = nib.load(str(img_path)).get_fdata().astype(np.float32)
img = (img - img.mean()) / (img.std() + 1e-8)
if img.ndim == 3:
    img = img[np.newaxis, ...]   # channel dim

probs = sliding_window_inference(img, model, device, patch_size=(128,128,128), overlap=0.5, batch_size=1)
print("probs shape:", probs.shape)
print("probs min/max/mean:", float(probs.min()), float(probs.max()), float(probs.mean()))
if probs.shape[0] >= 2:
    fg = probs[1]; bg = probs[0]
    print("fg min/max/mean:", float(fg.min()), float(fg.max()), float(fg.mean()))
    print("bg min/max/mean:", float(bg.min()), float(bg.max()), float(bg.mean()))
