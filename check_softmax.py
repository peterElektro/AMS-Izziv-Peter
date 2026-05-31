# check_softmax.py
import numpy as np, nibabel as nib, torch
from pathlib import Path
from run_inference import load_model, sliding_window_inference

img_path = Path(r"D:\izzivAMS\AMS-Izziv-Peter\data\nnunet_raw\Dataset501_ImageCAS\imagesVal\161.img.nii.gz")
model_path = r"D:\izzivAMS\AMS-Izziv-Peter\outputs\stunet\model_final.pth"
device = torch.device("cpu")
model = load_model(model_path, "models.stunet", device)

img = nib.load(str(img_path)).get_fdata().astype(np.float32)
img = (img - img.mean())/(img.std()+1e-8)
if img.ndim==3: img = img[np.newaxis,...]

probs = sliding_window_inference(img, model, device, patch_size=(128,128,128), overlap=0.5, batch_size=1)
exp = np.exp(probs - probs.max(axis=0, keepdims=True))
soft = exp / exp.sum(axis=0, keepdims=True)
fg = soft[1]
print("softmax fg min/max/mean:", float(fg.min()), float(fg.max()), float(fg.mean()))
print("voxels with fg>0.5:", int((fg>0.5).sum()), " / ", fg.size)
print("voxels with fg>0.1:", int((fg>0.1).sum()))

