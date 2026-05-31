# threshold_sweep_model.py
import numpy as np, nibabel as nib, os, torch
from pathlib import Path
from run_inference import load_model, sliding_window_inference

model_path = r"D:\izzivAMS\AMS-Izziv-Peter\outputs\stunet\model_final.pth"
img_dir = Path(r"D:\izzivAMS\AMS-Izziv-Peter\data\nnunet_raw\Dataset501_ImageCAS\imagesVal")
gt_dir  = Path(r"D:\izzivAMS\AMS-Izziv-Peter\data\nnunet_raw\Dataset501_ImageCAS\labelsVal")
device = torch.device("cpu")
model = load_model(model_path, "models.stunet", device)

def dice(a,b):
    a=a.astype(bool); b=b.astype(bool)
    inter=(a&b).sum()
    return 2*inter/(a.sum()+b.sum()+1e-8)

files = sorted([f for f in img_dir.iterdir() if f.suffix in ('.nii','.gz')])
thr_list = np.linspace(0.05,0.95,19)
results = {thr:[] for thr in thr_list}

for f in files:
    img = nib.load(str(f)).get_fdata().astype(np.float32)
    img = (img - img.mean())/(img.std()+1e-8)
    if img.ndim==3: img = img[np.newaxis,...]
    probs = sliding_window_inference(img, model, device, patch_size=(128,128,128), overlap=0.5, batch_size=1)
    # softmax
    exp = np.exp(probs - probs.max(axis=0, keepdims=True))
    soft = exp / exp.sum(axis=0, keepdims=True)
    prob_fg = soft[1] if soft.shape[0] > 1 else soft[0]
    # find GT
    base = f.stem
    # try to find matching GT file
    candidates = [g for g in gt_dir.iterdir() if g.name.startswith(base)]
    if not candidates:
        # try numeric prefix
        import re
        m = re.match(r"^(\d+)", base)
        if m:
            prefix = m.group(1)
            candidates = [g for g in gt_dir.iterdir() if g.name.startswith(prefix)]
    if not candidates:
        print("No GT for", f.name); continue
    gt = nib.load(str(candidates[0])).get_fdata()
    if prob_fg.shape != gt.shape:
        print("Shape mismatch", f.name, prob_fg.shape, candidates[0].name); continue
    for thr in thr_list:
        pred_bin = (prob_fg >= thr).astype(np.uint8)
        results[thr].append(dice(pred_bin, gt))

# summarize
best_thr, best_mean = None, -1
for thr in thr_list:
    vals = results[thr]
    if not vals: continue
    mean = float(np.mean(vals))
    print(f"thr {thr:.2f} -> mean {mean:.4f} n {len(vals)}")
    if mean > best_mean:
        best_mean = mean; best_thr = thr
print("\nBest thr:", best_thr, "mean", best_mean)
