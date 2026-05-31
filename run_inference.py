#!/usr/bin/env python3
"""
run_inference.py
Robustna skripta za sliding-window inferenco nad mapo z NIfTI volumni.
Uporaba:
  python run_inference.py --input_path /path/to/imagesVal --model_path /path/to/model.pth --output_path /path/to/out --patch_size 128 128 128 --overlap 0.5 --batch_size 2
"""
import argparse
from pathlib import Path
import numpy as np
import nibabel as nib
import torch
import importlib
from math import ceil
from scipy.ndimage import label

# -------------------------
# Sliding window inference
# -------------------------
def sliding_window_inference(volume, model, device, patch_size=(128,128,128), overlap=0.5, batch_size=4):
    """
    volume: numpy array shape (C,H,W,D) or (H,W,D)
    returns: numpy array (num_classes, H, W, D) of averaged logits/probs
    """
    if volume.ndim == 3:
        vol = volume[np.newaxis, ...]
    else:
        vol = volume
    C, H, W, D = vol.shape
    pz, py, px = patch_size
    stride = [max(1, int(p*(1-overlap))) for p in (pz,py,px)]

    def starts(dim, p, st):
        if dim <= p:
            return [0]
        n = int(ceil((dim - p) / st)) + 1
        return [min(i*st, dim-p) for i in range(n)]

    xs = starts(H, pz, stride[0])
    ys = starts(W, py, stride[1])
    zs = starts(D, px, stride[2])

    model.to(device)
    model.eval()

    count_map = np.zeros((H,W,D), dtype=np.float32)
    output_sum = None
    patches = []
    coords = []

    with torch.no_grad():
        for x in xs:
            for y in ys:
                for z in zs:
                    patch = vol[:, x:x+pz, y:y+py, z:z+px]
                    # pad patch if at border (shouldn't be needed due to starts logic, but safe)
                    if patch.shape[1:] != (pz,py,px):
                        pad = [(0,0)]
                        for s,ps in zip(patch.shape[1:], (pz,py,px)):
                            pad.append((0, ps - s))
                        patch = np.pad(patch, pad, mode='constant', constant_values=0)
                    patches.append(patch)
                    coords.append((x,y,z))
                    if len(patches) == batch_size:
                        batch = np.stack(patches, axis=0)  # (B,C,pz,py,px)
                        batch_t = torch.from_numpy(batch).float().to(device)
                        preds = model(batch_t)
                        preds = preds.cpu().numpy()
                        if output_sum is None:
                            num_classes = preds.shape[1]
                            output_sum = np.zeros((num_classes, H, W, D), dtype=np.float32)
                        for i, (cx,cy,cz) in enumerate(coords):
                            output_sum[:, cx:cx+pz, cy:cy+py, cz:cz+px] += preds[i]
                            count_map[cx:cx+pz, cy:cy+py, cz:cz+px] += 1.0
                        patches = []
                        coords = []
        # remaining
        if patches:
            batch = np.stack(patches, axis=0)
            batch_t = torch.from_numpy(batch).float().to(device)
            preds = model(batch_t).cpu().numpy()
            if output_sum is None:
                num_classes = preds.shape[1]
                output_sum = np.zeros((num_classes, H, W, D), dtype=np.float32)
            for i, (cx,cy,cz) in enumerate(coords):
                output_sum[:, cx:cx+pz, cy:cy+py, cz:cz+px] += preds[i]
                count_map[cx:cx+pz, cy:cy+py, cz:cz+px] += 1.0

    count_map[count_map==0] = 1.0
    output_avg = output_sum / count_map[np.newaxis, ...]
    return output_avg


# -------------------------
# Model loader (robust)
# -------------------------
def load_model(model_path, model_module, device):
    """
    Tries:
      1) import module and call get_model()
         then load state_dict if file is a state_dict
      2) if that fails, try torch.load(full_model)
    Returns model on device.
    """
    model_path = Path(model_path)
    # try import module and get_model()
    model = None
    try:
        mod = importlib.import_module(model_module)
        if hasattr(mod, "get_model"):
            model = mod.get_model()
        elif hasattr(mod, "build_stunet"):
            model = mod.build_stunet()
    except Exception:
        model = None

    # load checkpoint
    ckpt = torch.load(str(model_path), map_location=device)
    if model is not None:
        # ckpt may be state_dict or dict with 'model_state_dict'
        if isinstance(ckpt, dict):
            if "model_state_dict" in ckpt:
                state = ckpt["model_state_dict"]
            else:
                state = ckpt
            try:
                model.load_state_dict(state)
            except Exception:
                # try filtering keys if saved with module prefix
                new_state = {}
                for k,v in state.items():
                    nk = k.replace("module.", "") if k.startswith("module.") else k
                    new_state[nk] = v
                model.load_state_dict(new_state)
        else:
            # ckpt is not dict -> maybe full model
            model = ckpt
    else:
        # no model skeleton -> ckpt must be full model
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            raise RuntimeError("Model module import failed but checkpoint contains state_dict. Provide model module or save full model.")
        model = ckpt

    model.to(device)
    model.eval()
    return model


# -------------------------
# Postprocessing
# -------------------------
def postprocess_prediction(prob_map, method="argmax", threshold=0.5, keep_largest=True):
    """
    prob_map: (num_classes, H, W, D) or (1,H,W,D)
    returns binary or label volume (H,W,D) uint8
    """
    if method == "argmax":
        pred = np.argmax(prob_map, axis=0).astype(np.uint8)
    else:
        # assume binary class at index 1
        if prob_map.shape[0] == 1:
            prob = prob_map[0]
        else:
            prob = prob_map[1]
        pred = (prob >= threshold).astype(np.uint8)

    if keep_largest:
        labels, n = label(pred)
        if n > 1:
            counts = np.bincount(labels.flatten())
            counts[0] = 0
            largest = counts.argmax()
            pred = (labels == largest).astype(np.uint8)
    return pred


# -------------------------
# File processing
# -------------------------
def process_file(path, model, device, out_dir, patch_size=(128,128,128), overlap=0.5, batch_size=2, aggregation="argmax", threshold=0.5, no_cc=False):
    img = nib.load(str(path))
    vol = img.get_fdata().astype(np.float32)
    # simple normalization
    vol = (vol - vol.mean()) / (vol.std() + 1e-8)
    # ensure channel-first for sliding function: (C,Z,Y,X)
    if vol.ndim == 3:
        vol = vol[np.newaxis, ...]
    probs = sliding_window_inference(vol, model, device, patch_size=patch_size, overlap=overlap, batch_size=batch_size)
    pred = postprocess_prediction(probs, method=aggregation, threshold=threshold, keep_largest=not no_cc)
    out_file = Path(out_dir) / (Path(path).stem + "_pred.nii.gz")
    nib.save(nib.Nifti1Image(pred.astype(np.uint8), img.affine, img.header), str(out_file))
    print("Saved", out_file.name)


# -------------------------
# CLI
# -------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input_path", required=True, help="File or directory with NIfTI files")
    p.add_argument("--model_path", required=True, help="Path to model .pth")
    p.add_argument("--model_module", default="models.stunet", help="Python module with get_model() if checkpoint is state_dict")
    p.add_argument("--output_path", required=True, help="Directory to save predictions")
    p.add_argument("--patch_size", nargs=3, type=int, default=[128,128,128], help="Patch size Z Y X")
    p.add_argument("--overlap", type=float, default=0.5, help="Overlap fraction")
    p.add_argument("--batch_size", type=int, default=2, help="Number of patches per forward pass")
    p.add_argument("--aggregation", choices=["argmax","threshold"], default="argmax")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--no_cc", action="store_true", help="Disable keep-largest connected component")
    p.add_argument("--cpu", action="store_true", help="Force CPU")
    return p.parse_args()


def main():
    args = parse_args()
    inp = Path(args.input_path)
    out = Path(args.output_path); out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    model = load_model(args.model_path, args.model_module, device)

    # if input is directory, iterate over nifti files
    if inp.is_dir():
        files = sorted([f for f in inp.iterdir() if f.suffix in (".nii", ".gz") or f.name.endswith(".nii.gz")])
        if not files:
            raise FileNotFoundError(f"No NIfTI files found in {inp}")
        for f in files:
            try:
                process_file(f, model, device, out, patch_size=tuple(args.patch_size), overlap=args.overlap, batch_size=args.batch_size, aggregation=args.aggregation, threshold=args.threshold, no_cc=args.no_cc)
            except Exception as e:
                print(f"Error processing {f.name}: {e}")
    else:
        # single file
        process_file(inp, model, device, out, patch_size=tuple(args.patch_size), overlap=args.overlap, batch_size=args.batch_size, aggregation=args.aggregation, threshold=args.threshold, no_cc=args.no_cc)


if __name__ == "__main__":
    main()
