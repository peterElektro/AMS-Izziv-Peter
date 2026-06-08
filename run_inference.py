#!/usr/bin/env python3
"""
run_inference.py
Robustna skripta za sliding-window inferenco nad mapo z NIfTI volumni.
"""

import argparse
from pathlib import Path
import numpy as np
import nibabel as nib
import torch
import importlib
from scipy.ndimage import label

# 🔥 NOVO: uvozimo modularni sliding-window
from inference.sliding_window import sliding_window_inference


# -------------------------
# Model loader (robust)
# -------------------------
def load_model(model_path, model_module, device):
    model_path = Path(model_path)
    model = None

    try:
        mod = importlib.import_module(model_module)
        if hasattr(mod, "get_model"):
            model = mod.get_model()
        elif hasattr(mod, "build_stunet"):
            model = mod.build_stunet()
    except Exception:
        model = None

    ckpt = torch.load(str(model_path), map_location=device)

    if model is not None:
        if isinstance(ckpt, dict):
            if "model_state_dict" in ckpt:
                state = ckpt["model_state_dict"]
            elif "model_state" in ckpt:
                state = ckpt["model_state"]
            else:
                state = ckpt


            try:
                model.load_state_dict(state)
            except Exception:
                new_state = {}
                for k, v in state.items():
                    nk = k.replace("module.", "") if k.startswith("module.") else k
                    new_state[nk] = v
                model.load_state_dict(new_state)
        else:
            model = ckpt
    else:
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            raise RuntimeError("Model module import failed but checkpoint contains state_dict.")
        model = ckpt

    model.to(device)
    model.eval()
    return model


# -------------------------
# Postprocessing
# -------------------------
def postprocess_prediction(prob_map, method="argmax", threshold=0.5, keep_largest=True):
    if method == "argmax":
        pred = np.argmax(prob_map, axis=0).astype(np.uint8)
    else:
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
def process_file(path, model, device, out_dir,
                 patch_size=(128,128,128), overlap=0.5, batch_size=2,
                 aggregation="argmax", threshold=0.5, no_cc=False):

    img = nib.load(str(path))
    vol = img.get_fdata().astype(np.float32)

    # Normalizacija
    vol = (vol - vol.mean()) / (vol.std() + 1e-8)

    # Channel-first
    if vol.ndim == 3:
        vol = vol[np.newaxis, ...]

    # 🔥 KLIC modularnega sliding-window
    probs = sliding_window_inference(
        volume=vol,
        model=model,
        patch_size=patch_size,
        overlap=overlap,
        device=device
    )

    pred = postprocess_prediction(
        probs,
        method=aggregation,
        threshold=threshold,
        keep_largest=not no_cc
    )

    out_file = Path(out_dir) / (Path(path).stem + "_pred.nii.gz")
    nib.save(nib.Nifti1Image(pred.astype(np.uint8), img.affine, img.header), str(out_file))
    print("Saved", out_file.name)


# -------------------------
# CLI
# -------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input_path", required=True)
    p.add_argument("--model_path", required=True)
    p.add_argument("--model_module", default="models.stunet")
    p.add_argument("--output_path", required=True)
    p.add_argument("--patch_size", nargs=3, type=int, default=[128,128,128])
    p.add_argument("--overlap", type=float, default=0.5)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--aggregation", choices=["argmax","threshold"], default="argmax")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--no_cc", action="store_true")
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    inp = Path(args.input_path)
    out = Path(args.output_path); out.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    model = load_model(args.model_path, args.model_module, device)

    if inp.is_dir():
        files = sorted([f for f in inp.iterdir() if f.suffix in (".nii", ".gz") or f.name.endswith(".nii.gz")])
        if not files:
            raise FileNotFoundError(f"No NIfTI files found in {inp}")

        for f in files:
            try:
                process_file(
                    f, model, device, out,
                    patch_size=tuple(args.patch_size),
                    overlap=args.overlap,
                    batch_size=args.batch_size,
                    aggregation=args.aggregation,
                    threshold=args.threshold,
                    no_cc=args.no_cc
                )
            except Exception as e:
                print(f"Error processing {f.name}: {e}")

    else:
        process_file(
            inp, model, device, out,
            patch_size=tuple(args.patch_size),
            overlap=args.overlap,
            batch_size=args.batch_size,
            aggregation=args.aggregation,
            threshold=args.threshold,
            no_cc=args.no_cc
        )


if __name__ == "__main__":
    main()
