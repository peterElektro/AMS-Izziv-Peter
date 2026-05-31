#!/usr/bin/env python3
import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler
import torch.amp
import nibabel as nib
import numpy as np

from models.stunet import build_stunet
from dataloaders.nnunet_loader import create_dataloader
from metrics import dice_score

# ---------------------------
# Utility functions
# ---------------------------
def compute_pos_weight(labels_dir):
    lbl_dir = Path(labels_dir)
    tot = 0
    fg = 0
    for f in sorted(lbl_dir.glob("*.nii*")):
        a = nib.load(str(f)).get_fdata()
        tot += a.size
        fg += (a > 0).sum()
    if fg == 0:
        return 1.0
    return float(tot / (2.0 * fg))

def dice_loss_from_logits(logits, target):
    """
    logits: raw model output (N,1,Z,Y,X) or (N,Z,Y,X) after selecting channel
    target: float tensor same shape as logits (or without channel dim)
    """
    probs = torch.sigmoid(logits)
    if probs.dim() == target.dim() + 1 and probs.shape[1] == 1:
        probs = probs[:, 0]
    probs_flat = probs.reshape(probs.shape[0], -1)
    target_flat = target.reshape(target.shape[0], -1)
    num = 2.0 * (probs_flat * target_flat).sum(dim=1)
    den = probs_flat.sum(dim=1) + target_flat.sum(dim=1) + 1e-8
    loss = 1.0 - (num / den)
    return loss.mean()

def validate(model, val_loader, device):
    model.eval()
    dices = []
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            outputs = model(imgs)
            try:
                d = dice_score(outputs, labels)
            except Exception:
                probs = torch.sigmoid(outputs)
                if probs.dim() == labels.dim() + 1 and probs.shape[1] == 1:
                    probs = probs[:, 0]
                pred = (probs > 0.5).float()
                pred_flat = pred.reshape(pred.shape[0], -1)
                lbl_flat = labels.reshape(labels.shape[0], -1)
                inter = (pred_flat * lbl_flat).sum(dim=1)
                denom = pred_flat.sum(dim=1) + lbl_flat.sum(dim=1) + 1e-8
                d = (2.0 * inter / denom).cpu().numpy()
                d = float(np.mean(d))
            dices.append(float(d))

    if len(dices) == 0:
        return 0.0, 0.0, 0.0, 0.0
    arr = np.array(dices)
    return float(arr.mean()), float(arr.std()), float(arr.min()), float(arr.max())

# ---------------------------
# Training function
# ---------------------------
def train(model, dataset, output_dir, epochs, dataset_dir=None, resume_path=None, use_amp=True, lr=1e-4):
    train_loader, val_loader = dataset
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    labels_tr_dir = Path(dataset_dir) / "labelsTr" if dataset_dir is not None else None
    pos_weight = compute_pos_weight(labels_tr_dir) if (labels_tr_dir is not None and labels_tr_dir.exists()) else 1.0
    print(f"[run_train] Using device: {device} | pos_weight: {pos_weight:.4f}")

    bce_loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scaler = GradScaler() if (use_amp and device.type == "cuda") else None

    # resume if requested
    start_epoch = 0
    if resume_path:
        rp = Path(resume_path)
        if rp.exists():
            ckpt = torch.load(str(rp), map_location=device)
            if isinstance(ckpt, dict) and 'model_state' in ckpt:
                model.load_state_dict(ckpt['model_state'])
                if 'optimizer_state' in ckpt:
                    try:
                        optimizer.load_state_dict(ckpt['optimizer_state'])
                    except Exception:
                        print("[run_train] Warning: optimizer state could not be loaded (version mismatch).")
                start_epoch = int(ckpt.get('epoch', 0))
            else:
                model.load_state_dict(ckpt)
            print(f"[run_train] Resumed from {rp}, starting at epoch {start_epoch}")
        else:
            print(f"[run_train] Resume path {rp} not found, starting from scratch.")

    # optional TensorBoard writer
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=str(output_dir / "logs"))
    except Exception:
        writer = None

    best_val_dice = 0.0

    for epoch in range(start_epoch, epochs):
        model.train()
        running_loss = 0.0
        n_batches = 0

        for imgs, labels in train_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            labels_f = labels.float()
            # ensure channel dim for BCE logits [N,1,Z,Y,X]
            if labels_f.dim() == imgs.dim() - 1:
                labels_f = labels_f.unsqueeze(1)

            optimizer.zero_grad()
            if use_amp and device.type == "cuda" and scaler is not None:
                with torch.amp.autocast(device_type=device.type):
                    outputs = model(imgs)  # expected [N,C,Z,Y,X]
                    logits_fg = outputs[:, 1:2, ...]  # foreground channel
                    bce = bce_loss_fn(logits_fg, labels_f)
                    dice_l = dice_loss_from_logits(logits_fg, labels_f)
                    loss = dice_l + 0.5 * bce
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(imgs)
                logits_fg = outputs[:, 1:2, ...]
                bce = bce_loss_fn(logits_fg, labels_f)
                dice_l = dice_loss_from_logits(logits_fg, labels_f)
                loss = dice_l + 0.5 * bce
                loss.backward()
                optimizer.step()

            running_loss += float(loss.item())
            n_batches += 1

        avg_train_loss = running_loss / max(1, n_batches)
        mean_dice, std_dice, min_dice, max_dice = validate(model, val_loader, device)

        if writer is not None:
            writer.add_scalar("train/loss", avg_train_loss, epoch + 1)
            writer.add_scalar("val/dice_mean", mean_dice, epoch + 1)

        # checkpointing
        if (epoch + 1) % 5 == 0:
            torch.save({'epoch': epoch+1, 'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict()},
                       output_dir / f"model_epoch{epoch+1}.pth")
        if mean_dice > best_val_dice:
            best_val_dice = mean_dice
            torch.save({'epoch': epoch+1, 'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict()},
                       output_dir / "model_best.pth")

        print(f"[Epoch {epoch+1}/{epochs}] TrainLoss: {avg_train_loss:.4f} | Val Dice mean={mean_dice:.4f}, std={std_dice:.4f}")

    # final save
    torch.save({'epoch': epochs, 'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict()},
               output_dir / "model_final.pth")
    if writer is not None:
        writer.close()
    print(f"[run_train] Training complete. Final model saved to: {output_dir / 'model_final.pth'}")

# ---------------------------
# Dataset loader helper
# ---------------------------
def load_dataset(dataset_dir):
    print(f"[run_train] Loading dataset from: {dataset_dir}")
    dataset_dir = Path(dataset_dir)
    images_tr = sorted((dataset_dir / "imagesTr").glob("*.nii*"))
    images_val = sorted((dataset_dir / "imagesVal").glob("*.nii*")) if (dataset_dir / "imagesVal").exists() else []

    def prefixes(paths):
        return [p.stem.split('.')[0] for p in paths]

    train_ids = prefixes(images_tr)
    val_ids = prefixes(images_val)

    print(f"[run_train] Found {len(train_ids)} train ids and {len(val_ids)} val ids")
    train_loader = create_dataloader(dataset_dir, train_ids, batch_size=1, split="train", original_names=True, oversample_prob=0.6)
    val_loader   = create_dataloader(dataset_dir, val_ids,   batch_size=1, split="val",   original_names=True, shuffle=False)

    return train_loader, val_loader

# ---------------------------
# Model loader and CLI
# ---------------------------
def load_model(model_name):
    print(f"[run_train] Initializing model: {model_name}")
    if model_name == "stunet":
        return build_stunet()
    raise ValueError(f"Unknown model: {model_name}")

def main():
    parser = argparse.ArgumentParser(description="Train nnU-Net or STU-Net model")
    parser.add_argument("--dataset_dir", type=str, required=True,
                        help="Path to DatasetXXX_ImageCAS directory")
    parser.add_argument("--model", type=str, required=True,
                        choices=["nnunet", "stunet"],
                        help="Which model to train")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Where to save checkpoints and logs")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from (optional)")
    parser.add_argument("--no-amp", action="store_true",
                        help="Disable automatic mixed precision")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    args = parser.parse_args()

    dataset = load_dataset(Path(args.dataset_dir))
    model = load_model(args.model)
    train(model, dataset, Path(args.output_dir), args.epochs,
          dataset_dir=args.dataset_dir,
          resume_path=args.resume,
          use_amp=not args.no_amp,
          lr=args.lr)

if __name__ == "__main__":
    main()
