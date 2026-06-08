#!/usr/bin/env python3
import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
import nibabel as nib
import numpy as np

from models.stunet import build_stunet
from dataloaders.nnunet_loader import create_dataloader
from losses import BCEDiceLoss


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


def validate(model, val_loader, device):
    model.eval()
    dices = []
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            outputs = model(imgs)
            probs = torch.sigmoid(outputs)
            pred = (probs > 0.5).float()

            pred_flat = pred.reshape(pred.shape[0], -1)
            lbl_flat = labels.reshape(labels.shape[0], -1)

            inter = (pred_flat * lbl_flat).sum(dim=1)
            denom = pred_flat.sum(dim=1) + lbl_flat.sum(dim=1) + 1e-8
            d = (2.0 * inter / denom).cpu().numpy()
            dices.append(float(np.mean(d)))

    if len(dices) == 0:
        return 0, 0, 0, 0

    arr = np.array(dices)
    return float(arr.mean()), float(arr.std()), float(arr.min()), float(arr.max())


def train(model, dataset, output_dir, epochs, dataset_dir=None, resume_path=None, use_amp=True, lr=1e-4):
    train_loader, val_loader = dataset
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    labels_tr_dir = Path(dataset_dir) / "labelsTr"
    pos_weight = compute_pos_weight(labels_tr_dir)
    print(f"[run_train] Using device: {device} | pos_weight: {pos_weight:.4f}")

    criterion = BCEDiceLoss(bce_weight=0.5, dice_weight=0.5)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    scaler = GradScaler(device="cuda") if (use_amp and device.type == "cuda") else None

    best_val_dice = 0.0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        n_batches = 0

        for imgs, labels in train_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            labels_f = labels.float().unsqueeze(1)

            optimizer.zero_grad()

            if use_amp and scaler is not None:
                with autocast(device_type=device.type):
                    outputs = model(imgs)
                    loss = criterion(outputs, labels_f)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(imgs)
                loss = criterion(outputs, labels_f)
                loss.backward()
                optimizer.step()

            running_loss += float(loss.item())
            n_batches += 1

        scheduler.step()

        avg_train_loss = running_loss / max(1, n_batches)
        mean_dice, std_dice, min_dice, max_dice = validate(model, val_loader, device)

        print(f"[Epoch {epoch+1}/{epochs}] TrainLoss: {avg_train_loss:.4f} | Val Dice mean={mean_dice:.4f}, std={std_dice:.4f}")

        if mean_dice > best_val_dice:
            best_val_dice = mean_dice
            torch.save({'epoch': epoch+1, 'model_state': model.state_dict()},
                       output_dir / "model_best.pth")

    torch.save({'epoch': epochs, 'model_state': model.state_dict()},
               output_dir / "model_final.pth")
    print(f"[run_train] Training complete. Final model saved to: {output_dir / 'model_final.pth'}")


def load_dataset(dataset_dir):
    print(f"[run_train] Loading dataset from: {dataset_dir}")
    dataset_dir = Path(dataset_dir)

    images_tr = sorted((dataset_dir / "imagesTr").glob("*.nii*"))
    images_val = sorted((dataset_dir / "imagesVal").glob("*.nii*"))

    def prefixes(paths):
        return [p.stem.split('.')[0] for p in paths]

    train_ids = prefixes(images_tr)
    val_ids = prefixes(images_val)

    print(f"[run_train] Found {len(train_ids)} train ids and {len(val_ids)} val ids")

    train_loader = create_dataloader(dataset_dir, train_ids, batch_size=1,
                                     split="train", original_names=True, oversample_prob=0.6)
    val_loader = create_dataloader(dataset_dir, val_ids, batch_size=1,
                                   split="val", original_names=True, shuffle=False)

    return train_loader, val_loader


def load_model(model_name):
    print(f"[run_train] Initializing model: {model_name}")
    if model_name == "stunet":
        return build_stunet()
    raise ValueError(f"Unknown model: {model_name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--model", type=str, required=True, choices=["stunet"])
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--lr", type=float, default=1e-4)
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
