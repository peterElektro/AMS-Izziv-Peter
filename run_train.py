#!/usr/bin/env python3
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from models.stunet import build_stunet
from pathlib import Path
from dataloaders.nnunet_loader import create_dataloader
from metrics import dice_score

def load_dataset(dataset_dir):
    print(f"[run_train] Loading dataset from: {dataset_dir}")

    # TODO: read split file
    train_ids = list(range(1, 161))   # 160 primerov
    val_ids   = list(range(161, 181)) # 20 primerov

    train_loader = create_dataloader(dataset_dir, train_ids, batch_size=1)
    val_loader   = create_dataloader(dataset_dir, val_ids, batch_size=1)

    return train_loader, val_loader
def load_model(model_name):
    # TODO: import nnUNet or STUNet model here
    print(f"[run_train] Initializing model: {model_name}")
    if model_name == "stunet":
        return build_stunet()

    return None

def train(model, dataset, output_dir, epochs):
    train_loader, val_loader = dataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    print(f"[run_train] Using device: {device}")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for imgs, labels in train_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # VALIDACIJA
        mean_dice, std_dice, min_dice, max_dice = validate(model, val_loader, device)

        print(f"[Epoch {epoch+1}/{epochs}] "
              f"Loss: {running_loss:.4f} | "
              f"Dice mean={mean_dice:.4f}, std={std_dice:.4f}, "
              f"min={min_dice:.4f}, max={max_dice:.4f}")

    print(f"[run_train] Training complete. Saving model to: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / "model_final.pth")

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
    args = parser.parse_args()

    dataset = load_dataset(Path(args.dataset_dir))
    model = load_model(args.model)
    train(model, dataset, Path(args.output_dir), args.epochs)

def validate(model, val_loader, device):
    model.eval()
    dices = []

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            outputs = model(imgs)
            d = dice_score(outputs, labels)
            dices.append(d)

    if len(dices) == 0:
        return 0, 0, 0, 0

    dices = torch.tensor(dices)
    return dices.mean().item(), dices.std().item(), dices.min().item(), dices.max().item()

if __name__ == "__main__":
    main()
