import torch

def dice_score(pred, target, eps=1e-6):
    """
    pred: (B, C, D, H, W) logits
    target: (B, D, H, W) integer labels
    """
    pred = torch.argmax(pred, dim=1)

    pred = pred.float()
    target = target.float()

    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()

    dice = (2 * intersection + eps) / (union + eps)
    return dice.item()
