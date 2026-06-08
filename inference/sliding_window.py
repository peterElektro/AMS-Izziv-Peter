import torch
import numpy as np
from typing import Tuple


def create_gaussian_weight(patch_size, sigma_scale=1./8):
    pD, pH, pW = patch_size

    z = np.linspace(-1, 1, pD)
    y = np.linspace(-1, 1, pH)
    x = np.linspace(-1, 1, pW)
    zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")

    sigma = sigma_scale
    weight = np.exp(-(xx**2 + yy**2 + zz**2) / (2 * sigma**2))
    weight = weight / weight.max()

    return weight.astype(np.float32)


def compute_strides(patch_size: Tuple[int, int, int], overlap: float):
    pD, pH, pW = patch_size
    return (
        int(pD * (1 - overlap)),
        int(pH * (1 - overlap)),
        int(pW * (1 - overlap)),
    )


def allocate_output_buffers(volume_shape, num_classes=1):
    C, D, H, W = volume_shape
    output = np.zeros((num_classes, D, H, W), dtype=np.float32)
    norm_map = np.zeros((1, D, H, W), dtype=np.float32)
    return output, norm_map


def sliding_window_inference(volume: np.ndarray,
                             model: torch.nn.Module,
                             patch_size: Tuple[int, int, int],
                             overlap: float,
                             device: str = "cuda"):

    model.eval()
    model.to(device)

    C, D, H, W = volume.shape
    stride = compute_strides(patch_size, overlap)

    output, norm_map = allocate_output_buffers((1, D, H, W))

    gaussian = create_gaussian_weight(patch_size)

    # -------------------------
    # FIX 1: cover full volume
    # -------------------------
    zs = list(range(0, D - patch_size[0] + 1, stride[0]))
    ys = list(range(0, H - patch_size[1] + 1, stride[1]))
    xs = list(range(0, W - patch_size[2] + 1, stride[2]))

    if zs[-1] != D - patch_size[0]:
        zs.append(D - patch_size[0])
    if ys[-1] != H - patch_size[1]:
        ys.append(H - patch_size[1])
    if xs[-1] != W - patch_size[2]:
        xs.append(W - patch_size[2])

    gaussian = gaussian.astype(np.float32)

    with torch.no_grad():
        for z in zs:
            for y in ys:
                for x in xs:

                    patch = volume[:, 
                                   z:z+patch_size[0],
                                   y:y+patch_size[1],
                                   x:x+patch_size[2]]

                    patch = torch.from_numpy(patch).unsqueeze(0).float().to(device)

                    # forward pass
                    pred = model(patch)
                    pred = torch.sigmoid(pred)
                    pred = pred.cpu().numpy()[0, 0]

                    # apply gaussian weighting
                    weighted_pred = pred * gaussian

                    # accumulate numerator
                    output[0,
                           z:z+patch_size[0],
                           y:y+patch_size[1],
                           x:x+patch_size[2]] += weighted_pred

                    # accumulate denominator (CRITICAL FIX)
                    norm_map[0,
                             z:z+patch_size[0],
                             y:y+patch_size[1],
                             x:x+patch_size[2]] += gaussian

    # -------------------------
    # FIX 2: stable normalization
    # -------------------------
    eps = 1e-5
    output = output / (norm_map + eps)

    return output