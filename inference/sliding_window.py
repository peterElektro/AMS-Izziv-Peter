import torch
import numpy as np
import torch.nn.functional as F

def sliding_window_inference(volume, model, patch_size, overlap, device):
    """
    volume: numpy array (C, D, H, W)
    patch_size: tuple (pD, pH, pW)
    overlap: float (0.0–0.9)
    """

    model.eval()
    model.to(device)

    C, D, H, W = volume.shape
    pD, pH, pW = patch_size

    stride = (
        int(pD * (1 - overlap)),
        int(pH * (1 - overlap)),
        int(pW * (1 - overlap)),
    )

    output = np.zeros((1, D, H, W), dtype=np.float32)
    norm_map = np.zeros((1, D, H, W), dtype=np.float32)

    with torch.no_grad():
        for z in range(0, D - pD + 1, stride[0]):
            for y in range(0, H - pH + 1, stride[1]):
                for x in range(0, W - pW + 1, stride[2]):

                    patch = volume[:, z:z+pD, y:y+pH, x:x+pW]
                    patch = torch.from_numpy(patch).unsqueeze(0).float().to(device)

                    pred = model(patch)
                    pred = F.softmax(pred, dim=1)
                    pred = pred.cpu().numpy()[0, 1]  # kanal 1 = foreground

                    output[0, z:z+pD, y:y+pH, x:x+pW] += pred
                    norm_map[0, z:z+pD, y:y+pH, x:x+pW] += 1

    output = output / np.maximum(norm_map, 1e-6)
    return output
