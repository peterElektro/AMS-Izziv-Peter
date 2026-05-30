import torch
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset
from pathlib import Path


# ---------------------------------------------------------
#  PATCH EXTRACTOR (random 3D patch, 128×128×128)
# ---------------------------------------------------------
def random_patch(img, lbl, patch_size=(128, 128, 128)):
    D, H, W = img.shape
    pd, ph, pw = patch_size

    # Če je slika manjša od patcha → pad
    if D < pd or H < ph or W < pw:
        pad_d = max(0, pd - D)
        pad_h = max(0, ph - H)
        pad_w = max(0, pw - W)
        img = np.pad(img, ((0, pad_d), (0, pad_h), (0, pad_w)))
        lbl = np.pad(lbl, ((0, pad_d), (0, pad_h), (0, pad_w)))
        D, H, W = img.shape

    d0 = np.random.randint(0, D - pd + 1)
    h0 = np.random.randint(0, H - ph + 1)
    w0 = np.random.randint(0, W - pw + 1)

    img_patch = img[d0:d0+pd, h0:h0+ph, w0:w0+pw]
    lbl_patch = lbl[d0:d0+pd, h0:h0+ph, w0:w0+pw]

    return img_patch, lbl_patch


# ---------------------------------------------------------
#  DATASET CLASS
# ---------------------------------------------------------
class NnUNetDataset(Dataset):
    def __init__(self, dataset_dir, case_ids):
        self.dataset_dir = Path(dataset_dir)
        self.case_ids = case_ids

    def __len__(self):
        return len(self.case_ids)

    def __getitem__(self, idx):
        case_id = self.case_ids[idx]

        img_path = self.dataset_dir / "imagesTr" / f"case_{case_id:04d}_0000.nii.gz"
        lbl_path = self.dataset_dir / "labelsTr" / f"case_{case_id:04d}.nii.gz"

        img = nib.load(str(img_path)).get_fdata().astype(np.float32)
        lbl = nib.load(str(lbl_path)).get_fdata().astype(np.int64)

        # PATCH-BASED TRAINING
        img_patch, lbl_patch = random_patch(img, lbl, patch_size=(128, 128, 128))

        # Pretvorba v PyTorch tenzorje
        img_patch = torch.tensor(img_patch)[None, ...]  # (C, D, H, W)
        lbl_patch = torch.tensor(lbl_patch)

        return img_patch, lbl_patch


# ---------------------------------------------------------
#  DATALOADER FACTORY
# ---------------------------------------------------------
def create_dataloader(dataset_dir, case_ids, batch_size=1, shuffle=True):
    dataset = NnUNetDataset(dataset_dir, case_ids)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
