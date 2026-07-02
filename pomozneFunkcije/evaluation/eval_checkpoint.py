import os
import random
from pathlib import Path
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader

def random_patch(img, lbl, patch_size=(128, 128, 128)):
    z, y, x = img.shape
    pz, py, px = patch_size

    pad_z = max(0, pz - z)
    pad_y = max(0, py - y)
    pad_x = max(0, px - x)
    if pad_z or pad_y or pad_x:
        img = np.pad(img, ((0, pad_z), (0, pad_y), (0, pad_x)), mode='constant', constant_values=0)
        lbl = np.pad(lbl, ((0, pad_z), (0, pad_y), (0, pad_x)), mode='constant', constant_values=0)
        z, y, x = img.shape

    z0 = np.random.randint(0, z - pz + 1) if z - pz + 1 > 0 else 0
    y0 = np.random.randint(0, y - py + 1) if y - py + 1 > 0 else 0
    x0 = np.random.randint(0, x - px + 1) if x - px + 1 > 0 else 0

    img_patch = img[z0:z0 + pz, y0:y0 + py, x0:x0 + px]
    lbl_patch = lbl[z0:z0 + pz, y0:y0 + py, x0:x0 + px]

    return img_patch, lbl_patch

class NnUNetDataset(Dataset):
    """
    Dataset with optional oversampling of foreground cases.
    Expects folders: imagesTr/imagesVal/imagesTs and labelsTr/labelsVal/labelsTs
    """
    def __init__(self, dataset_dir, case_ids, split="train", original_names=False,
                 patch_size=(128,128,128), transform=None, oversample_prob=0.0):
        self.dataset_dir = Path(dataset_dir)
        self.case_ids = [str(i) for i in case_ids]
        self.all_case_ids = list(self.case_ids)
        self.split = split  # "train","val","test"
        self.original_names = original_names
        self.patch_size = tuple(patch_size)
        self.transform = transform
        self.oversample_prob = float(oversample_prob)

        split_map = {"train": "Tr", "val": "Val", "test": "Ts"}
        suf = split_map.get(split, "Tr")
        self.images_dir = self.dataset_dir / f"images{suf}"
        self.labels_dir = self.dataset_dir / f"labels{suf}"

        if not self.images_dir.exists() or not self.labels_dir.exists():
            raise FileNotFoundError(f"Expected folders {self.images_dir} and {self.labels_dir}")

        # compute foreground cases for oversampling (only for train)
        self.fg_cases = []
        if self.split == "train" and self.oversample_prob > 0:
            for p in sorted((self.labels_dir).glob("*.nii*")):
                try:
                    a = nib.load(str(p)).get_fdata()
                    if (a > 0).sum() > 0:
                        self.fg_cases.append(p.stem.split('.')[0])
                except Exception:
                    # skip unreadable files
                    continue

    def __len__(self):
        return len(self.case_ids)

    def _find_paths(self, cid):
        # try original naming first if requested
        if self.original_names:
            img_name = f"{cid}.img.nii.gz"
            lbl_name = f"{cid}.label.nii.gz"
            img_path = self.images_dir / img_name
            lbl_path = self.labels_dir / lbl_name
            if img_path.exists() and lbl_path.exists():
                return img_path, lbl_path
        # fallback to case_XXXX style
        try:
            num = int(cid)
        except:
            num = None
        if num is not None:
            img_name = f"case_{num:04d}_0000.nii.gz"
            lbl_name = f"case_{num:04d}.nii.gz"
            img_path = self.images_dir / img_name
            lbl_path = self.labels_dir / lbl_name
            if img_path.exists() and lbl_path.exists():
                return img_path, lbl_path
        # try flexible matching (any file that starts with cid)
        for f in self.images_dir.iterdir():
            if f.is_file() and f.name.startswith(cid):
                base = f.stem
                candidates = list(self.labels_dir.glob(f"{cid}*.nii*")) + list(self.labels_dir.glob(f"{base}*.nii*"))
                if candidates:
                    return f, candidates[0]
        raise FileNotFoundError(f"Missing image/label for id {cid}. Tried original and case patterns in {self.images_dir}")

    def _sample_case_id(self, idx):
        # If training and oversample enabled, sample foreground case with probability oversample_prob
        if self.split == "train" and self.oversample_prob > 0 and len(self.fg_cases) > 0 and random.random() < self.oversample_prob:
            return random.choice(self.fg_cases)
        # otherwise use provided index (deterministic) or random fallback
        if idx is None:
            return random.choice(self.all_case_ids)
        return self.case_ids[idx]

    def __getitem__(self, idx):
        # idx may be integer; allow oversampling by sampling a different case id
        cid = self._sample_case_id(idx)
        img_path, lbl_path = self._find_paths(cid)

        img = nib.load(str(img_path)).get_fdata().astype(np.float32)
        lbl = nib.load(str(lbl_path)).get_fdata().astype(np.int64)

        # ensure 3D arrays
        if img.ndim == 4 and img.shape[0] in (1,):
            img = img[0]
        if lbl.ndim > 3:
            lbl = np.squeeze(lbl)

        # patch-based training
        img_patch, lbl_patch = random_patch(img, lbl, patch_size=self.patch_size)

        # add channel dim for image (C, Z, Y, X)
        img_patch = np.expand_dims(img_patch, axis=0)

        img_t = torch.from_numpy(img_patch).float()
        lbl_t = torch.from_numpy(lbl_patch).long()

        if self.transform:
            img_t, lbl_t = self.transform(img_t, lbl_t)

        return img_t, lbl_t

def create_dataloader(dataset_dir, case_ids, batch_size=1, split="train", original_names=False,
                      shuffle=True, num_workers=0, patch_size=(128,128,128), pin_memory=True, oversample_prob=0.0):
    ds = NnUNetDataset(dataset_dir, case_ids, split=split, original_names=original_names,
                       patch_size=patch_size, transform=None, oversample_prob=oversample_prob)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)
