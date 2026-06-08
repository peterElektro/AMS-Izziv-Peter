import os
import random
from pathlib import Path
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader


# -----------------------------
# NORMALIZACIJA
# -----------------------------
def normalize(img):
    img = np.clip(img, -200, 800)
    mean = img.mean()
    std = img.std() + 1e-8
    return (img - mean) / std


# -----------------------------
# AUGMENTACIJA
# -----------------------------
def augment(img, lbl):
    # flip Z
    if random.random() < 0.5:
        img = np.flip(img, axis=0).copy()
        lbl = np.flip(lbl, axis=0).copy()

    # flip Y
    if random.random() < 0.5:
        img = np.flip(img, axis=1).copy()
        lbl = np.flip(lbl, axis=1).copy()

    # flip X
    if random.random() < 0.5:
        img = np.flip(img, axis=2).copy()
        lbl = np.flip(lbl, axis=2).copy()

    # gaussian noise
    if random.random() < 0.3:
        img = img + np.random.normal(0, 0.05, img.shape)

    return img, lbl


# -----------------------------
# MIRROR PADDING
# -----------------------------
def mirror_pad(img, lbl, patch_size):
    z, y, x = img.shape
    pz, py, px = patch_size

    pad_z = max(0, pz - z)
    pad_y = max(0, py - y)
    pad_x = max(0, px - x)

    if pad_z or pad_y or pad_x:
        img = np.pad(img,
                     ((0, pad_z), (0, pad_y), (0, pad_x)),
                     mode='reflect')
        lbl = np.pad(lbl,
                     ((0, pad_z), (0, pad_y), (0, pad_x)),
                     mode='constant', constant_values=0)

    return img, lbl


# -----------------------------
# RANDOM PATCH
# -----------------------------
def random_patch(img, lbl, patch_size=(128, 128, 128)):
    img, lbl = mirror_pad(img, lbl, patch_size)

    z, y, x = img.shape
    pz, py, px = patch_size

    z0 = np.random.randint(0, z - pz + 1)
    y0 = np.random.randint(0, y - py + 1)
    x0 = np.random.randint(0, x - px + 1)

    img_patch = img[z0:z0 + pz, y0:y0 + py, x0:x0 + px]
    lbl_patch = lbl[z0:z0 + pz, y0:y0 + py, x0:x0 + px]

    return img_patch, lbl_patch


# -----------------------------
# DATASET
# -----------------------------
class NnUNetDataset(Dataset):
    def __init__(self, dataset_dir, case_ids, split="train", original_names=False,
                 patch_size=(128,128,128), oversample_prob=0.0):

        self.dataset_dir = Path(dataset_dir)
        self.case_ids = [str(i) for i in case_ids]
        self.all_case_ids = list(self.case_ids)
        self.split = split
        self.original_names = original_names
        self.patch_size = tuple(patch_size)
        self.oversample_prob = float(oversample_prob)

        split_map = {"train": "Tr", "val": "Val", "test": "Ts"}
        suf = split_map.get(split, "Tr")
        self.images_dir = self.dataset_dir / f"images{suf}"
        self.labels_dir = self.dataset_dir / f"labels{suf}"

        if not self.images_dir.exists() or not self.labels_dir.exists():
            raise FileNotFoundError(f"Expected folders {self.images_dir} and {self.labels_dir}")

        # oversampling foreground cases
        self.fg_cases = []
        if self.split == "train" and self.oversample_prob > 0:
            for p in sorted(self.labels_dir.glob("*.nii*")):
                try:
                    a = nib.load(str(p)).get_fdata()
                    if (a > 0).sum() > 0:
                        self.fg_cases.append(p.stem.split('.')[0])
                except:
                    continue

    def __len__(self):
        return len(self.case_ids)

    # -----------------------------
    # FIND IMAGE + LABEL PATHS
    # -----------------------------
    def _find_paths(self, cid):
        if self.original_names:
            img_name = f"{cid}.img.nii.gz"
            lbl_name = f"{cid}.label.nii.gz"
            img_path = self.images_dir / img_name
            lbl_path = self.labels_dir / lbl_name
            if img_path.exists() and lbl_path.exists():
                return img_path, lbl_path

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

        for f in self.images_dir.iterdir():
            if f.is_file() and f.name.startswith(cid):
                candidates = list(self.labels_dir.glob(f"{cid}*.nii*"))
                if candidates:
                    return f, candidates[0]

        raise FileNotFoundError(f"Missing image/label for id {cid}")

    # -----------------------------
    # OVERSAMPLING
    # -----------------------------
    def _sample_case_id(self, idx):
        if self.split == "train" and self.oversample_prob > 0 and len(self.fg_cases) > 0 and random.random() < self.oversample_prob:
            return random.choice(self.fg_cases)
        if idx is None:
            return random.choice(self.all_case_ids)
        return self.case_ids[idx]

    # -----------------------------
    # GET ITEM
    # -----------------------------
    def __getitem__(self, idx):
        cid = self._sample_case_id(idx)
        img_path, lbl_path = self._find_paths(cid)

        img = nib.load(str(img_path)).get_fdata().astype(np.float32)
        lbl = nib.load(str(lbl_path)).get_fdata().astype(np.int64)

        lbl = (lbl > 0).astype(np.int64)

        if img.ndim == 4 and img.shape[0] == 1:
            img = img[0]
        if lbl.ndim > 3:
            lbl = np.squeeze(lbl)

        # normalize BEFORE augment
        img = normalize(img)

        # augment only on train
        if self.split == "train":
            img, lbl = augment(img, lbl)

        # patch extraction
        img_patch, lbl_patch = random_patch(img, lbl, patch_size=self.patch_size)

        # add channel dim
        img_patch = np.expand_dims(img_patch, axis=0)

        img_t = torch.from_numpy(img_patch).float()
        lbl_t = torch.from_numpy(lbl_patch).long()

        return img_t, lbl_t


# -----------------------------
# DATALOADER
# -----------------------------
def create_dataloader(dataset_dir, case_ids, batch_size=1, split="train", original_names=False,
                      shuffle=True, num_workers=6, patch_size=(128,128,128),
                      pin_memory=True, oversample_prob=0.0):

    ds = NnUNetDataset(
        dataset_dir,
        case_ids,
        split=split,
        original_names=original_names,
        patch_size=patch_size,
        oversample_prob=oversample_prob
    )

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True,
        prefetch_factor=4
    )

    return loader
