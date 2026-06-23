import nibabel as nib
import numpy as np
import os

mask_path = r"D:\izzivAMS\AMS-Izziv-Peter\nnUNet_raw\Dataset501_ImageCAS\labelsTr\001.nii.gz"

img = nib.load(mask_path)
data = img.get_fdata()

print("Shape:", data.shape)
print("Unique values:", np.unique(data))
print("Dtype:", data.dtype)
