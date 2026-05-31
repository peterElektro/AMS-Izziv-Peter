import nibabel as nib, pathlib, numpy as np
p=pathlib.Path(r"D:\izzivAMS\AMS-Izziv-Peter\data_preproc\Dataset501_ImageCAS\labelsTr")
tot=0; fg=0
for f in p.glob("*.nii*"):
    a=nib.load(str(f)).get_fdata()
    tot+=a.size; fg+=(a>0).sum()
print("fg ratio:", fg/tot)