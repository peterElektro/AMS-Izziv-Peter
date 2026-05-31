import os, shutil, json

BASE = r"D:\izzivAMS\AMS-Izziv-Peter\data\nnunet_raw\Dataset501_ImageCAS"
RAW  = r"D:\izzivAMS\AMS-Izziv-Peter\data\raw"
SPLIT_JSON = r"D:\izzivAMS\AMS-Izziv-Peter\splits\split1_test.json"

imagesTr = os.path.join(BASE, "imagesTr")
imagesVal = os.path.join(BASE, "imagesVal")
imagesTs = os.path.join(BASE, "imagesTs")
labelsTr = os.path.join(BASE, "labelsTr")
labelsVal = os.path.join(BASE, "labelsVal")
labelsTs = os.path.join(BASE, "labelsTs")

for d in (imagesTr, imagesVal, imagesTs, labelsTr, labelsVal, labelsTs):
    os.makedirs(d, exist_ok=True)

with open(SPLIT_JSON, "r") as f:
    split = json.load(f)

def src_paths(idx):
    idx = str(idx)
    return os.path.join(RAW, f"{idx}.img.nii.gz"), os.path.join(RAW, f"{idx}.label.nii.gz")

def copy_pair(idx, dst_img_dir, dst_lbl_dir):
    img_src, lbl_src = src_paths(idx)
    if not os.path.exists(img_src) or not os.path.exists(lbl_src):
        raise FileNotFoundError(f"Missing pair for id {idx}: {img_src} or {lbl_src}")
    shutil.copy2(img_src, os.path.join(dst_img_dir, os.path.basename(img_src)))
    shutil.copy2(lbl_src, os.path.join(dst_lbl_dir, os.path.basename(lbl_src)))

for i in split.get("train", []):
    copy_pair(i, imagesTr, labelsTr)

for i in split.get("val", []):
    copy_pair(i, imagesVal, labelsVal)

for i in split.get("test", []):
    copy_pair(i, imagesTs, labelsTs)

print("Clean split done: original names preserved in images*/labels*.")
