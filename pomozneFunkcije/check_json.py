import json

path = r"D:\izzivAMS\AMS-Izziv-Peter\nnUNet_raw\Dataset501_ImageCAS\dataset.json"
with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)

print("JSON LOADED OK:")
print(data)
