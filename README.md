# 🚀 STU-Net za segmentacijo koronarnih arterij (AMS Izziv 2026)

![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)
![Docker](https://img.shields.io/badge/Docker-Reproducible-2496ED)
![nnU-Net](https://img.shields.io/badge/nnU--Net-Compatible-success)
![CUDA](https://img.shields.io/badge/CUDA-GPU-green)

---

## 📖 Opis projekta

Implementacija **3D STU-Net-Lite+** za segmentacijo koronarnih arterij (CAS) na CTA volumnih iz podatkovnega nabora **ImageCAS**.

Projekt je bil najprej razvit lokalno, nato pa popolnoma reproduciran v Docker okolju na strežniku **LSTWorker**.

> **Opomba**
>
> Za učenje sem uporabil **manjši obseg podatkov (1–160)**, ker imam doma počasnejšo internetno povezavo in bi prenos ter trening celotnega nabora (1–600) trajala bistveno dlje.
>
> - **Train:** 1–160
> - **Validation:** 161–180
> - **Test:** 181–200

---

# ⭐ Glavne značilnosti

- ⚡ Implementacija **STU-Net-Lite+**
- 📦 nnU-Net kompatibilen dataloader
- 🧠 Patch-based training (128×128×128)
- 📊 Dice metrika + validacija po epochah
- 🪟 Sliding-window 3D inferenca
- 🔁 Primerjava z nnU-Net baseline
- 🐳 Popolna Docker reproducibilnost
- 🔄 Podpora za 4-fold cross-validation
- 🎞 GIF vizualizacija segmentacij
- 📈 Primerjava rezultatov STU-Net in nnU-Net

---

# 📁 Struktura projekta

```text
AMS-Izziv-Peter
│
├── models
│   └── stunet.py
│
├── dataloaders
│   └── nnunet_loader.py
│
├── metrics.py
├── run_train.py
├── run_inference.py
├── run_test.py
├── convert_to_nnunet.py
├── Dockerfile
└── README.md
```

---

# 🐳 Docker reproducibilnost

Projekt je popolnoma reproducibilen v Docker okolju.

## Zagon Docker okolja

```bash
docker run --rm -it \
  --gpus all \
  --shm-size=16g \
  -v /media/FastDataMama/peterT/AMS-Izziv-Peter:/workspace \
  ams-izziv-peter bash
```

---

# 🔄 Pretvorba ImageCAS → nnU-Net format

AMS izziv zahteva uporabo podatkov v **nnU-Net** strukturi (`imagesTr`, `labelsTr`, `imagesTs`).

## Struktura

```text
Dataset501_ImageCAS
│
├── imagesTr
├── labelsTr
├── imagesTs
└── dataset.json
```

## Pretvorba

```bash
python convert_to_nnunet.py \
    --input_dir /media/FastDataMama/izziv/ImageCAS \
    --output_dir data/nnunet_raw/Dataset501_ImageCAS
```

---

# 🧩 Pregled skript

| Skripta | Namen |
|---------|-------|
| `run_train.py` | Trening STU-Net-Lite+ |
| `run_inference.py` | Sliding-window inferenca |
| `run_test.py` | Evalvacija (Dice, HD95) |
| `nnUNetv2_train` | Trening nnU-Net baseline |
| `nnUNetv2_predict` | Inferenca nnU-Net baseline |
| `medpy_eval.py` | Evalvacija nnU-Net |
| `convert_to_nnunet.py` | Pretvorba ImageCAS → nnU-Net |

---

# ⚙️ CLI argumenti

## `run_train.py`

| Argument | Opis |
|----------|------|
| `--dataset_dir` | Pot do nnU-Net dataseta |
| `--model` | Ime modela (`stunet`) |
| `--output_dir` | Kam shraniti modele |
| `--epochs` | Število epoh |
| `--resume` | Nadaljevanje treninga |

---

## `run_test.py`

| Argument | Opis |
|----------|------|
| `--dataset_dir` | Pot do testnega seta |
| `--model_path` | Pot do `model_best.pth` |
| `--output_dir` | Izhodna mapa za metrike |

---

## `run_inference.py`

| Argument | Opis |
|----------|------|
| `--model_path` | Pot do modela |
| `--input_dir` | Pot do `imagesTs` |
| `--output_dir` | Mapa za predikcije |

---

# 🔄 4-fold Cross Validation

Projekt podpira vse štiri folde, kot zahteva AMS izziv.

## Primer za Fold 1

```bash
python run_train.py \
    --dataset_dir data/fold1 \
    --model stunet \
    --output_dir outputs/fold1 \
    --epochs 200

python run_test.py \
    --dataset_dir data/fold1 \
    --model_path outputs/fold1/model_best.pth \
    --output_dir metrics/fold1
```

Za ostale folde:

```text
data/fold2
data/fold3
data/fold4
```

Vsak fold vsebuje svoj `dataset.json` in svoj train/validation/test razrez.

---

# ✅ Faze razvoja

- [x] FAZA 1 — Osnovni STU-Net trening
- [x] FAZA 2 — Validacija + Dice metrika
- [x] FAZA 3 — Sliding-window inferenca
- [x] FAZA 4 — Eval na testnem setu
- [x] FAZA 5 — nnU-Net baseline
- [x] FAZA 6 — Primerjava modelov
- [x] FAZA 7 — Docker reproducibilnost
- [x] Pretvorba ImageCAS → nnU-Net
- [x] Podpora za 4-fold cross-validation

🧪 FAZA 1 — Osnovni STU‑Net trening
STU‑Net‑Lite+ implementiran

nnU‑Net kompatibilen dataloader

Patch‑based training (128×128×128)

GPU pospešek

Modeli v outputs/stunet/

Uporabljeni spliti (moj projekt)
Train: 1–160

Val: 161–180

Test: 181–200

Razlog: doma imam slab internet, zato sem treniral na manjšem obsegu, da je učenje hitreje končalo.

Zagon
bash
python run_train.py \
  --dataset_dir data/nnunet_raw/Dataset501_ImageCAS \
  --model stunet \
  --output_dir outputs/stunet \
  --epochs 1
📊 FAZA 2 — Validacija + Dice metrika
Dice mean/std/min/max

Validacija po epochah

Train/val split: 1–160 / 161–180

Primer izpisa
bash
[Epoch 1/1] Loss: 17.6550 | Dice mean=0.0012 | std=0.0034 | min=0.0000 | max=0.0148
🪟 FAZA 3 — Sliding‑window inferenca
3D sliding‑window

Rekonstrukcija volumna

Predikcije v .nii.gz

Zagon
bash
python run_inference.py \
  --model_path outputs/stunet_long/model_best.pth \
  --input_dir data/nnunet_raw/Dataset501_ImageCAS/imagesTs \
  --output_dir outputs/stunet_predictions
🧪 FAZA 4 — Eval STU‑Net (test 181–200)
Rezultati
Mean Dice: 0.20

Mean HD95: 175 mm

<p align="center">
<img src="gif_AMSIzziv.gif" width="600">
</p>

🧠 FAZA 5 — nnU‑Net baseline
Rezultati
Mean Dice: 0.77

Mean IoU: 0.63

Range Dice: 0.65–0.86

⚔️ FAZA 6 — Primerjava modelov
Model	Eval set	Mean Dice	Mean HD95
STU‑Net	Test (181–200)	0.20	175 mm
nnU‑Net baseline	Val (1–160)	0.77	–


🐳 FAZA 7 — Docker reproducibilnost
Dockerfile

Reproducibilni trening STU‑Net

Reproducibilni trening nnU‑Net

Prilagojene poti in ukazi

Celoten workflow preko CLI

📈 Rezultati 200‑epoh STU‑Net treninga
Train loss: 1.49 → ~1.0

Najboljši val Dice (mean): ≈ 0.27

Tipičen razpon val Dice: 0.10–0.25