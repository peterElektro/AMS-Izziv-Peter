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

# ✅ Razvoj projekta

Projekt je bil razvit postopoma skozi več faz, od osnovne implementacije STU-Net do popolnoma reproducibilnega Docker okolja.

| Faza | Opis | Status |
|------|------|:------:|
| 🧪 FAZA 1 | Osnovni STU-Net trening | ✅ |
| 📊 FAZA 2 | Validacija in Dice metrika | ✅ |
| 🪟 FAZA 3 | Sliding-window inferenca | ✅ |
| 📈 FAZA 4 | Evalvacija na testnem setu | ✅ |
| 🧠 FAZA 5 | nnU-Net baseline | ✅ |
| ⚔️ FAZA 6 | Primerjava modelov | ✅ |
| 🐳 FAZA 7 | Docker reproducibilnost | ✅ |
| 🔄 | Pretvorba ImageCAS → nnU-Net | ✅ |
| 📂 | 4-fold Cross Validation | ✅ |

---

# 🧪 FAZA 1 — Osnovni STU-Net trening

### Izvedene funkcionalnosti

- ✅ Implementacija **STU-Net-Lite+**
- ✅ nnU-Net kompatibilen dataloader
- ✅ Patch-based training (**128×128×128**)
- ✅ GPU pospešen trening
- ✅ Samodejno shranjevanje modelov

Modeli se shranjujejo v:

```text
outputs/stunet/
```

## Uporabljeni razrez podatkov

| Namen | Primeri |
|--------|---------|
| Train | 1–160 |
| Validation | 161–180 |
| Test | 181–200 |

> **Opomba**
>
> Zaradi počasnejše internetne povezave sem za razvoj uporabil manjši del podatkovnega nabora. Cilj je bil hitrejši razvoj in testiranje implementacije.

### Zagon treninga

```bash
python run_train.py \
    --dataset_dir data/nnunet_raw/Dataset501_ImageCAS \
    --model stunet \
    --output_dir outputs/stunet \
    --epochs 1
```

---

# 📊 FAZA 2 — Validacija in Dice metrika

### Implementirano

- ✅ Dice Mean
- ✅ Dice Std
- ✅ Dice Min
- ✅ Dice Max
- ✅ Validacija po vsaki epohi

Train / Validation razrez:

```text
Train: 1–160
Validation: 161–180
```

### Primer izpisa

```text
[Epoch 1/1]
Loss: 17.6550

Dice
-------------------------
Mean : 0.0012
Std  : 0.0034
Min  : 0.0000
Max  : 0.0148
```

---

# 🪟 FAZA 3 — Sliding-window inferenca

### Implementirano

- ✅ 3D Sliding Window
- ✅ Rekonstrukcija celotnega volumna
- ✅ Shranjevanje `.nii.gz` predikcij

### Zagon

```bash
python run_inference.py \
    --model_path outputs/stunet_long/model_best.pth \
    --input_dir data/nnunet_raw/Dataset501_ImageCAS/imagesTs \
    --output_dir outputs/stunet_predictions
```

---

# 📈 FAZA 4 — Evalvacija STU-Net

Evalvacija je bila izvedena na testnem delu podatkov (**181–200**).

## Rezultati

| Metrika | Rezultat |
|---------|----------:|
| Mean Dice | **0.20** |
| Mean HD95 | **175 mm** |

### Vizualizacija

<p align="center">
<img src="gif_AMSIzziv.gif" width="700">
</p>

---

# 🧠 FAZA 5 — nnU-Net Baseline

Za primerjavo je bil uporabljen tudi osnovni nnU-Net model.

## Rezultati

| Metrika | Rezultat |
|---------|----------:|
| Mean Dice | **0.77** |
| Mean IoU | **0.63** |
| Dice Range | **0.65–0.86** |

---

# ⚔️ FAZA 6 — Primerjava modelov

| Model | Evalvacijski sklop | Mean Dice | Mean HD95 |
|------|-------------------|----------:|----------:|
| **STU-Net** | Test (181–200) | **0.20** | **175 mm** |
| **nnU-Net** | Validation | **0.77** | — |

---

# 🐳 FAZA 7 — Docker reproducibilnost

Projekt je v celoti reproducibilen z uporabo Dockerja.

Vključuje:

- ✅ Dockerfile
- ✅ Reproducibilen trening STU-Net
- ✅ Reproducibilen trening nnU-Net
- ✅ Prilagojene poti in CLI ukaze
- ✅ Celoten workflow od treninga do evalvacije

---

# 📈 Rezultati po 200 epohah treninga

| Metrika | Vrednost |
|---------|---------:|
| Začetni Train Loss | **1.49** |
| Končni Train Loss | **~1.0** |
| Najboljši Validation Dice | **≈ 0.27** |
| Tipičen Validation Dice | **0.10–0.25** |

> **Povzetek**
>
> Implementacija uspešno podpira celoten cevovod:
>
> - pripravo podatkov,
> - trening,
> - validacijo,
> - inferenco,
> - evalvacijo,
> - primerjavo z nnU-Net baseline,
> - popolno reproducibilnost v Docker okolju.