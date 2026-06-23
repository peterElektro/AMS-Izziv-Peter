# STU‑Net za segmentacijo koronarnih arterij (AMS Izziv 2026)

Implementacija 3D STU‑Net‑Lite+ za segmentacijo koronarnih arterij (CAS) na CTA volumnih iz dataseta ImageCAS.

Projekt sledi zahtevam iz izziva pri predmetu AMS:
- delujoč trening,
- validacija,
- testiranje,
- primerjava z nnU‑Net baseline,
- popolna reproducibilnost (Docker).

Repozitorij vsebuje:
- STU‑Net‑Lite+ model,
- nnU‑Net kompatibilen dataloader,
- optimiziran trening pipeline,
- validacijo z Dice metriko,
- sliding‑window inferenco,
- baseline primerjavo,
- vizualizacijo segmentacij (GIF).

---

## FAZA 1 — Osnovni STU‑Net trening (zaključeno)
- Implementiran STU‑Net‑Lite+ model.
- Pripravljen nnU‑Net kompatibilen dataloader.
- Uveden patch‑based training (128×128×128).
- Trening deluje na GPU.
- Model se shranjuje v outputs/stunet/.

**Struktura projekta**
/models  
 stunet.py  
/dataloaders  
 nnunet_loader.py  
run_train.py

**Zagon**
python run_train.py \
  --dataset_dir data/nnunet_raw/Dataset501_ImageCAS \
  --model stunet \
  --output_dir outputs/stunet \
  --epochs 1

---

## FAZA 2 — Validacija in Dice metrika (zaključeno)
- Dodana Dice metrika (mean/std/min/max).
- Validacija po vsaki epohi.
- Ločen train/val split:
  - train: 1–160
  - val: 161–180
- Izpis statistike po epochah.

Primer izpisa:
[Epoch 1/1] Loss: 17.6550 | Dice mean=0.0012, std=0.0034, min=0.0000, max=0.0148

**Struktura**
/models  
 stunet.py  
/dataloaders  
 nnunet_loader.py  
metrics.py  
run_train.py

---

## FAZA 3 — Sliding‑window inference (zaključeno)
- 3D sliding‑window inferenca.
- Rekonstrukcija celotnega volumna.
- Shranjevanje predikcij v .nii.gz.
- Skripta: run_inference.py.

**Zagon**
python run_inference.py \
  --model_path outputs/stunet_long/model_best.pth \
  --input_dir data/nnunet_raw/Dataset501_ImageCAS/imagesTs \
  --output_dir outputs/stunet_predictions

---

## FAZA 4 — Eval STU‑Net (test 181–200)
- Eval na testnem delu CAS (181–200).
- Izračun Dice in HD95.
- Pripravljena GIF vizualizacija: gif_AMSIzziv.gif.

**Rezultati**
- Mean Dice: 0.20
- Mean HD95: 175 mm
<p align="center">
  <img src="gif_AMSIzziv.gif" width="600">
</p>

---

## FAZA 5 — nnU‑Net baseline (zaključeno)
- Implementacija osnovnega nnU‑Net baseline.
- Trening na 1–160.
- Eval na internem validation splitu.
- Pridobljen summary.json.

**Rezultati**
- Mean Dice: 0.77
- Mean IoU: 0.63
- Range Dice: 0.65–0.86

---

## FAZA 6 — Primerjava modelov

| Model           | Eval set        | Mean Dice | Mean HD95 |
|-----------------|-----------------|-----------|-----------|
| STU‑Net         | Test (181–200)  | 0.20      | 175 mm    |
| nnU‑Net baseline| Val (1–160)     | 0.77      | –         |

---

## FAZA 7 — Docker in reproducibilnost
- Dockerfile.
- Reproducibilni trening.
- CLI‑based workflow.

---

## models/stunet.py
- Implementacija STU‑Net‑Lite+.
- ConvBlock3d, SEBlock3d, STUBlock3d.
- build_stunet() vrne inicializiran model.
- Podpira AMP, channels_last_3d in torch.compile.

## dataloaders/nnunet_loader.py
- Nalaganje NIfTI (.nii.gz) volumnov.
- imagesTr/imagesVal/imagesTs + labels struktura.
- Random 3D patch extraction.
- Oversampling foreground primerov.
- Optimiziran DataLoader:
  - num_workers=6
  - pin_memory=True
  - persistent_workers=True
  - prefetch_factor=4

## run_train.py
- Glavni trening skript.
- AMP (mixed precision).
- torch.compile(model).
- channels_last_3d.
- cudnn.benchmark=True.
- Validacija po epochah.
- Shranjevanje model_best.pth in model_final.pth.

## metrics.py
- Dice metrika (mean/std/min/max).

## run_inference.py
- Sliding‑window inferenca.
- Shranjevanje predikcij v .nii.gz.

## run_test.py
- Eval na test setu.
- Primerjava STU‑Net vs nnU‑Net baseline.

---

## Osnovni trening (1 epoha)
python run_train.py \
  --dataset_dir data/nnunet_raw/Dataset501_ImageCAS \
  --model stunet \
  --output_dir outputs/stunet \
  --epochs 1

## Polni trening (200 epoh)
python run_train.py \
  --dataset_dir data/nnunet_raw/Dataset501_ImageCAS \
  --model stunet \
  --output_dir outputs/stunet_long \
  --epochs 200

## Nadaljevanje treninga
python run_train.py \
  --dataset_dir data/nnunet_raw/Dataset501_ImageCAS \
  --model stunet \
  --output_dir outputs/stunet_long \
  --epochs 200 \
  --resume outputs/stunet_long/model_best.pth

---

# Dnevni napredek — Petek, 05.06.2026

## 1. Popolna optimizacija DataLoaderja
- num_workers=6
- pin_memory=True
- persistent_workers=True
- prefetch_factor=4  
→ GPU usage iz 40% → 90–100%.

## 2. Optimizacija trening pipeline‑a
- torch.compile(model)
- channels_last_3d
- cudnn.benchmark=True
- Non‑blocking GPU transfer
- Stabiliziran pos_weight
- Čiščenje run_train.py

## 3. Stabilen učni trend (10 epoch)
- Epoch 1:  Dice 0.0357
- Epoch 5:  Dice 0.0735
- Epoch 6:  Dice 0.0991
- Epoch 10: Dice 0.1209

## 4. Priprava za 200‑epoch run
- Pipeline stabilen.
- GPU optimalno izkoriščen.

## 5. Naslednji koraki
- Normalizacija intenzitet.
- Augmentacija.
- SoftDice loss.
- Sliding‑window inferenca.
- nnU‑Net baseline.

---

# Rezultati 200‑epoh STU‑Net treninga

Trening na Dataset501_ImageCAS (patch‑based, 128³, brez augmentacije):

- Train loss: 1.49 → ~1.0
- Najboljši val Dice (mean): ≈ 0.27
- Tipičen razpon val Dice: 0.10–0.25

