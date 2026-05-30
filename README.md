# STU‑Net za segmentacijo koronarnih arterij (AMS Izziv 2026)

Repozitorij vsebuje implementacijo metode STU‑Net za segmentacijo koronarnih arterij (CAS) na 3D CTA slikah iz dataseta ImageCAS.  
Projekt sledi zahtevam iz izziva pri predmetu AMS: delujoč trening, validacija, testiranje, primerjava z nnU‑Net baseline in reproducibilnost.

---

## FAZA 1 — Osnovni trening STU‑Net (zaključeno)

### Kaj je bilo narejeno
- implementiran STU‑Net model  
- pripravljen dataloader v nnU‑Net formatu  
- uveden patch‑based training (128×128×128)  
- trening deluje na GPU  
- model se shrani v `outputs/stunet/`

### Struktura projekta v tej fazi
/models
stunet.py
/dataloaders
nnunet_loader.py
run_train.py

### Kako poganjam to fazo
python run_train.py \
--dataset_dir data/nnunet_raw/Dataset501_ImageCAS \
--model stunet \
--output_dir outputs/stunet \
--epochs 1

---

## FAZA 2 — Validacija in Dice metrika (zaključeno)

### Kaj je bilo narejeno
- dodana datoteka `metrics.py` z Dice metriko  
- dodan validacijski del v `run_train.py`  
- izpis Dice statistike po epochih  
- ločen train/val del dataseta

Primer izpisa:
[Epoch 1/1] Loss: 17.6550 | Dice mean=0.0012, std=0.0034, min=0.0000, max=0.0148

### Struktura projekta v tej fazi
/models
stunet.py
/dataloaders
nnunet_loader.py
metrics.py
run_train.py

### Kako poganjam to fazo
Enako kot v FAZI 1, saj se validacija izvede samodejno:
python run_train.py --dataset_dir ... --model stunet --output_dir ... --epochs 1

---

## FAZA 3 — Sliding‑window inference in shranjevanje predikcij (naslednje)

### Cilji
- implementacija 3D sliding‑window inferenca  
- združevanje patchov v celoten volumen  
- shranjevanje predikcij v .nii.gz  
- priprava skript `run_inference.py` in `run_test.py`

### Struktura projekta po zaključku te faze
/models
stunet.py
/dataloaders
nnunet_loader.py
/inference
sliding_window.py
metrics.py
run_train.py
run_test.py
run_inference.py

### Kako se bo poganjalo
Primer (ko bo implementirano):
python run_inference.py \
--input_path data/nnunet_raw/Dataset501_ImageCAS/imagesTs \
--model_path outputs/stunet/model_final.pth \
--output_path outputs/stunet/predictions


---

## FAZA 4 — Primerjava z nnU‑Net baseline

### Cilji
- implementacija osnovnega nnU‑Net baseline  
- trening na istem splitu kot STU‑Net  
- eval na celotnih volumnih  
- priprava tabele rezultatov

### Struktura projekta po tej fazi
/models
stunet.py
nnunet_baseline.py
/dataloaders
nnunet_loader.py
/inference
sliding_window.py
metrics.py
run_train.py
run_test.py
run_inference.py

### Kako se bo poganjalo
Primer:
python run_train.py --model nnunet_baseline ...
python run_test.py --model nnunet_baseline ...

---

## FAZA 5 — Končna tabela rezultatov

### Cilji
- izračun Mean/Std/Min/Max Dice  
- grafi učenja  
- priprava rezultatov za poročilo

### Struktura projekta
Dodana bo mapa z rezultati:
/results
dice_scores.csv
learning_curves.png

---

## FAZA 6 — Docker in reproducibilnost

### Cilji
- priprava Dockerfile  
- zagotovitev delovanja skript preko CLI argumentov  
- navodila za reproduciranje rezultatov

### Kako se bo poganjalo
Primer:
docker run --gpus all -v .:/workspace container python3 run_train.py ...


---

## Opomba

Faze so zapisane zato, da je napredek jasen in da lahko delo nadaljujem brez izgube konteksta.  
Repozitorij se bo sproti dopolnjeval z datotekami za testiranje, inferenco in Docker.
