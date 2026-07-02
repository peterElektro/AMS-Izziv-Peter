<h1 style="color:#2b7cff; font-size:32px; font-weight:700; margin-bottom:0;">
STU‑Net za segmentacijo koronarnih arterij (AMS Izziv 2026)
</h1>

<p style="font-size:17px; line-height:1.55;">
Implementacija 3D STU‑Net‑Lite+ za segmentacijo koronarnih arterij (CAS) na CTA volumnih iz dataseta ImageCAS.
<br><br>
Projekt je bil najprej v celoti razvit na lokalnem PC‑ju, nato pa reproduciran v Docker okolju na LSTWorker strežniku, kjer so bile prilagojene poti, ukazi in sintakse zaradi novejših različic knjižnic.
</p>

<hr style="border:1px solid #2b7cff;">

<h2 style="color:#2b7cff; font-size:26px; font-weight:700;">
Zahteve iz AMS izziva
</h2>

<ul style="font-size:17px; line-height:1.55;">
<li>delujoč trening</li>
<li>validacija</li>
<li>testiranje</li>
<li>primerjava z nnU‑Net baseline</li>
<li>popolna reproducibilnost (Docker)</li>
</ul>

<h2 style="color:#2b7cff; font-size:26px; font-weight:700;">
Repozitorij vsebuje
</h2>

<ul style="font-size:17px; line-height:1.55;">
<li>STU‑Net‑Lite+ model</li>
<li>nnU‑Net kompatibilen dataloader</li>
<li>optimiziran trening pipeline</li>
<li>validacijo z Dice metriko</li>
<li>sliding‑window inferenco</li>
<li>baseline primerjavo</li>
<li>vizualizacijo segmentacij (GIF)</li>
<li>Dockerfile za reproducibilnost</li>
</ul>

<hr style="border:1px solid #2b7cff;">

<h2 style="color:#2b7cff; font-size:26px; font-weight:700;">
Docker reproducibilnost
</h2>

<p style="font-size:17px; line-height:1.55;">
Projekt je popolnoma reproducibilen v Docker okolju na LSTWorker strežniku.
</p>

<h3 style="color:#2b7cff; font-size:22px; font-weight:600;">Zagon Docker okolja z GPU podporo</h3>

```bash
docker run --rm -it --gpus all --shm-size=16g \
  -v /media/FastDataMama/peterT/AMS-Izziv-Peter:/workspace \
  ams-izziv-peter bash
<h3 style="color:#2b7cff; font-size:22px; font-weight:600;">Struktura skript</h3>

<table>
<tr><th>Skripta</th><th>Namen</th></tr>
<tr><td>run_train.py</td><td>Trening STU‑Net‑Lite+</td></tr>
<tr><td>run_inference.py</td><td>Sliding‑window inferenca STU‑Net</td></tr>
<tr><td>run_test.py</td><td>Evalvacija STU‑Net (Dice, HD95)</td></tr>
<tr><td>nnUNetv2_train</td><td>Trening nnU‑Net baseline</td></tr>
<tr><td>nnUNetv2_predict</td><td>Inferenca nnU‑Net baseline</td></tr>
<tr><td>medpy_eval.py</td><td>Evalvacija nnU‑Net predikcij</td></tr>
</table>

<hr style="border:1px solid #2b7cff;">

<h2 style="color:#2b7cff; font-size:26px; font-weight:700;">
FAZA 1 — Osnovni STU‑Net trening (zaključeno)
</h2>

<ul style="font-size:17px; line-height:1.55;">
<li>Implementiran STU‑Net‑Lite+ model.</li>
<li>Pripravljen nnU‑Net kompatibilen dataloader.</li>
<li>Uveden patch‑based training (128×128×128).</li>
<li>Trening deluje na GPU.</li>
<li>Model se shranjuje v <code>outputs/stunet/</code>.</li>
</ul>

<h3 style="color:#2b7cff; font-size:22px; font-weight:600;">Struktura projekta</h3>

bash
/models  
    stunet.py  
/dataloaders  
    nnunet_loader.py  
run_train.py
<h3 style="color:#2b7cff; font-size:22px; font-weight:600;">Zagon</h3>

bash
python run_train.py \
  --dataset_dir data/nnunet_raw/Dataset501_ImageCAS \
  --model stunet \
  --output_dir outputs/stunet \
  --epochs 1
<hr style="border:1px solid #2b7cff;">

<h2 style="color:#2b7cff; font-size:26px; font-weight:700;">
FAZA 2 — Validacija in Dice metrika (zaključeno)
</h2>

<ul style="font-size:17px; line-height:1.55;">
<li>Dodana Dice metrika (mean/std/min/max).</li>
<li>Validacija po vsaki epohi.</li>
<li>Train/val split: 1–160 / 161–180.</li>
<li>Izpis statistike po epochah.</li>
</ul>

<h3 style="color:#2b7cff; font-size:22px; font-weight:600;">Primer izpisa</h3>

bash
[Epoch 1/1] Loss: 17.6550 | Dice mean=0.0012 | std=0.0034 | min=0.0000 | max=0.0148
<hr style="border:1px solid #2b7cff;">

<h2 style="color:#2b7cff; font-size:26px; font-weight:700;">
FAZA 3 — Sliding‑window inference (zaključeno)
</h2>

<ul style="font-size:17px; line-height:1.55;">
<li>3D sliding‑window inferenca.</li>
<li>Rekonstrukcija celotnega volumna.</li>
<li>Shranjevanje predikcij v .nii.gz.</li>
<li>Skripta: <code>run_inference.py</code>.</li>
</ul>

bash
python run_inference.py \
  --model_path outputs/stunet_long/model_best.pth \
  --input_dir data/nnunet_raw/Dataset501_ImageCAS/imagesTs \
  --output_dir outputs/stunet_predictions
<hr style="border:1px solid #2b7cff;">

<h2 style="color:#2b7cff; font-size:26px; font-weight:700;">
FAZA 4 — Eval STU‑Net (test 181–200)
</h2>

<ul style="font-size:17px; line-height:1.55;">
<li>Eval na testnem delu CAS (181–200).</li>
<li>Izračun Dice in HD95.</li>
<li>GIF vizualizacija: <code>gif_AMSIzziv.gif</code>.</li>
</ul>

<h3 style="color:#2b7cff; font-size:22px; font-weight:600;">Rezultati</h3>

<div style="background:#e8f2ff; padding:12px; border-radius:8px; font-size:17px;">
<b>Mean Dice:</b> 0.20<br>
<b>Mean HD95:</b> 175 mm
</div>

<p align="center">
<img src="gif_AMSIzziv.gif" width="600">
</p>

<hr style="border:1px solid #2b7cff;">

<h2 style="color:#2b7cff; font-size:26px; font-weight:700;">
FAZA 5 — nnU‑Net baseline (zaključeno)
</h2>

<ul style="font-size:17px; line-height:1.55;">
<li>Implementacija osnovnega nnU‑Net baseline.</li>
<li>Trening na 1–160 (lokalni PC).</li>
<li>Eval na internem validation splitu.</li>
<li>Pridobljen <code>summary.json</code>.</li>
<li>Baseline reproduciran v Dockerju.</li>
</ul>

<h3 style="color:#2b7cff; font-size:22px; font-weight:600;">Rezultati</h3>

<div style="background:#e8f2ff; padding:12px; border-radius:8px; font-size:17px;">
<b>Mean Dice:</b> 0.77<br>
<b>Mean IoU:</b> 0.63<br>
<b>Range Dice:</b> 0.65–0.86
</div>

<hr style="border:1px solid #2b7cff;">

<h2 style="color:#2b7cff; font-size:26px; font-weight:700;">
FAZA 6 — Primerjava modelov
</h2>

<table>
<tr><th>Model</th><th>Eval set</th><th>Mean Dice</th><th>Mean HD95</th></tr>
<tr><td>STU‑Net</td><td>Test (181–200)</td><td>0.20</td><td>175 mm</td></tr>
<tr><td>nnU‑Net baseline</td><td>Val (1–160)</td><td>0.77</td><td>–</td></tr>
</table>

<hr style="border:1px solid #2b7cff;">

<h2 style="color:#2b7cff; font-size:26px; font-weight:700;">
FAZA 7 — Docker in reproducibilnost
</h2>

<ul style="font-size:17px; line-height:1.55;">
<li>Dockerfile.</li>
<li>Reproducibilni trening STU‑Net.</li>
<li>Reproducibilni trening nnU‑Net.</li>
<li>Prilagojene poti in ukazi.</li>
<li>Celoten workflow je mogoče pognati iz CLI.</li>
</ul>

<hr style="border:1px solid #2b7cff;">

<h2 style="color:#2b7cff; font-size:26px; font-weight:700;">
Implementacije skript
</h2>

<h3 style="color:#2b7cff; font-size:22px; font-weight:600;">models/stunet.py</h3>

<ul style="font-size:17px; line-height:1.55;">
<li>Implementacija STU‑Net‑Lite+.</li>
<li>ConvBlock3d, SEBlock3d, STUBlock3d.</li>
<li><code>build_stunet()</code> vrne inicializiran model.</li>
<li>Podpira AMP, channels_last_3d, torch.compile.</li>
</ul>

<h3 style="color:#2b7cff; font-size:22px; font-weight:600;">dataloaders/nnunet_loader.py</h3>

<ul style="font-size:17px; line-height:1.55;">
<li>Nalaganje NIfTI (.nii.gz) volumnov.</li>
<li>imagesTr/imagesVal/imagesTs + labels struktura.</li>
<li>Random 3D patch extraction.</li>
<li>Oversampling foreground primerov.</li>
<li>Optimiziran DataLoader (num_workers=6, pin_memory=True, persistent_workers=True, prefetch_factor=4).</li>
</ul>

<h3 style="color:#2b7cff; font-size:22px; font-weight:600;">run_train.py</h3>

<ul style="font-size:17px; line-height:1.55;">
<li>AMP (mixed precision).</li>
<li><code>torch.compile(model)</code>.</li>
<li>channels_last_3d.</li>
<li>cudnn.benchmark=True.</li>
<li>Validacija po epochah.</li>
<li>Shranjevanje <code>model_best.pth</code> in <code>model_final.pth</code>.</li>
</ul>

<h3 style="color:#2b7cff; font-size:22px; font-weight:600;">metrics.py</h3>

<ul style="font-size:17px; line-height:1.55;">
<li>Dice metrika (mean/std/min/max).</li>
</ul>

<h3 style="color:#2b7cff; font-size:22px; font-weight:600;">run_inference.py</h3>

<ul style="font-size:17px; line-height:1.55;">
<li>Sliding‑window inferenca.</li>
<li>Shranjevanje predikcij v .nii.gz.</li>
</ul>

<h3 style="color:#2b7cff; font-size:22px; font-weight:600;">run_test.py</h3>

<ul style="font-size:17px; line-height:1.55;">
<li>Eval na test setu.</li>
<li>Primerjava STU‑Net vs nnU‑Net baseline.</li>
</ul>

<hr style="border:1px solid #2b7cff;">

<h2 style="color:#2b7cff; font-size:26px; font-weight:700;">
Rezultati 200‑epoh STU‑Net treninga
</h2>

<div style="background:#e8f2ff; padding:12px; border-radius:8px; font-size:17px;">
<b>Train loss:</b> 1.49 → ~1.0<br>
<b>Najboljši val Dice (mean):</b> ≈ 0.27<br>
<b>Tipičen razpon val Dice:</b> 0.10–0.25
</div>