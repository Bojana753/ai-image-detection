# Može li mašina prepoznati mašinu?
### Detekcija AI-generisanih slika korišćenjem dubokog učenja

<p align="center">
  <img src="robot-painting.jpg" width="700"/>
</p>

> Sistem prima sliku na ulazu i klasifikuje da li je u pitanju **AI-generisana slika** ili **realna fotografija** — koristeći transfer learning sa EfficientNet-B4 i Grad-CAM vizualizaciju.

**Autor:** Bojana Milošević, RA85-2022  
**Asistent:** Aleksandra Kaplar  
**Predmet:** Soft Computing — FTN Novi Sad, 2026

---

## Napomena: ovaj projekat je polazna tačka diplomskog rada

Ovo je projekat iz predmeta Soft Computing. Rezultat od 99% u nastavku važi **samo za CIFAKE dataset**. Kada sam model kasnije proverila na drugim slikama, ispostavilo se da se taj rezultat ne prenosi van CIFAKE-a (detalji u poglavlju [Šta se pokazalo nakon projekta](#šta-se-pokazalo-nakon-projekta)).

Zato u svom diplomskom radu proveravam zašto se to dešava i kako može da se popravi. Diplomski rad je u izradi, a eksperimentalni deo je završen.

---

## Rezultati na CIFAKE datasetu

| Metrika | Vrednost |
|---------|----------|
| **Accuracy** | **99%** |
| F1-Score | 0.99 |
| AUC-ROC | 0.9992 |
| Precision (FAKE) | 0.98 |
| Recall (FAKE) | 0.99 |

Ovi rezultati su dobijeni na test skupu CIFAKE-a, a najbolji model je birao po tom istom test skupu, pa su verovatno previše optimistični.

### Confusion Matrix
![Confusion Matrix](results/confusion_matrix.png)

### ROC Kriva
![ROC Curve](results/roc_curve.png)

---

## Šta se pokazalo nakon projekta

Model sam dodatno proverila na slikama veće rezolucije:

- Na pravim fotografijama visoke rezolucije bio je tačan u samo **6%** slučajeva (balansirana tačnost oko 50%), jer je skoro sve proglašavao za AI sliku.
- Potpuno sivu sliku, bez ikakvog sadržaja, proglasio je za AI sliku sa verovatnoćom **0.999**.
- Grad-CAM je na svakoj slici označavao **gornji desni ugao**, nezavisno od sadržaja.

Zaključak: model je naučio da razlikuje CIFAKE slike (32×32, uvećane) od ostalih slika, a ne prave slike od AI slika.

### Diplomski rad: nastavak projekta

U diplomskom radu (*Detekcija AI-generisanih slika: zašto detektori daju lažno visoke rezultate i kako to sprečiti*) trenirala sam tri nova modela (EfficientNet-B4) na slikama veće rezolucije, samo na slikama generatora Stable Diffusion v1.5, a testirala ih na još 6 generatora koje nisu videla u treningu.

| Model | Podaci za trening | Svrha |
|-------|-------------------|-------|
| A | tiny GenImage: AI slike su PNG, prave JPEG različitog kvaliteta | provera da li model uči format fajla umesto sadržaja |
| B | Unbiased tiny GenImage: sve slike JPEG kvaliteta 96 | glavni model |
| C | isto kao B + augmentacija (JPEG 60–100, smanjivanje, zamućenje) | otpornost na slike sa interneta |

Glavni nalazi:

- Svaki model je treniran 3 puta (seed 42, 1 i 2). Razlike između modela su mnogo veće od razlika između treninga, pa rezultati nisu slučajni.
- Model A ima 95.8% na svom test skupu, ali na generatorima van porodice Stable Diffusion pada skoro na nivo nasumičnog pogađanja. Na drugom datasetu i Model B pada na 56–89%.
- Na čistim slikama Model C je skoro jednako dobar kao B (98.0% prema 98.6%), ali lošije prepoznaje generatore koje nije video. Ne postoji jedan najbolji model: B je bolji za čiste slike, a C za slike kakve se sreću na internetu.
- Ugao u Grad-CAM-u uzrokuje zero padding. Kada sam nule zamenila odrazom slike (reflect padding) i ponovo istrenirala model, pažnja u gornjem desnom uglu je pala sa 57% na 3%, uz mali pad tačnosti na pravim slikama (sa 97.9% na 94.9%).
- Grad-CAM je pouzdan samo kada model sliku proglasi za AI. Kod pravih slika mapa pokazuje samo ugao.

Ograničenja: modeli su trenirani na samo jednom generatoru (1750 AI slika), korišćeni generatori su iz 2022. i 2023. godine, i nisam utvrdila zašto se ugao javlja baš u gornjem desnom delu slike.

---

## Dataset (projekat iz Soft Computing-a)

**CIFAKE — Real and AI-Generated Synthetic Images** ([Kaggle](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images))

- 120,000 slika ukupno (60,000 realnih + 60,000 AI-generisanih)
- Realne slike iz CIFAR-10 dataseta
- AI slike generisane pomoću **Stable Diffusion**
- Podela: 100,000 train / 20,000 test

---

## Metodologija (projekat iz Soft Computing-a)

- **Arhitektura:** EfficientNet-B4 (pretreniran na ImageNet)
- **Pristup:** Transfer learning — fine-tuning klasifikatora
- **Data augmentation:** random flip, rotacija, color jitter
- **Optimizer:** AdamW | **Epochs:** 10 | **Batch size:** 8
- **Explainability:** Grad-CAM vizualizacija

---

## Grad-CAM

Grad-CAM prikazuje koje regione slike model koristi za svoju odluku. U ovom projektu sam to prvobitno tumačila kao dokaz da model prepoznaje AI artefakte (savršene refleksije, neprirodno osvetljenje, glatke konture). Kasnija provera je pokazala da su mape bile pod jakim uticajem ugla slike, pa ih treba tumačiti oprezno.

---

## Cross-Generator Test

Model je treniran na Stable Diffusion slikama, a testiran na **DALL-E** slikama (auto, meda, žena) i postigao je 100% tačnost. To je bio test na samo tri slike, pa ne dokazuje generalizaciju na druge AI generatore. Kasnija evaluacija na 7 generatora (vidi gore) je pouzdaniji pokazatelj.

---

## Pokretanje

### Instalacija
```bash
python -m venv venv
source venv/Scripts/activate  # Windows
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install scikit-learn matplotlib numpy pillow tqdm grad-cam
```

### Trening
```bash
python src/train.py
```

### Evaluacija
```bash
python src/evaluate.py
```

### Predikcija jedne slike
```bash
python src/predict.py "putanja/do/slike.jpg"
```

### Grad-CAM vizualizacija
```bash
python src/gradcam.py
```

---

## Struktura projekta

```
ai-image-detection/
├── data/
│   ├── train/                  # 100,000 slika za trening
│   ├── test/                   # 20,000 slika za testiranje
│   └── cross_generator_test/   # DALL-E test slike
├── models/
│   └── best_model.pth          # Najbolji model (Epoch 9, 98.62%)
├── results/
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   └── gradcam_*.jpg
└── src/
    ├── dataset.py
    ├── model.py
    ├── train.py
    ├── evaluate.py
    ├── gradcam.py
    └── predict.py
```

---

## Ograničenja projekta iz Soft Computing-a

- Model je treniran na CIFAR-10 slikama (originalno 32×32 px), što dovodi do mutnih vizualizacija
- Model ne radi na slikama van CIFAKE-a (6% tačnosti na pravim fotografijama visoke rezolucije)
- Grad-CAM pokazuje bias ka uglovima slike (uzrok je zero padding)
- Najbolji model je biran po test skupu, što daje previše optimističan rezultat

---

## Poster

<p align="center">
  <img src="poster.png" width="700"/>
</p>

[Pogledaj poster](https://www.canva.com/design/DAHCWm23Zeg/zkD0GKahKvSdYiU1y0RoWg/edit?utm_content=DAHCWm23Zeg&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton)
