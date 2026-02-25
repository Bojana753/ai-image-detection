# AI Image Detection

Sistem za automatsku detekciju AI-generisanih slika korišćenjem EfficientNet-B4 modela sa Grad-CAM vizualizacijom.

## Opis projekta

Sistem prima sliku na ulazu i klasifikuje da li je u pitanju AI-generisana slika ili realna fotografija. Koristi transfer learning pristup sa pretreniranim EfficientNet-B4 modelom i Grad-CAM tehniku za vizualizaciju delova slike koje model koristi za odluku.

## Rezultati

| Metrika | Vrednost |
|---------|----------|
| Accuracy | 99% |
| F1-Score | 0.99 |
| AUC-ROC | 0.9992 |
| Precision (FAKE) | 0.98 |
| Recall (FAKE) | 0.99 |

## Dataset

[CIFAKE - Real and AI-Generated Synthetic Images](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images)
- 120,000 slika (60,000 realnih + 60,000 AI-generisanih)
- Realne slike iz CIFAR-10 dataseta
- AI slike generisane pomoću Stable Diffusion

## Tehnologije

- Python 3.12
- PyTorch 2.7
- EfficientNet-B4 (ImageNet pretrained)
- Grad-CAM
- scikit-learn, matplotlib, OpenCV

## Struktura projekta
```
ai-image-detection/
├── data/
│   ├── train/          # 100,000 slika za trening
│   ├── test/           # 20,000 slika za testiranje
│   └── cross_generator_test/  # DALL-E test slike
├── models/
│   └── best_model.pth  # Najbolji model (Epoch 9, 98.62%)
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

## Ograničenja

- Model je treniran na CIFAR-10 slikama (originalno 32x32 piksela) što dovodi do mutnih vizualizacija
- Grad-CAM pokazuje bias ka uglovima slike zbog karakteristika dataseta
- Model je primarno treniran na Stable Diffusion slikama, pa generalizacija na druge generatore (DALL-E, Midjourney) može varirati

## Poster
[Pogledaj poster](https://www.canva.com/design/DAHCWm23Zeg/zkD0GKahKvSdYiU1y0RoWg/edit?utm_content=DAHCWm23Zeg&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton)

## Autor

Bojana Milošević, RA85-2022  
Asistent: Aleksandra Kaplar