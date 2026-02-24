import torch
import torch.nn as nn
from torchvision import models

def get_model(device):
    # Učitaj pretreniran EfficientNet-B4
    model = models.efficientnet_b4(weights='IMAGENET1K_V1')
    
    # Zameni poslednji sloj za binarnu klasifikaciju
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, 1)  # 1 izlaz = REAL ili FAKE
    )
    
    model = model.to(device)
    return model


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Koristim: {device}")
    model = get_model(device)
    print("Model uspešno učitan!")
    print(f"Broj parametara: {sum(p.numel() for p in model.parameters()):,}")