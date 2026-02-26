import torch.nn as nn
from torchvision import models


def get_model(device):
    model = models.efficientnet_b4(weights='IMAGENET1K_V1')
    
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, 1)
    )
    
    return model.to(device)