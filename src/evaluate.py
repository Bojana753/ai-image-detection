import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (confusion_matrix, classification_report,
                              roc_auc_score, roc_curve, ConfusionMatrixDisplay)
from tqdm import tqdm
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset import get_dataloaders
from model import get_model


def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating"):
            images = images.to(device)
            outputs = model(images).squeeze(1)
            probs = torch.sigmoid(outputs).cpu().numpy()
            preds = (probs > 0.5).astype(int)
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    _, test_loader = get_dataloaders('data', batch_size=32)

    model = get_model(device)
    model.load_state_dict(torch.load('models/best_model.pth', map_location=device))

    labels, preds, probs = evaluate(model, test_loader, device)

    print(classification_report(labels, preds, target_names=['REAL', 'FAKE']))
    print(f"AUC-ROC: {roc_auc_score(labels, probs):.4f}")

    cm = confusion_matrix(labels, preds)
    ConfusionMatrixDisplay(cm, display_labels=['REAL', 'FAKE']).plot(cmap='Blues')
    plt.title('Confusion Matrix')
    plt.savefig('results/confusion_matrix.png', dpi=150)
    plt.close()

    fpr, tpr, _ = roc_curve(labels, probs)
    auc = roc_auc_score(labels, probs)
    plt.plot(fpr, tpr, label=f'AUC = {auc:.4f}')
    plt.plot([0, 1], [0, 1], '--', color='gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.savefig('results/roc_curve.png', dpi=150)
    plt.close()


if __name__ == '__main__':
    main()