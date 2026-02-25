import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision import transforms
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import get_model


def load_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)


def denormalize(tensor):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    img = std * img + mean
    return np.clip(img, 0, 1)


def run_gradcam(image_path, model, device):
    input_tensor = load_image(image_path).to(device)
    rgb_img = denormalize(input_tensor)

    target_layer = [model.features[-1]]
    cam = GradCAM(model=model, target_layers=target_layer)

    output = model(input_tensor)
    prob = torch.sigmoid(output).item()
    pred_label = "FAKE" if prob > 0.5 else "REAL"
    confidence = prob if prob > 0.5 else 1 - prob

    targets = [ClassifierOutputTarget(0)]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]
    visualization = show_cam_on_image(rgb_img.astype(np.float32), grayscale_cam, use_rgb=True)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(rgb_img)
    axes[0].set_title('Original')
    axes[0].axis('off')
    axes[1].imshow(visualization)
    axes[1].set_title(f'Grad-CAM | {pred_label} ({confidence*100:.1f}%)')
    axes[1].axis('off')
    plt.tight_layout()

    output_path = os.path.join('results', f'gradcam_{os.path.basename(image_path)}')
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Predikcija: {pred_label} ({confidence*100:.1f}%)")
    print(f"Sačuvano: {output_path}")


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(device)
    model.load_state_dict(torch.load('models/best_model.pth', map_location=device))
    model.eval()

    for label in ['REAL', 'FAKE']:
        folder = os.path.join('data', 'test', label)
        images = os.listdir(folder)[:2]
        for img_name in images:
            img_path = os.path.join(folder, img_name)
            print(f"\n--- {label}: {img_name} ---")
            run_gradcam(img_path, model, device)


if __name__ == '__main__':
    main()