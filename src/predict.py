import torch
import sys
import os
from PIL import Image
from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import get_model


def predict(image_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = get_model(device)
    model.load_state_dict(torch.load('models/best_model.pth', map_location=device))
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor).squeeze(1)
        prob = torch.sigmoid(output).item()

    label = "FAKE (AI-generisana)" if prob > 0.5 else "REAL (Realna fotografija)"
    confidence = prob if prob > 0.5 else 1 - prob

    print(f"Slika: {image_path}")
    print(f"Predikcija: {label}")
    print(f"Pouzdanost: {confidence*100:.2f}%")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Upotreba: python src/predict.py <putanja_do_slike>")
    else:
        predict(sys.argv[1])