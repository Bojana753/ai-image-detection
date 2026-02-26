import sys
sys.path.insert(0, 'src')
import torch
from model import get_model
from gradcam import run_gradcam

device = torch.device('cuda')
model = get_model(device)
model.load_state_dict(torch.load('models/best_model.pth'))
model.eval()

run_gradcam('data/cross_generator_test/dalle_woman.jpg', model, device)