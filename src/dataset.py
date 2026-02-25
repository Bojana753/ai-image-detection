import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class CIFAKEDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.transform = transform
        
        for label, folder in enumerate(['REAL', 'FAKE']):
            folder_path = os.path.join(root_dir, folder)
            for img_name in os.listdir(folder_path):
                if img_name.endswith('.jpg'):
                    self.samples.append((os.path.join(folder_path, img_name), label))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label


def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], 
                             [0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], 
                             [0.229, 0.224, 0.225])
    ])
    
    return train_transform, test_transform


def get_dataloaders(data_dir, batch_size=32):
    train_transform, test_transform = get_transforms()
    
    train_dataset = CIFAKEDataset(os.path.join(data_dir, 'train'), train_transform)
    test_dataset = CIFAKEDataset(os.path.join(data_dir, 'test'), test_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        
    return train_loader, test_loader