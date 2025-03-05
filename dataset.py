import os
import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class ThermalDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.transform = transform
        self.image_filenames = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.png'))]

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.folder_path, self.image_filenames[idx])
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        image = Image.fromarray(image)  # OpenCV -> PIL formatı
        if self.transform:
            image = self.transform(image)
        return image

# Veri dönüşümleri
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

def get_dataloader(folder_path, batch_size):
    dataset = ThermalDataset(folder_path, transform=transform)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
