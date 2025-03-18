import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np


class ThermalDataset(Dataset):
    def __init__(self, root, transform=None, augmentations_per_image=5):
        self.root = root
        self.transform = transform
        self.augmentations_per_image = augmentations_per_image
        self.images = [os.path.join(root, f) for f in os.listdir(root) if f.endswith('.jpg')]

    def __len__(self):
        return len(self.images) * self.augmentations_per_image

    def __getitem__(self, idx):
        img_path = self.images[idx // self.augmentations_per_image]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img



# Ağ mimarileri
class Generator(nn.Module):
    def __init__(self, nz= 100, ngf=64, nc=3):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 16, 4, 1, 0, bias=False),   # 4x4
            nn.BatchNorm2d(ngf * 16),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, 2, 1, bias=False),  # 8x8
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),  # 16x16
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),  # 32x32
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),      # 64x64
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),           # 128x128
            nn.Tanh()
        )

    def forward(self, input):
        return self.model(input)


class Discriminator(nn.Module):
    def __init__(self, nc, ndf):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),           # 64x64
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),      # 32x32
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),  # 16x16
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),  # 8x8
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 8, ndf * 16, 4, 2, 1, bias=False), # 4x4
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 16, 1, 4, 1, 0, bias=False),       # 1x1
        )

    def forward(self, input):
        return self.model(input)


# Parametreler
nz = 100  # Latent vektör boyutu
ngf = 128  # Generator filtre boyutu (128 olarak değiştirildi)
ndf = 128  # Discriminator filtre boyutu (128 olarak değiştirildi)
nc = 3  # RGB görüntüler için kanal sayısı
batch_size = 8
epochs = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dönüşümler ve veri yükleyici
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),  # ±10 derece döndürme
    transforms.RandomResizedCrop(128, scale=(0.9, 1.1)),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


dataset = ThermalDataset("Termal", transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

print(f"dataset have image count: {len(dataset)}")
# Model oluşturma
generator = Generator(nz, ngf, nc).to(device)
discriminator = Discriminator(nc, ndf).to(device)

# Kayıp fonksiyonları ve optimizasyon
g_criterion = nn.BCEWithLogitsLoss()
d_criterion = nn.BCEWithLogitsLoss()
g_optimizer = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.9))
d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.9))

# Eğitim döngüsü
for epoch in range(epochs):
    generator.train() # Modeli eğitim moduna al
    discriminator.train() # Modeli eğitim moduna al
    for i, data in enumerate(dataloader):
        real_images = data.to(device)
        batch_size = real_images.size(0)

        # Gerçek veriler için etiketler (Label Smoothing)
        real_labels = torch.ones((batch_size, 1), device=device)
        fake_labels = torch.zeros((batch_size, 1), device=device)

        # Discriminator için eğitim
        discriminator.zero_grad()
        output = discriminator(real_images).view(-1, 1)
        loss_real = d_criterion(output, real_labels)
        loss_real.backward()

        noise = torch.randn(batch_size, nz, 1, 1, device=device)
        fake_images = generator(noise)
        output = discriminator(fake_images.detach()).view(-1, 1)
        loss_fake = d_criterion(output, fake_labels)
        loss_fake.backward()
        # torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 1.0)  # Ağırlık kırpma gereksiz görüldü 
        d_optimizer.step()

        # Generator için eğitim
        generator.zero_grad()
        output = discriminator(fake_images).view(-1, 1)
        loss_gen = g_criterion(output, real_labels)
        loss_gen.backward()
        g_optimizer.step()

    print(
        f"Epoch [{epoch + 1}/{epochs}] Loss D: {loss_real.item() + loss_fake.item():.4f}, Loss G: {loss_gen.item():.4f}")

    if epoch % 10 == 0:
        with torch.no_grad():
            fake_images = generator(torch.randn(16, nz, 1, 1, device=device)).cpu()
        save_path = f"generated_epoch_{epoch}.png"
        vutils.save_image(fake_images, save_path, normalize=True)
        print(f"Görüntü kaydedildi: {save_path}")
