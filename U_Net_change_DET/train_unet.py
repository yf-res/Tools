import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision import models
import os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

# U-Net Model Definition
class UNet(nn.Module):
    def __init__(self, in_channels=6, out_channels=1):
        super(UNet, self).__init__()

        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
            )

        self.encoder1 = conv_block(in_channels, 64)
        self.encoder2 = conv_block(64, 128)
        self.encoder3 = conv_block(128, 256)
        self.encoder4 = conv_block(256, 512)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = conv_block(512, 1024)

        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.decoder4 = conv_block(1024, 512)

        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder3 = conv_block(512, 256)

        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder2 = conv_block(256, 128)

        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder1 = conv_block(128, 64)

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.pool(e1))
        e3 = self.encoder3(self.pool(e2))
        e4 = self.encoder4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        d4 = self.upconv4(b)
        d4 = torch.cat((d4, e4), dim=1)
        d4 = self.decoder4(d4)

        d3 = self.upconv3(d4)
        d3 = torch.cat((d3, e3), dim=1)
        d3 = self.decoder3(d3)

        d2 = self.upconv2(d3)
        d2 = torch.cat((d2, e2), dim=1)
        d2 = self.decoder2(d2)

        d1 = self.upconv1(d2)
        d1 = torch.cat((d1, e1), dim=1)
        d1 = self.decoder1(d1)

        return torch.sigmoid(self.final_conv(d1))

# Dataset Class
class ImageDifferenceDataset(Dataset):
    def __init__(self, image_dir, warped_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.warped_dir = warped_dir
        self.mask_dir = mask_dir
        self.image_filenames = sorted(os.listdir(image_dir))
        self.transform = transform

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_name = self.image_filenames[idx]
        img_path = os.path.join(self.image_dir, img_name)
        warped_path = os.path.join(self.warped_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)

        image = Image.open(img_path).convert("RGB")
        warped_image = Image.open(warped_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.transform:
            image = self.transform(image)
            warped_image = self.transform(warped_image)
            mask = self.transform(mask)

        input_image = torch.cat((image, warped_image), dim=0)  # Concatenate as 6-channel input (2 images)
        return input_image, mask

# Training Function
def train_unet():
    # Directories
    dataset_dir = "dataset"
    image_dir = os.path.join(dataset_dir, "images")
    warped_dir = os.path.join(dataset_dir, "warped")
    mask_dir = os.path.join(dataset_dir, "masks")

    # Check if dataset exists
    if not os.path.exists(image_dir) or not os.path.exists(warped_dir) or not os.path.exists(mask_dir):
        print("Error: Dataset directories not found!")
        return

    # Transformations
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    # Dataset & DataLoader
    dataset = ImageDifferenceDataset(image_dir, warped_dir, mask_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    # Device Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model, Loss, Optimizer
    model = UNet(in_channels=6, out_channels=1).to(device)
    criterion = nn.BCELoss()  # Binary Cross Entropy for segmentation
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Training Loop
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for inputs, masks in tqdm(dataloader, desc=f"Epoch [{epoch+1}/{num_epochs}]"):
            inputs, masks = inputs.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss / len(dataloader):.4f}")

    # Save the model
    model_path = "unet_image_difference.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved at {model_path}")

# Run Training
if __name__ == "__main__":
    train_unet()
