import torch
import torch.nn as nn
import torchvision.transforms as transforms
import cv2
import numpy as np
import argparse
from PIL import Image
import matplotlib.pyplot as plt
from train_unet import UNet


# Inference function
def predict(model_path, image_path, warped_path):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    image = transform(Image.open(image_path).convert("RGB"))
    warped_image = transform(Image.open(warped_path).convert("RGB"))

    input_tensor = torch.cat((image, warped_image), dim=0).unsqueeze(0)

    model = UNet(in_channels=6, out_channels=1)
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()

    with torch.no_grad():
        prediction = model(input_tensor).squeeze().numpy()

    plt.imshow(prediction, cmap="hot")
    plt.title("Predicted Differences")
    plt.colorbar()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--warped", required=True)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()

    predict(args.model, args.image, args.warped)
