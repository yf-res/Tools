import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import cv2


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 8, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 8, 3, stride=2, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 8, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class MLAnomalyDetector:
    def __init__(self, patch_size=64, epochs=50, batch_size=32):
        self.model = Autoencoder()
        self.patch_size = patch_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def extract_patches(self, image):
        patches = []
        for i in range(0, image.shape[0] - self.patch_size + 1, self.patch_size // 2):
            for j in range(0, image.shape[1] - self.patch_size + 1, self.patch_size // 2):
                patch = image[i:i+self.patch_size, j:j+self.patch_size]
                patches.append(patch)
        return np.array(patches)

    def train(self, image):
        patches = self.extract_patches(image)
        patches = patches[:, np.newaxis, :, :] / 255.0  # Normalize and add channel dimension
        dataset = TensorDataset(torch.FloatTensor(patches))
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters())

        self.model.train()
        for epoch in range(self.epochs):
            for data in dataloader:
                img = data[0].to(self.device)
                optimizer.zero_grad()
                outputs = self.model(img)
                loss = criterion(outputs, img)
                loss.backward()
                optimizer.step()

    def detect_anomalies(self, image, threshold=3):
        self.model.eval()
        patches = self.extract_patches(image)
        patches = patches[:, np.newaxis, :, :] / 255.0
        dataset = TensorDataset(torch.FloatTensor(patches))
        dataloader = DataLoader(dataset, batch_size=self.batch_size)

        reconstruction_errors = []
        with torch.no_grad():
            for data in dataloader:
                img = data[0].to(self.device)
                outputs = self.model(img)
                error = ((outputs - img) ** 2).mean(dim=(1, 2, 3))
                reconstruction_errors.extend(error.cpu().numpy())

        reconstruction_errors = np.array(reconstruction_errors)
        mean_error = np.mean(reconstruction_errors)
        std_error = np.std(reconstruction_errors)
        anomaly_scores = (reconstruction_errors - mean_error) / std_error

        # Create anomaly map
        anomaly_map = np.zeros_like(image, dtype=float)
        count_map = np.zeros_like(image, dtype=float)
        idx = 0
        for i in range(0, image.shape[0] - self.patch_size + 1, self.patch_size // 2):
            for j in range(0, image.shape[1] - self.patch_size + 1, self.patch_size // 2):
                anomaly_map[i:i+self.patch_size, j:j+self.patch_size] += anomaly_scores[idx]
                count_map[i:i+self.patch_size, j:j+self.patch_size] += 1
                idx += 1

        anomaly_map /= count_map
        anomalies = anomaly_map > threshold

        return anomalies, anomaly_map

# Usage
# detector = MLAnomalyDetector()
# img = cv2.imread('your_image.png', 0)  # Read as grayscale
# detector.train(img)  # Train on the image (assumes most of the image is "normal")
# anomalies, anomaly_map = detector.detect_anomalies(img)
# cv2.imshow('Detected Anomalies', anomalies.astype(np.uint8) * 255)
# cv2.imshow('Anomaly Map', anomaly_map / anomaly_map.max())
# cv2.waitKey(0)


"""
Key points about these methods:

Multi-scale Analysis:

Can detect anomalies of various sizes
Works well for both bright and dark anomalies
Doesn't require training, but may need parameter tuning


Machine Learning (Autoencoder):

Can potentially detect more complex anomalies
Requires training, which can be done on the image itself (assuming most of the image is "normal")
May struggle with very small datasets or images with many anomalies
Can be computationally intensive, especially for large images



Both methods have their strengths and may be suitable for different scenarios. The multi-scale analysis is generally faster and doesn't require training, making it good for quick anomaly detection. The machine learning approach can potentially detect more complex anomalies but requires more computational resources and data.
To use these methods effectively, you may need to:

Adjust the parameters (number of scales, thresholds, etc.) based on your specific images and anomaly characteristics.
For the ML approach, consider training on a larger dataset of "normal" images if available.
Experiment with different preprocessing techniques (e.g., histogram equalization) to enhance anomalies before detection.
"""


"""
Multi-scale Analysis (Difference of Gaussians Pyramid):

This method uses the Difference of Gaussians (DoG) technique at multiple scales to detect anomalies.
Key components:
a) DoG Pyramid:

We create a pyramid of DoG images by subtracting Gaussian-blurred versions of the original image.
Each level of the pyramid represents a different scale.

b) build_dog_pyramid method:

Iterates through different scales.
For each scale, it creates two Gaussian-blurred versions of the image with different sigma values.
Subtracts these to create a DoG image.

c) detect_anomalies method:

Builds the DoG pyramid.
Combines responses from all scales by summing the absolute values.
Thresholds the combined response to identify anomalies.

The principle behind this approach is that anomalies will show up as strong responses in the DoG images, potentially at different scales depending on their size.

Machine Learning Approach (Autoencoder):

This method uses an autoencoder neural network to learn the "normal" patterns in the image and detect anomalies as deviations from these patterns.
Key components:
a) Autoencoder Architecture:

Encoder: Compresses the input image into a lower-dimensional representation.
Decoder: Attempts to reconstruct the original image from this representation.

b) Training Process:

Extracts patches from the input image.
Trains the autoencoder to reconstruct these patches.
The assumption is that "normal" patterns will be well-reconstructed, while anomalies will not.

c) Anomaly Detection:

After training, we run new image patches through the autoencoder.
Calculate reconstruction error for each patch.
High reconstruction error indicates potential anomalies.

d) extract_patches method:

Splits the image into overlapping patches for processing.

e) train method:

Prepares the data (normalizes, creates DataLoader).
Trains the autoencoder using Mean Squared Error loss.

f) detect_anomalies method:

Runs patches through the trained autoencoder.
Calculates reconstruction errors and normalizes them.
Creates an anomaly map by assembling the patch-wise anomaly scores.

The principle here is that the autoencoder will learn to reconstruct "normal" patterns well, but will struggle with anomalies, resulting in higher reconstruction errors for anomalous regions.
"""