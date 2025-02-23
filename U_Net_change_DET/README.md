# Image Difference Detection with U-Net

This repository provides a **U-Net-based model** to detect **real differences** between two images while ignoring warping-induced noise. It includes:

- **Dataset generation** (warping images, adding real differences).
- **Training a U-Net model** for difference detection.
- **Inference on new images**.

## 🔧 Installation

Ensure you have Python 3.7+ installed. Install the required dependencies:

```bash
pip install torch torchvision opencv-python numpy pillow tqdm matplotlib
```

📜 Dataset Generation
To generate a dataset, use generate_dataset.py. This script:

Takes an input image.
Warps it using random transformations.
Reverse warps it to induce noise.
Introduces real differences (e.g., added objects).
Saves original, warped, and mask images.

Usage
```bash
python generate_dataset.py --image sample.jpg --output dataset --num_samples 100
```

Dataset Structure
The generated dataset will be saved in:

dataset/
├── images/       # Original images
├── warped/       # Warped images with noise
├── masks/        # Ground truth masks (real differences)

🏋️ Training the U-Net Model
To train the model on the generated dataset, run train_unet.py.

```bash
python train_unet.py
```

The model will be saved as:

unet_image_difference.pth

Training Details
Uses Binary Cross-Entropy (BCE) Loss for segmentation.
Input: (original + warped) image pair.
Output: Mask showing real differences.

🔍 Inference
To test the trained model on new images, use inference.py.

```bash
python inference.py --image original.png --warped warped.png --model unet_image_difference.pth
```

Expected Output
The model will highlight real differences while ignoring noise.
A heatmap of detected changes will be displayed.

📌 Notes
Ensure dataset directories (images/, warped/, masks/) exist before training.
The model ignores warping noise but detects actual differences.

Use larger datasets for better accuracy.