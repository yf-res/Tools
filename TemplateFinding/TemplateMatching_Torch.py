import cv2
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import gradio as gr
import matplotlib.pyplot as plt

# Global variables
original_image = None
template = None

# Load pre-trained ResNet model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet34(pretrained=True).to(device)
model.eval()

# Remove the last fully-connected layer
model = torch.nn.Sequential(*list(model.children())[:-1])

# Image transformation
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def crop_template(image):
    global original_image, template
    original_image = image.copy()
    roi = cv2.selectROI("Select Template", image)
    cv2.destroyAllWindows()
    template = image[int(roi[1]):int(roi[1] + roi[3]), int(roi[0]):int(roi[0] + roi[2])]
    return template


def modify_template(scale, rotation, noise, warp):
    global template
    if template is None:
        return None

    # Scale
    height, width = template.shape[:2]
    new_size = (int(width * scale), int(height * scale))
    modified = cv2.resize(template, new_size)

    # Rotate
    matrix = cv2.getRotationMatrix2D((new_size[0] // 2, new_size[1] // 2), rotation, 1)
    modified = cv2.warpAffine(modified, matrix, new_size)

    # Add noise
    noise_img = np.random.normal(0, noise, modified.shape).astype(np.uint8)
    modified = cv2.add(modified, noise_img)

    # Warp
    rows, cols = modified.shape[:2]
    src_points = np.float32([[0, 0], [cols - 1, 0], [0, rows - 1], [cols - 1, rows - 1]])
    dst_points = np.float32(
        [[0, 0], [cols - 1, 0], [int(warp * cols), rows - 1], [cols - 1 - int(warp * cols), rows - 1]])
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    modified = cv2.warpPerspective(modified, matrix, (cols, rows))

    return modified


def extract_features(img):
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    img = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model(img)
    return features.squeeze().cpu().numpy()


def find_template(method):
    global original_image, template
    if original_image is None or template is None:
        return None

    if method == "Template Matching":
        result = cv2.matchTemplate(original_image, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        top_left = max_loc
        bottom_right = (top_left[0] + template.shape[1], top_left[1] + template.shape[0])

    elif method == "Feature Matching":
        orb = cv2.ORB_create()
        kp1, des1 = orb.detectAndCompute(template, None)
        kp2, des2 = orb.detectAndCompute(original_image, None)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        h, w = template.shape[:2]
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)
        top_left = tuple(map(int, dst[0][0]))
        bottom_right = tuple(map(int, dst[2][0]))

    elif method == "CNN Feature Matching":
        template_features = extract_features(template)

        best_score = -np.inf
        best_loc = None

        for y in range(0, original_image.shape[0] - template.shape[0], 20):
            for x in range(0, original_image.shape[1] - template.shape[1], 20):
                roi = original_image[y:y + template.shape[0], x:x + template.shape[1]]
                roi_features = extract_features(roi)
                score = np.dot(template_features, roi_features) / (
                            np.linalg.norm(template_features) * np.linalg.norm(roi_features))
                if score > best_score:
                    best_score = score
                    best_loc = (x, y)

        top_left = best_loc
        bottom_right = (top_left[0] + template.shape[1], top_left[1] + template.shape[0])

    result_img = original_image.copy()
    cv2.rectangle(result_img, top_left, bottom_right, (0, 0, 255), 2)

    # Zoom into the found region
    padding = 50
    zoom_top_left = (max(0, top_left[0] - padding), max(0, top_left[1] - padding))
    zoom_bottom_right = (min(original_image.shape[1], bottom_right[0] + padding),
                         min(original_image.shape[0], bottom_right[1] + padding))
    zoomed_img = result_img[zoom_top_left[1]:zoom_bottom_right[1], zoom_top_left[0]:zoom_bottom_right[0]]

    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(zoomed_img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.tight_layout()

    return plt


# Gradio interface
def process(image, scale, rotation, noise, warp, method):
    global template
    template = crop_template(image)
    modified_template = modify_template(scale, rotation, noise, warp)
    result = find_template(method)
    return modified_template, result


iface = gr.Interface(
    fn=process,
    inputs=[
        gr.Image(type="numpy", label="Upload Image"),
        gr.Slider(0.5, 2.0, step=0.1, label="Scale"),
        gr.Slider(-180, 180, step=1, label="Rotation"),
        gr.Slider(0, 50, step=1, label="Noise"),
        gr.Slider(0, 0.5, step=0.01, label="Warp"),
        gr.Radio(["Template Matching", "Feature Matching", "CNN Feature Matching"], label="Method")
    ],
    outputs=[
        gr.Image(type="numpy", label="Modified Template"),
        gr.Plot(label="Result")
    ],
    title="Template Matching with Modifications",
    description="Upload an image, crop a template, apply modifications, and find it in the original image."
)

iface.launch()


"""
The main changes in this PyTorch version are:

We've replaced TensorFlow and Keras imports with PyTorch and torchvision imports.
Instead of VGG16, we're now using a pre-trained ResNet18 model from torchvision. We remove the last fully-connected layer to use it as a feature extractor.
We've defined a transformation pipeline using torchvision.transforms to preprocess the input images for the ResNet model.
The extract_features function now uses PyTorch's model to extract features from the images.
We've added CUDA support, so if a GPU is available, the model will use it for faster computation.

The rest of the implementation remains largely the same, including the Gradio interface and the other template matching methods.
To use this script:

Install the required libraries: pip install opencv-python numpy torch torchvision matplotlib gradio
Run the script
Open the provided URL in your web browser
Upload an image
Use the GUI to crop a template from the image
Adjust the modification parameters (scale, rotation, noise, warp)
Choose a matching method
Click "Submit" to see the results

This PyTorch implementation should provide similar functionality to the TensorFlow version, with the added benefit of GPU acceleration if available.
"""