import cv2
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import gradio as gr
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')
plt.ion()

# Global variables
original_image = None
original_template = None
modified_template = None

# Load pre-trained ResNet model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet34(weights=None)  # pretrained=True
model.load_state_dict(torch.load("pytorch_resnet34.pth"))
model.to(device)
model.eval()

# Remove the last fully-connected layer and adaptive average pooling
model = torch.nn.Sequential(*list(model.children())[:-2])  # can be also -1

# Image transformation
# transform = transforms.Compose([
#     transforms.Resize(256),
#     transforms.CenterCrop(224),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def crop_template(image):
    global original_image
    original_image = image.copy()
    roi = cv2.selectROI("Select Template", image)
    cv2.destroyAllWindows()
    template = image[int(roi[1]):int(roi[1] + roi[3]), int(roi[0]):int(roi[0] + roi[2])]
    return template


def modify_template(scale, rotation, noise, warp):
    global original_image, original_template

    if original_template is None:
        return None

    # Scale
    height, width = original_template.shape[:2]
    new_size = (int(width * scale), int(height * scale))
    modified = cv2.resize(original_template, new_size)

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


def extract_features_cpu(img):
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    img = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model(img)
    return features.squeeze().cpu().numpy()


def extract_features_gpu(img):
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    img = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model(img)
    return features.squeeze()


def extract_features_gpu_full(img):
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    img = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model(img)
    return features


def find_template(method):
    global original_image, original_template, modified_template
    if original_image is None or modified_template is None:
        return None

    if method == "Template Matching":
        result = cv2.matchTemplate(original_image, modified_template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        top_left = max_loc
        bottom_right = (top_left[0] + modified_template.shape[1], top_left[1] + modified_template.shape[0])
    elif method == "Template Matching (GPU)":
        # Convert images to grayscale
        gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        gray_template = cv2.cvtColor(modified_template, cv2.COLOR_BGR2GRAY)

        # Upload images to GPU
        gpu_image = cv2.cuda_GpuMat()
        gpu_template = cv2.cuda_GpuMat()
        gpu_result = cv2.cuda_GpuMat()

        gpu_image.upload(gray_image)
        gpu_template.upload(gray_template)

        # Perform template matching on GPU
        cv2.cuda.templateMatching(gpu_image, gpu_template, cv2.TM_CCOEFF_NORMED, gpu_result)

        # Download result from GPU
        result = gpu_result.download()

        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        top_left = max_loc
        bottom_right = (top_left[0] + modified_template.shape[1], top_left[1] + modified_template.shape[0])

    elif method == "Feature Matching":
        orb = cv2.ORB_create()
        kp1, des1 = orb.detectAndCompute(modified_template, None)
        kp2, des2 = orb.detectAndCompute(original_image, None)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        h, w = modified_template.shape[:2]
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)
        top_left = tuple(map(int, dst[0][0]))
        bottom_right = tuple(map(int, dst[2][0]))

    elif method == "CNN Feature Matching":
        template_features = extract_features_cpu(modified_template)

        best_score = -np.inf
        best_loc = None

        for y in range(0, original_image.shape[0] - modified_template.shape[0], 20):
            for x in range(0, original_image.shape[1] - modified_template.shape[1], 20):
                roi = original_image[y:y + modified_template.shape[0], x:x + modified_template.shape[1]]
                roi_features = extract_features_cpu(roi)
                score = np.dot(template_features.flatten(), roi_features.flatten()) / (
                        np.linalg.norm(template_features.flatten()) * np.linalg.norm(roi_features.flatten()))
                if score > best_score:
                    best_score = score
                    best_loc = (x, y)

        top_left = best_loc
        bottom_right = (top_left[0] + modified_template.shape[1], top_left[1] + modified_template.shape[0])
    elif method == "CNN Full image Feature Matching":
        """
        Key improvements in this version:

        We've modified the ResNet model to remove the last fully-connected layer and the adaptive average pooling layer. This allows us to get a feature map output instead of a single vector.
        The extract_features function now returns the feature maps directly, without any pooling or flattening.
        In the "CNN Feature Matching" method:

        We extract features from both the template and the entire image in one pass each.
        We use torch.nn.functional.conv2d to compute the normalized cross-correlation between the image features and the template features. This operation is equivalent to sliding the template features over the image features and computing the similarity at each position.
        We find the location of the maximum correlation, which corresponds to the best match position.
        We convert the feature map coordinates back to image coordinates using a scale factor.



        This approach is much faster because:

        It processes the entire image in one forward pass through the network, instead of processing each crop separately.
        It uses GPU-accelerated convolution operations to compute the similarity between the template and all possible positions in the image simultaneously.

        The speed improvement should be substantial, especially for larger images. The exact speedup will depend on your GPU and the size of the images, but it could easily be orders of magnitude faster than the previous implementation.
        Remember to adjust the Gradio interface and other parts of the code as necessary to accommodate these changes. This optimized CNN Feature Matching method should now be much more competitive in terms of speed with the other matching methods.
        """
        # Extract features from the template
        template_features = extract_features_gpu_full(modified_template)

        # Extract features from the entire image
        image_features = extract_features_gpu_full(original_image)

        # Compute normalized cross-correlation
        correlation = torch.nn.functional.conv2d(
            image_features,
            template_features,
            padding='valid'
        )

        # Find the location of the maximum correlation
        _, _, max_h, max_w = correlation.shape
        max_loc = torch.argmax(correlation.view(max_h * max_w))
        max_y, max_x = max_loc // max_w, max_loc % max_w

        # Convert back to image coordinates
        scale_factor = original_image.shape[0] / image_features.shape[2]
        top_left = (int(max_x.item() * scale_factor), int(max_y.item() * scale_factor))
        bottom_right = (top_left[0] + modified_template.shape[1], top_left[1] + modified_template.shape[0])

    elif method == "CNN Feature Matching Fast":
        """
        Key improvements in this version:

        We create a tensor of all possible crops from the original image. We use a step size of 20 pixels to reduce the number of crops and speed up processing, but you can adjust this value based on your needs.
        We process all crops in a single forward pass through the model, which is much faster than processing each crop individually.
        We use PyTorch's cosine_similarity function to compute the similarity between the template features and all crop features in one operation.
        We use torch.max to find the best matching crop efficiently.

        This approach leverages GPU parallelism and PyTorch's efficient tensor operations, which should result in a significant speed improvement over the previous implementation, especially for large images.
        Note that this method still requires a considerable amount of memory, as it creates a tensor of all possible crops. If you run into memory issues with very large images, you may need to process the image in batches or further increase the step size.
        You can adjust the trade-off between speed and accuracy by modifying the step size in the crop creation loop. A larger step size will be faster but might miss the optimal match, while a smaller step size will be more accurate but slower.
        This implementation should provide a good balance between speed and accuracy for the CNN Feature Matching method.
        """
        # Extract features from the template
        template_features = extract_features_gpu(modified_template)

        # Prepare the original image for feature extraction
        h, w = original_image.shape[:2]
        th, tw = modified_template.shape[:2]

        # Create a tensor of all possible crops
        crops = []
        for y in range(0, h - th + 1, 20):  # Step size of 20 for faster processing
            for x in range(0, w - tw + 1, 20):
                crop = original_image[y:y + th, x:x + tw]
                crops.append(transform(Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))))

        crops_tensor = torch.stack(crops).to(device)

        # Extract features from all crops in a single forward pass
        with torch.no_grad():
            crop_features = model(crops_tensor).squeeze()

        # Compute similarity scores
        similarity_scores = torch.nn.functional.cosine_similarity(crop_features, template_features.unsqueeze(0))

        # Find the best match
        best_score, best_idx = torch.max(similarity_scores, dim=0)
        best_y, best_x = divmod(best_idx.item() * 20, w - tw + 1)

        top_left = (best_x, best_y)
        bottom_right = (top_left[0] + tw, top_left[1] + th)

    result_img = original_image.copy()
    cv2.rectangle(result_img, top_left, bottom_right, (0, 0, 255), 2)

    # Zoom into the found region
    padding = 200
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
def process(image, template, scale, rotation, noise, warp, method):
    # template = crop_template(image)
    global original_template, original_image, modified_template
    original_image = image.copy()
    original_template = template.copy()
    modified_template = modify_template(scale, rotation, noise, warp)
    result = find_template(method)
    return modified_template, result


iface = gr.Interface(
    fn=process,
    inputs=[
        gr.Image(type="numpy", label="Upload Image"),
        gr.Image(type="numpy", label="Upload Template"),
        gr.Slider(0.5, 2.0, 1.0, step=0.1, label="Scale"),
        gr.Slider(-180, 180, 0.0, step=1, label="Rotation"),
        gr.Slider(0, 50, step=1, label="Noise"),
        gr.Slider(0, 0.5, step=0.01, label="Warp"),
        gr.Radio(["Template Matching", "Template Matching (GPU)", "Feature Matching", "CNN Feature Matching",
                  "CNN Full image Feature Matching", "CNN Feature Matching Fast"], label="Method")
    ],
    outputs=[
        gr.Image(type="numpy", label="Modified Template"),
        gr.Plot(label="Result")
    ],
    title="Template Matching with Modifications",
    description="Upload an image and a template to search in it."
                "Apply modifications, and find it in the original image."
)

iface.launch()
