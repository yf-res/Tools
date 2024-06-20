import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the images
img1 = cv2.imread('image1.png', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('image2.png', cv2.IMREAD_GRAYSCALE)

# Calculate the optical flow using Farneback method
flow = cv2.calcOpticalFlowFarneback(img1, img2, None, 0.5, 3, 15, 3, 5, 1.2, 0)

# Function to warp an image based on optical flow
def warp_flow(img, flow):
    h, w = img.shape[:2]
    flow_map = np.column_stack(np.meshgrid(np.arange(w), np.arange(h)))
    new_map = (flow_map + flow).astype(np.float32)
    return cv2.remap(img, new_map, None, cv2.INTER_LINEAR)

# Generate the interpolated images
interpolation_factor_0_25 = warp_flow(img1, flow * 0.25)
interpolation_factor_0_75 = warp_flow(img1, flow * 0.75)

# Convert images to RGB for displaying
img1_rgb = cv2.cvtColor(img1, cv2.COLOR_GRAY2RGB)
img2_rgb = cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB)
interp_0_25_rgb = cv2.cvtColor(interpolation_factor_0_25, cv2.COLOR_GRAY2RGB)
interp_0_75_rgb = cv2.cvtColor(interpolation_factor_0_75, cv2.COLOR_GRAY2RGB)

# Display the results
plt.figure(figsize=(10, 5))
plt.subplot(1, 4, 1), plt.imshow(img1_rgb), plt.title('Image 1')
plt.subplot(1, 4, 2), plt.imshow(img2_rgb), plt.title('Image 2')
plt.subplot(1, 4, 3), plt.imshow(interp_0_25_rgb), plt.title('Interpolation 0.25')
plt.subplot(1, 4, 4), plt.imshow(interp_0_75_rgb), plt.title('Interpolation 0.75')
plt.show()

# Save the interpolated images
cv2.imwrite('interpolated_0_25.png', interpolation_factor_0_25)
cv2.imwrite('interpolated_0_75.png', interpolation_factor_0_75)
