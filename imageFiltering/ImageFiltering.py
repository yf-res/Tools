import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom


# Create a sample 1D signal
original_signal = np.array([0, 1, 2, 3, 4, 5, 6, 7])

# Down-sample by a factor of 2
downsampled_signal = zoom(original_signal, 0.5, order=1)

# Up-sample back to original size by a factor of 2
upsampled_signal = zoom(downsampled_signal, 2, order=1)

# Plot the signals
plt.figure(figsize=(12, 6))

plt.subplot(1, 3, 1)
plt.stem(original_signal, use_line_collection=True)
plt.title('Original Signal')
plt.grid(True)

plt.subplot(1, 3, 2)
plt.stem(downsampled_signal, use_line_collection=True)
plt.title('Down-sampled Signal')
plt.grid(True)

plt.subplot(1, 3, 3)
plt.stem(upsampled_signal, use_line_collection=True)
plt.title('Up-sampled Signal')
plt.grid(True)

plt.tight_layout()
plt.show()

# Show the shift
print("Original Signal: ", original_signal)
print("Down-sampled Signal: ", downsampled_signal)
print("Up-sampled Signal: ", upsampled_signal)


# Create a simple example image (a grid with different colors)
original_image = np.tile(np.arange(16).reshape(4, 4), (2, 2))

# Down-sample by a factor of 2
downsampled_image = zoom(original_image, 0.5, order=1)

# Up-sample back to original size by a factor of 2
upsampled_image = zoom(downsampled_image, 2, order=1)

# Plot the images
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(original_image, cmap='gray', interpolation='nearest')
axes[0].set_title('Original Image')
axes[0].axis('off')

axes[1].imshow(downsampled_image, cmap='gray', interpolation='nearest')
axes[1].set_title('Down-sampled Image')
axes[1].axis('off')

axes[2].imshow(upsampled_image, cmap='gray', interpolation='nearest')
axes[2].set_title('Up-sampled Image')
axes[2].axis('off')

plt.tight_layout()
plt.show()

# Show the pixel values
print("Original Image:\n", original_image)
print("Down-sampled Image:\n", downsampled_image)
print("Up-sampled Image:\n", upsampled_image)

import cv2
import numpy as np
import matplotlib.pyplot as plt


def show_half_pixel_shift(image_path):
    # Load image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Downsample the image by a factor of 2
    downsampled_image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)

    # Upsample the image back by a factor of 2
    upsampled_image = cv2.resize(downsampled_image, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)

    # Plot original, downsampled, and upsampled images
    plt.figure(figsize=(12, 8))

    plt.subplot(1, 3, 1)
    plt.title('Original Image')
    plt.imshow(image, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title('Downsampled Image')
    plt.imshow(downsampled_image, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title('Upsampled Image')
    plt.imshow(upsampled_image, cmap='gray')
    plt.axis('off')

    plt.show()

    # Calculate and print pixel value shifts
    original_center_pixel = image[image.shape[0] // 2, image.shape[1] // 2]
    upsampled_center_pixel = upsampled_image[image.shape[0] // 2, image.shape[1] // 2]

    print(f'Original center pixel value: {original_center_pixel}')
    print(f'Upsampled center pixel value: {upsampled_center_pixel}')

    # Difference image
    difference_image = cv2.absdiff(image, upsampled_image)
    plt.figure(figsize=(6, 6))
    plt.title('Difference Image')
    plt.imshow(difference_image, cmap='gray')
    plt.axis('off')
    plt.show()


# Example usage
show_half_pixel_shift('../Misc_14.png')

