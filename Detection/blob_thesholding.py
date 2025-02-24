import cv2
import numpy as np
from scipy.ndimage import label, center_of_mass
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Define a bi-cubic polynomial function for fitting
def bicubic(x, y, coeffs):
    """Returns the value of a bicubic polynomial for given x and y."""
    a, b, c, d, e, f, g, h, i = coeffs
    return (a*x**3 + b*x**2*y + c*x*y**2 + d*y**3 + e*x**2 + f*x*y + g*y**2 + h*x + i)

# Helper function to fit a bicubic polynomial to the region around the center of the blob
def fit_bicubic(blob, initial_guess):
    """Fit a bicubic polynomial to the image region around the blob's center."""
    # Get the coordinates of the blob
    y_indices, x_indices = np.where(blob > 0)  # blob > 0 means foreground pixels
    x_data = x_indices
    y_data = y_indices
    z_data = blob[x_indices, y_indices]

    # Fit a bicubic polynomial to the data
    params, _ = curve_fit(lambda p, x, y: bicubic(x, y, p), 
                          (x_data, y_data), 
                          z_data, 
                          p0=initial_guess)
    return params

# Function to find the best threshold based on blob size
def find_best_threshold(image, min_blob_size, max_blob_size, step=5):
    best_threshold = None
    best_center = None
    best_size = None

    # Iterate over possible thresholds to find the best one
    for threshold in np.arange(0, 255, step):
        _, thresholded = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)

        # Label connected components
        labeled, num_labels = label(thresholded)
        
        for label_id in range(1, num_labels + 1):
            # Get the blob region
            blob = (labeled == label_id).astype(np.uint8)
            
            # Find the center of mass
            center = center_of_mass(blob)
            
            if center == (0.0, 0.0):  # Skip empty blobs
                continue
            
            # Calculate the blob size
            blob_size = np.sum(blob)
            
            # Check if the blob size is within the desired range
            if min_blob_size <= blob_size <= max_blob_size:
                # Fit a bicubic polynomial to get subpixel accuracy for center of mass
                initial_guess = np.zeros(9)  # Initial guess for bicubic polynomial coefficients
                params = fit_bicubic(blob, initial_guess)
                
                # Find the new center of gravity using the fit (you can use the coefficients to find the center)
                # We might approximate the center by the max point of the bicubic fit
                x_center = np.argmax([bicubic(x, center[1], params) for x in range(blob.shape[1])])
                y_center = np.argmax([bicubic(center[0], y, params) for y in range(blob.shape[0])])
                
                if best_center is None or blob_size > best_size:  # We keep the largest valid blob
                    best_center = (x_center, y_center)
                    best_size = blob_size
                    best_threshold = threshold

    return best_threshold, best_center, best_size

def main(image_path, min_blob_size=500, max_blob_size=5000):
    # Load the image (assuming a grayscale image)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("Error: Image could not be loaded.")
        return

    # Normalize the image to [0, 255]
    image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)

    # Find the best threshold
    best_threshold, best_center, best_size = find_best_threshold(
        image, min_blob_size, max_blob_size)

    print(f"Best Threshold: {best_threshold}")
    print(f"Best Blob Center: {best_center}")
    print(f"Best Blob Size: {best_size}")

    # Plot the original image and the thresholded image
    _, thresholded = cv2.threshold(image, best_threshold, 255, cv2.THRESH_BINARY)
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')
    
    plt.subplot(1, 2, 2)
    plt.imshow(thresholded, cmap='gray')
    plt.title(f'Thresholded Image (Threshold={best_threshold})')
    
    plt.show()

if __name__ == "__main__":
    image_path = "path_to_your_image.jpg"  # Replace with your image path
    main(image_path)
