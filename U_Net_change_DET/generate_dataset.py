import os
import cv2
import numpy as np
import argparse
from tqdm import tqdm


def generate_dataset(image_path, output_dir, num_samples=100):
    """
    Generates a dataset by warping and reverse warping images to introduce noise.

    Parameters:
        image_path (str): Path to the base image.
        output_dir (str): Directory to save the dataset.
        num_samples (int): Number of warped images to generate.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/images", exist_ok=True)
    os.makedirs(f"{output_dir}/warped", exist_ok=True)
    os.makedirs(f"{output_dir}/masks", exist_ok=True)

    # Load base image
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    height, width, _ = image.shape

    for i in tqdm(range(num_samples), desc="Generating dataset"):
        try:
            # Generate random affine transformation matrix
            margin = 10  # Ensuring transformations stay within a reasonable limit
            src_pts = np.array([[margin, margin], [width - margin, margin], [margin, height - margin]],
                               dtype=np.float32)
            dst_pts = src_pts + np.random.uniform(-5, 5, size=src_pts.shape).astype(np.float32)  # Small perturbations

            transform_matrix = cv2.getAffineTransform(src_pts, dst_pts)

            # Apply transformation
            warped_image = cv2.warpAffine(image, transform_matrix, (width, height), flags=cv2.INTER_LINEAR)

            # Inverse transformation
            inverse_matrix = cv2.invertAffineTransform(transform_matrix)
            recovered_image = cv2.warpAffine(warped_image, inverse_matrix, (width, height), flags=cv2.INTER_LINEAR)

            # Compute difference mask (absolute difference)
            difference_mask = cv2.absdiff(image, recovered_image)
            difference_mask = cv2.cvtColor(difference_mask, cv2.COLOR_BGR2GRAY)
            _, binary_mask = cv2.threshold(difference_mask, 25, 255, cv2.THRESH_BINARY)

            # Introduce real changes randomly (simulate real differences)
            if np.random.rand() > 0.7:  # 30% probability of real change
                x, y = np.random.randint(0, width - 20), np.random.randint(0, height - 20)
                cv2.rectangle(recovered_image, (x, y), (x + 20, y + 20), (255, 255, 255), -1)  # Add a white patch
                cv2.rectangle(binary_mask, (x, y), (x + 20, y + 20), 255, -1)  # Mark as real change

            # Save images
            cv2.imwrite(f"{output_dir}/images/{i}.png", image)
            cv2.imwrite(f"{output_dir}/warped/{i}.png", recovered_image)
            cv2.imwrite(f"{output_dir}/masks/{i}.png", binary_mask)

        except Exception as e:
            print(f"Error processing sample {i}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to the input image")
    parser.add_argument("--output", type=str, required=True, help="Output directory for the dataset")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of samples to generate")

    args = parser.parse_args()
    generate_dataset(args.image, args.output, args.num_samples)
