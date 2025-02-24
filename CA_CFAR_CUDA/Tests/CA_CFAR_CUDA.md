# Overview of CUDA Kernels for Image Processing

This document provides an overview of the two CUDA kernels implemented for image processing: `convolve_cube` and `threshold_and_detect`. The kernels work together to perform convolution operations on an image and apply threshold-based detection using statistical measures.

## 1. `convolve_cube` Kernel

### Functionality
The `convolve_cube` kernel performs a convolution operation across a set of kernels (filter matrices) applied to an input image. The input image consists of both the original image and a precomputed squared image. For each pixel, the kernel calculates two results: an average (`out_avg`) and a square average (`out_sqr`). These results are then normalized by the sum of the kernel values.

### Key Operations:
- **Kernel Initialization:**  
  The kernel is designed to handle multiple filters (N kernels). Each block processes a specific kernel (n) and a pixel location (r, c).

- **Border Replication:**  
  To avoid out-of-bound errors during convolution, the kernel uses border replication for regions near the image border. This means the pixel values at the borders are replicated.

- **Accumulation:**  
  For each kernel, the image values (both the original and squared images) are multiplied by the corresponding kernel values and accumulated.

- **Normalization:**  
  After accumulating the results, they are normalized by the sum of the kernel values to compute the average and square average for each pixel in the image.

---

## 2. `threshold_and_detect` Kernel

### Functionality
The `threshold_and_detect` kernel performs thresholding and detection based on the results from the `convolve_cube` kernel (i.e., average and square average). For each pixel, the variance and standard deviation are calculated, followed by a threshold computation. If the pixel value exceeds the computed threshold, the pixel is marked as a "detection."

### Key Operations:
- **Variance and Standard Deviation:**  
  For each pixel, the variance is calculated using the formula:  
  `variance = sqr_val - avg_val^2`.  
  The standard deviation is derived as:  
  `std_val = sqrt(variance)`.

- **Thresholding:**  
  A threshold for detection is computed using the formula:  
  `cell_thr = avg_val + thr * std_val`,  
  where `thr` is a predefined multiplier for the standard deviation.

- **Detection Map:**  
  If the pixel value from the original image exceeds the computed threshold, a detection is triggered, and the corresponding entry in the detection map is set to `true`.

---

## Relationship Between the Kernels
The `convolve_cube` kernel computes statistical properties (mean and variance) of the image using multiple kernels, while the `threshold_and_detect` kernel uses these statistics to perform a detection task based on thresholding. Together, these kernels allow for efficient image processing and detection in parallel.

### Workflow:
1. The `convolve_cube` kernel calculates the average and square average for each pixel.
2. The `threshold_and_detect` kernel processes these results to compute variances and standard deviations, followed by thresholding.
3. If the pixel's value exceeds the threshold, it is marked as a detection in the detection map.

These two kernels are designed to be executed in parallel, leveraging the power of CUDA for high-performance image processing.
