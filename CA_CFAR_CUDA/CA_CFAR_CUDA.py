# Convolution - 3D:

import cupy as cp
from cupy import RawKernel

# Note: The code below uses 3D grid logic:
#   blockIdx.z = n (the kernel index)
#   (blockIdx.y, threadIdx.y) = row
#   (blockIdx.x, threadIdx.x) = column

# CUDA kernel to convolve the image and squared image with all filters at once:
cube_convolution_code = r'''
extern "C" __global__
void convolve_cube(
    const float* __restrict__ img,        // shape: (H, W)
    const float* __restrict__ img_sq,     // shape: (H, W)
    const float* __restrict__ kernels,    // shape: (N, K, K)
    float* __restrict__ out_avg,          // shape: (N, H, W)
    float* __restrict__ out_sqr,          // shape: (N, H, W)
    const int H,                          // image height
    const int W,                          // image width
    const int K,                          // kernel width (assume KxK)
    const int N,                          // number of kernels
    const float* __restrict__ kernel_sums // array: sum of kernels entries
)
{
    // Identify which kernel (n), which row (r), and which col (c) this thread handles
    int n = blockIdx.z;  // each kernel is handled in block dimension z
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;

    // Ensure we are within valid range
    if (n >= N || r >= H || c >= W) {
        return;
    }

    // Half-size for "replicate border" indexing
    // E.g. if K=3, halfK = 1; if K=5, halfK=2, etc.
    int halfK = K / 2; 

    float accum = 0.0f;      // for the average image
    float accum_sq = 0.0f;   // for the square-average image

    // Loop over the kernel
    for (int kr = 0; kr < K; kr++) {
        for (int kc = 0; kc < K; kc++) {

            // Compute the input image coordinates with border replication
            int rr = r + (kr - halfK);
            int cc = c + (kc - halfK);

            // Replicate border
            if (rr < 0)   rr = 0;
            if (rr >= H)  rr = H - 1;
            if (cc < 0)   cc = 0;
            if (cc >= W)  cc = W - 1;

            // Fetch image, squared image, and the (n-th) kernel value
            float val     = img[rr * W + cc];
            float val_sq  = img_sq[rr * W + cc];
            float k_val   = kernels[n * K * K + kr * K + kc];

            // Accumulate
            accum    += val    * k_val;
            accum_sq += val_sq * k_val;
        }
    }

    // Write out the normalized results
    // out_avg, out_sqr both have shape (N,H,W)
    int out_idx = n * (H * W) + r * W + c;
    float sum_n = kernel_sums[n];
    out_avg[out_idx] = accum / sum_n;
    out_sqr[out_idx] = accum_sq / sum_n;
}
'''

# CUDA kernel to compute var, std_img, cell_thr, and detection_map for the cube of images:
var_thresholding_code = r'''
extern "C" __global__
void threshold_and_detect(
    const float* __restrict__ image,     // shape: (H, W)
    const float* __restrict__ avg_cube,  // shape: (N, H, W)
    const float* __restrict__ sqr_cube,  // shape: (N, H, W)
    float thr,
    bool* __restrict__ detection_map,    // shape: (H, W)
    int H,
    int W,
    int N
)
{
    // Row index, col index
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;

    // Make sure we are in range
    if (r >= H || c >= W) {
        return;
    }

    // We want to do a bitwise OR of detection across all N kernels
    bool final_detection = false;

    // Read the original image pixel
    float pixel_val = image[r * W + c];

    // For each kernel result n, compute detection
    for (int n = 0; n < N; ++n) {
        // 1. Load average and square-average
        float avg_val = avg_cube[n * H * W + r * W + c];
        float sqr_val = sqr_cube[n * H * W + r * W + c];

        // 2. variance = sqr_val - avg_val^2
        float var_val = sqr_val - (avg_val * avg_val);

        // 3. standard deviation
        float std_val = sqrtf(var_val);

        // 4. threshold
        float cell_thr = avg_val + thr * std_val;

        // 5. detection: (image > cell_thr)
        if (pixel_val > cell_thr) {
            final_detection = true;
        }
    }

    // Write the final OR (boolean) to the detection_map
    detection_map[r * W + c] = final_detection;
}
'''


# Compile the CUDA kernels
cube_convolution_kernel = RawKernel(cube_convolution_code, "convolve_cube")
threshold_and_detect_kernel = RawKernel(var_thresholding_code, "threshold_and_detect")


def convolve_image_cube(
        img: cp.ndarray,  # shape: (H, W)
        kernels: cp.ndarray,  # shape: (N, K, K)
        kernel_sums: cp.ndarray,  # shape: (N)
        out_avg: cp.ndarray,  # shape: (N, H, W)
        out_sqr: cp.ndarray,  # shape: (N, H, W)
        block=(16, 16)  # threads per block (x=col, y=row)
):
    """
    Convolve the input 2D image (and its square) with each of N kernels,
    producing N average-images and N square-average-images.
    Returns (out_avg, out_sqr) each shape (N,H,W).

    :param img: 2D image (H,W)
    :param kernels: 3D kernels (N,K,K)
    :param kernel_sums: kernels sums (N) - kernel_sums[i] holds the sum of all elements of kernels[i]
    :param out_avg: 3D average image (N,H,W)
    :param out_sqr: 3D square-average image (N,H,W)
    :param block: threads per block (x=col, y=row)
    :return: out_avg, out_sqr - average image and square-average image of shape (N,H,W)
    """

    H, W = img.shape
    N, K, _ = kernels.shape

    # Convert arrays to float32
    img_f = img.astype(cp.float32, copy=False)
    kernels_f = kernels.astype(cp.float32, copy=False)
    kernel_sums = kernel_sums.astype(cp.float32, copy=False)

    # Make the squared image
    img_sq_f = cp.power(img_f, 2)

    # Grid size:
    #   - x-dim: (W // block.x) + 1 if not multiple
    #   - y-dim: (H // block.y) + 1 if not multiple
    #   - z-dim: N
    grid_x = (W + block[0] - 1) // block[0]
    grid_y = (H + block[1] - 1) // block[1]
    grid_z = N

    grid = (grid_x, grid_y, grid_z)

    # Launch the kernel
    # kernel<<<grid, block>>>(...)
    cube_convolution_kernel(
        grid,
        block,
        (
            img_f,
            img_sq_f,
            kernels_f,
            out_avg,
            out_sqr,
            cp.int32(H),
            cp.int32(W),
            cp.int32(K),
            cp.int32(N),
            kernel_sums
        )
    )

    return out_avg, out_sqr


# Function to compute stats (var, std_img, cell_thr, and detection_map)
def threshold_and_detect(
    image: cp.ndarray,          # shape (H, W), float32
    avg_cube: cp.ndarray,       # shape (N, H, W), float32
    sqr_cube: cp.ndarray,       # shape (N, H, W), float32
    detection_map: cp.ndarray,  # shape (H, W), bool
    thr: float,
    block=(16, 16)
):
    """
    For each pixel (r,c), loop over all N kernels results and perform:
        var = sqr - avg^2
        std = sqrt(var)
        cell_thr = avg + thr * std
        detection_bit = (image[r,c] > cell_thr)
    Then perform OR all detection bits together (over n=0..N-1).
    Returns a boolean detection_map of shape (H,W).

    :param image: 2D image (H,W)
    :param avg_cube: 3D average image (N,H,W) - results of image convolution with N filters
    :param sqr_cube: 3D square-average image (N,H,W) - results of square image convolution with N filters
    :param detection_map: will hold the results of the thresholding
    :param thr: threshold value
    :param block: threads per block (x=col, y=row)
    :return: detection_map
    """
    # Basic shape checks
    H, W = image.shape
    N, H2, W2 = avg_cube.shape
    assert H == H2 and W == W2, "avg_cube shape must match image shape in last two dims"
    assert sqr_cube.shape == (N, H, W), "sqr_cube shape must match (N,H,W)"

    # Grid
    grid_x = (W + block[0] - 1) // block[0]
    grid_y = (H + block[1] - 1) // block[1]
    grid = (grid_x, grid_y, 1)

    # Ensure float32
    image_f    = image.astype(cp.float32, copy=False)
    avg_cube_f = avg_cube.astype(cp.float32, copy=False)
    sqr_cube_f = sqr_cube.astype(cp.float32, copy=False)

    threshold_and_detect_kernel(
        grid,
        block,
        (
            image_f,
            avg_cube_f,
            sqr_cube_f,
            cp.float32(thr),
            detection_map,
            cp.int32(H),
            cp.int32(W),
            cp.int32(N)
        )
    )

    return detection_map


# Example usage
if __name__ == "__main__":
    # Example input image and computed avg_image, sqr_pixel_mean
    thr = 1.5  # Example threshold
    H, W = 512, 512
    image_cp = cp.random.randn(H, W).astype(cp.float32)

    # Suppose we have 5 different kernels (N=5), each 3x3
    N, K = 5, 3
    kernels_cp = cp.random.randn(N, K, K).astype(cp.float32)
    kernels_sum = cp.sum(kernels_cp, axis=0)

    # Create output arrays
    out_avg = cp.zeros((N, H, W), dtype=cp.float32)
    out_sqr = cp.zeros((N, H, W), dtype=cp.float32)
    detection_map = cp.zeros((H, W), dtype=cp.bool_)

    # Convolve
    avg_cube, sqr_cube = convolve_image_cube(image_cp, kernels_cp, kernels_sum, out_avg, out_sqr, block=(16, 16))

    print("avg_cube shape:", avg_cube.shape)  # (5, H, W)
    print("sqr_cube shape:", sqr_cube.shape)  # (5, H, W)

    # Get the results back to CPU
    avg_image = avg_cube.get()
    sqr_pixel_mean = sqr_cube.get()

    # Compute stats (var, std_img, cell_thr, detection_map)
    # We already have:
    #   image_cp: shape (H,W), e.g. from cp.random.rand(...)
    #   avg_cube, sqr_cube: shape (N,H,W), from prior convolution steps

    detection_map = threshold_and_detect(
        image_cp,
        avg_cube,
        sqr_cube,
        detection_map,
        thr=thr,
        block=(16, 16)
    )

    print("detection_map shape:", detection_map.shape)   # (H, W)
    print("detection_map dtype:", detection_map.dtype)   # bool
    print("Example 5x5 corner of detection_map:\n", detection_map[:5,:5].get())
