import numpy as np
import cv2
import time
import cupy as cp
import matplotlib.pyplot as plt
import matplotlib
from Detectors.CA_CFAR_CUDA import convolve_image_cube, threshold_and_detect

matplotlib.use('TkAgg')
plt.ion()


def initialize_kernels(kernel_sizes: list, guard_size: int):
    """
    Create a cube of kernels based on the length of kernel_sizes array.

    :param kernel_sizes: a list of kernel sizes
    :param guard_size: the "guard" size - same for all kernels
    :return: a cube of shape (N, W, H) where W == H
    """
    max_size = max(kernel_sizes)
    num_kernels = len(kernel_sizes)
    kernel_sums = np.zeros(num_kernels)
    kernel_cube = np.zeros((num_kernels, max_size, max_size))
    cent = max_size // 2
    for i in range(num_kernels):
        kernel_half_size = kernel_sizes[i] // 2
        guard_half_size = (kernel_sizes[i] - guard_size) // 2
        kernel_cube[i, cent - kernel_half_size:cent + kernel_half_size + 1,
        cent - kernel_half_size:cent + kernel_half_size + 1] = 1
        kernel_cube[i, cent - guard_half_size:cent + guard_half_size + 1,
        cent - guard_half_size:cent + guard_half_size + 1] = 0
        kernel_sums[i] = np.sum(kernel_cube[i, :, :], dtype=cp.float32)
    return kernel_cube, kernel_sums


def op_ca_cfar(image, thr=7, window_size=13, guard_s=11, zeroed_kernel=-1):
    # guard_s = 5
    # window_size = 9
    cell_size = (window_size, window_size)
    guard_size = (guard_s, guard_s)
    # detection_map = np.full((image.shape[0], image.shape[1]), False)
    cent = cell_size[0] // 2
    kernel = np.ones(cell_size)
    kernel[cent - guard_size[0] // 2:cent + guard_size[0] // 2 + 1,
    cent - guard_size[1] // 2:cent + guard_size[1] // 2 + 1] = 0
    if zeroed_kernel == 1:
        # Left
        kernel[:, 0] = 0
    elif zeroed_kernel == 2:
        # Up
        kernel[0, :] = 0
    elif zeroed_kernel == 3:
        # Right
        kernel[:, -1] = 0
    elif zeroed_kernel == 4:
        # Bottom
        kernel[-1, :] = 0

    # kernel = kernel/np.nansum(kernel)
    # avg_image = convolve2d(image, kernel, mode='same') / convolve2d(~np.isnan(frame), kernel, mode='same')

    # avg_image = convolve2d(image, kernel, mode='same')/ np.sum(kernel)
    # sqr_pixel_mean = convolve2d(np.float_power(image, 2), kernel, mode='same')/ np.sum(kernel)
    # sqr_pixel_mean = convolve2d(image**2, kernel, mode='same')/ np.sum(kernel)

    avg_image = cv2.filter2D(image.astype(np.float32), -1, kernel, borderType=cv2.BORDER_REPLICATE) / np.sum(kernel)
    sqr_pixel_mean = cv2.filter2D(np.float_power(image, 2), -1, kernel, borderType=cv2.BORDER_REPLICATE) / np.sum(
        kernel)

    var = sqr_pixel_mean - np.float_power(avg_image, 2)

    std_img = np.sqrt(var)
    cell_thr = avg_image + thr * std_img
    detection_map = image > cell_thr

    return avg_image, sqr_pixel_mean, detection_map, cell_thr, std_img


# ---------------------- EXAMPLE USAGE ----------------------
if __name__ == "__main__":
    debug = False
    use_random_image = False
    H, W = 150, 150
    iterations = 11  # Number of iterations for time estimation - will be set to 1 for debug
    thr = 1.5  # threshold for detection map
    guard_size = 2  # number of pixels for filter guard
    kernel_size = [7, 15, 21, 53, 75]  # filter sizes

    if use_random_image:
        image = np.random.random((H, W)).astype(np.uint16)
    else:
        image = cv2.imread("/mnt/nfs_storage/Public/Yehuda/BlurDataSet/images/Misc_320.png", cv2.IMREAD_GRAYSCALE)
        if image.shape[0] > W or image.shape[1] > H:
            image = image[0:H, 0:W].astype(np.uint16)
            H, W = image.shape
        else:
            H, W = image.shape

    # Initializations:
    detection_map_cpu = np.zeros((H, W), dtype=np.bool_)
    kernels, kernels_sum = initialize_kernels(kernel_size, guard_size)
    N = kernel_size.__len__()
    mean_image = np.zeros((N, H, W), dtype=np.float32)
    sqr_pixel_mean = np.zeros((N, H, W), dtype=np.float32)

    if debug:
        iterations = 1

    time_start = time.time()
    for i in range(iterations):
        idx = 0
        for ks in kernel_size:
            ks_win = ks - guard_size
            mean_image[idx], sqr_pixel_mean[idx], bw, cell_thr, _ = op_ca_cfar(image, thr=thr, window_size=ks,
                                                                               guard_s=ks_win, zeroed_kernel=-1)  # 7
            detection_map_cpu = np.bitwise_or(detection_map_cpu, bw)
            idx += 1

    print(f"Original Execution Time: {(time.time() - time_start) / 10:.4f} seconds")

    if np.sum(detection_map_cpu) == 0:
        print("all zeros... Can't compare bit-exactness --> Exit()")
        exit()

    # Create output arrays
    out_avg = cp.zeros((N, H, W), dtype=cp.float32)
    out_sqr = cp.zeros((N, H, W), dtype=cp.float32)
    detection_map = cp.zeros((H, W), dtype=cp.bool_)
    detection_map_gpu = np.zeros((H, W), dtype=cp.bool_)
    for i in range(iterations):
        # opt out the initialization of the GPU:
        if i == 1:
            time_start = time.time()

        # upload to GPU
        image_cp = cp.array(image)
        kernels_cp = cp.array(kernels)
        kernels_sum_cp = cp.array(kernels_sum)

        out_avg, out_sqr = convolve_image_cube(
            image_cp,  # shape: (H, W)
            kernels_cp,  # shape: (N, K, K)
            kernels_sum_cp,  # shape: (N)
            out_avg,  # shape: (N, H, W)
            out_sqr,  # shape: (N, H, W)
            block=(16, 16)  # threads per block (x=col, y=row)
        )

        if debug:
            for k in range(N):
                diff_avg_img = mean_image[k] - out_avg[k].get()
                diff_avg_sqr_img = sqr_pixel_mean[k] - out_sqr[k].get()
                if np.sum(abs(diff_avg_img)) > 0:
                    print('no match between average images CPU/GPU')
                    exit()
                if np.sum(abs(diff_avg_sqr_img)) > 0:
                    print('no match between square average images CPU/GPU')
                    exit()

        detection_map = threshold_and_detect(
            image_cp,  # shape (H, W), float32
            out_avg,  # shape (N, H, W), float32
            out_sqr,  # shape (N, H, W), float32
            detection_map,
            thr,
            block=(16, 16)
        )
        detection_map_gpu = detection_map.get()

    print(f"GPU Execution Time: {(time.time() - time_start) / 10:.4f} seconds")

    if debug:
        plt.figure(), plt.imshow(detection_map.get(), cmap='gray'), plt.title('GPU detection results')
        plt.figure(), plt.imshow(detection_map_cpu, cmap='gray'), plt.title('CPU detection results')
        plt.show()

    diff = detection_map.get().astype(np.uint8) - detection_map_cpu.astype(np.uint8)
    if debug and np.sum(diff) == 0:
        print('results are binary exact!')
