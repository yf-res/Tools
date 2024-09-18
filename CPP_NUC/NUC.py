# python setup.py build_ext --inplace

import numpy as np
from nuc_wrapper import py_nuc_correction
from numba import njit, prange
import cupy as cp
import time
import numpy as np
from numba import njit, prange
import fast_nuc


def nuc_correction_numpy(input_image, gain, offset):
    return input_image * gain + offset
    # return np.clip(input_image * gain + offset, 0, 65535).astype(np.uint16)


@njit(parallel=True)
def nuc_correction_numba(input_image, gain, offset):
    height, width = input_image.shape
    output = np.empty((height, width), dtype=np.uint16)

    for i in prange(height):
        for j in range(width):
            val = input_image[i, j] * gain[i, j] + offset[i, j]
            output[i, j] = max(0, min(65535, val))

    return output


def nuc_correction_cupy(input_gpu, gain_gpu, offset_gpu):
    # input_gpu = cp.asarray(input_image)
    # gain_gpu = cp.asarray(gain)
    # offset_gpu = cp.asarray(offset)

    result_gpu = input_gpu * gain_gpu + offset_gpu
    return result_gpu
    # return cp.asnumpy(result_gpu).astype(np.uint16)


def benchmark(func, input_image, gain, offset, n_runs=10):
    times = []
    for _ in range(n_runs):
        start = time.time()
        _ = func(input_image, gain, offset)
        end = time.time()
        times.append(end - start)
    return np.mean(times), np.std(times)


# Generate test data
input_image = np.random.randint(0, 65536, (640, 1024), dtype=np.uint16)
gain = np.random.uniform(0.9, 1.1, (640, 1024)).astype(np.uint16)
offset = np.random.uniform(-100, 100, (640, 1024)).astype(np.uint16)

# Benchmark NumPy version
numpy_mean, numpy_std = benchmark(nuc_correction_numpy, input_image, gain, offset)
print(f"NumPy: {numpy_mean:.6f} ± {numpy_std:.6f} seconds")

# Benchmark Numba version
numba_mean, numba_std = benchmark(nuc_correction_numba, input_image, gain, offset)
print(f"Numba: {numba_mean:.6f} ± {numba_std:.6f} seconds")

# Benchmark CuPy version
input_gpu = cp.asarray(input_image)
gain_gpu = cp.asarray(gain)
offset_gpu = cp.asarray(offset)
cupy_mean, cupy_std = benchmark(nuc_correction_cupy, input_gpu, gain_gpu, offset_gpu)
print(f"CuPY: {cupy_mean:.6f} ± {cupy_std:.6f} seconds")

# Benchmark CPP version
cpp_mean, cpp_std = benchmark(py_nuc_correction, input_image, gain, offset)
print(f"CPP: {cpp_mean:.6f} ± {cpp_std:.6f} seconds")

# Benchmark FastNUC version
# Create FastNUC object
nuc = fast_nuc.FastNUC(gain, offset)
# Apply non-uniformity correction
times = []
for _ in range(10):
    start = time.time()
    _ = nuc.correct(input_image)
    end = time.time()
    times.append(end - start)
print(f"FastNUC: {np.mean(times):.6f} ± {np.std(times):.6f} seconds")

print()
