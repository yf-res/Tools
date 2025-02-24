import numpy as np

def create_scaled_gaussian_kernel(size, A, center_value_fraction):
    """
    This function creates a gaussian target intensity kernel where the center pixel value is A
    and, based on the center_value_fraction% that represents the center pixel - calculates the total sum of the kernel
    :param size:
    :param A:
    :param center_value_fraction:
    :return:
    """
    # Calculate sigma
    total_energy = A / center_value_fraction
    sigma_squared = 1 / (2 * np.pi * center_value_fraction)
    sigma = np.sqrt(sigma_squared)

    # Generate the kernel
    kernel = np.zeros((size, size))
    center = size // 2

    for x in range(size):
        for y in range(size):
            kernel[x, y] = np.exp(-((x - center) ** 2 + (y - center) ** 2) / (2 * sigma ** 2))

    # Normalize the kernel so that the sum of all elements is 2.5A
    kernel_sum = np.sum(kernel)
    kernel = (kernel / kernel_sum) * total_energy

    # Scale the kernel so that the center value is A
    scaling_factor = A / kernel[center, center]
    kernel *= scaling_factor

    return kernel


def create_gaussian_kernel(size, A, center_value_fraction):
    """
    This function calculate a gaussian kernel that sum's up to the value A,
    where the center pixel represent center_value_fraction% of A
    :param size:
    :param A:
    :param center_value_fraction:
    :return:
    """
    # Calculate sigma
    sigma_squared = 1 / (2 * np.pi * center_value_fraction)
    sigma = np.sqrt(sigma_squared)
    # if size>=11:
    #     sigma = sigma*2

    # Generate the kernel
    kernel = np.zeros((size, size))
    center = size // 2

    for x in range(size):
        for y in range(size):
            kernel[x, y] = np.exp(-((x - center) ** 2 + (y - center) ** 2) / (2 * sigma ** 2))

    # Normalize the kernel to have a sum of A
    kernel_sum = np.sum(kernel)
    kernel = (kernel / kernel_sum) * A

    return kernel