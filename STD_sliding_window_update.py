"""
This file explains why there is a need to calculate the full STD when trying the update it on a sliding window.
If you want to correctly update the STD on a sliding window - a Taylor series expansion is needed on the STD function.
"""
import numpy as np


def update_sliding_std(std_k, x_old, x_new, k):
    # Update sum of squares
    sum_squares_k = std_k**2 * (k - 1)
    sum_squares_k -= x_old**2
    sum_squares_k += x_new**2

    # Calculate new mean
    mean_k = (std_k**2 * k + x_new - x_old) / k

    # Compute new STD
    new_std = (sum_squares_k / (k - 1) - mean_k**2) ** 0.5
    return new_std

# Example usage:
vector_values = [1, 3, -1]
window_size = 3
initial_std = np.std(vector_values)
print('original std: ', initial_std)# Calculated STD for the first 3 elements
new_value = 100

# Update STD after adding the new value
updated_std = update_sliding_std(initial_std, vector_values[0], new_value, window_size)
print("Updated Sliding Window STD:", updated_std)
new_values = np.hstack([vector_values[1::], new_value])
print('fully calculate std: ', np.std(new_values))

