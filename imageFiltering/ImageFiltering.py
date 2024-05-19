import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

# Create a sample 1D signal (image row)
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
print("Up-sampled Signal: ", upsampled_signal)