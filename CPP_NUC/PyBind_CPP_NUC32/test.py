import numpy as np
import fast_nuc

# Create sample gain and offset matrices
height, width = 1080, 1920
gain = np.random.uniform(0.9, 1.1, (height, width)).astype(np.float32)
offset = np.random.uniform(-100, 100, (height, width)).astype(np.float32)

# Create FastNUC object
nuc = fast_nuc.FastNUC(gain, offset)

# Create a sample input image
input_image = np.random.randint(0, 65536, (height, width), dtype=np.uint16)

# Apply non-uniformity correction
corrected_image = nuc.correct(input_image)

print(corrected_image.shape)