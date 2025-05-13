import numpy as np
import cv2


class Dob:

    @staticmethod
    def dob_filter(image, k, m):
        if k >= m:
            raise ValueError("Size of W1 (k) must be smaller than size of W2 (m)")

        # Compute the sum of pixels in the k x k and m x m boxes
        box_filter1 = Dob.create_box_filter(k) * (k ** 2)
        box_filter2 = Dob.create_box_filter(m) * (m ** 2)

        S1 = Dob.apply_filter(image, box_filter1)
        S2 = Dob.apply_filter(image, box_filter2)

        # Calculate the DoB result
        dob_result = S1 / (k ** 2) - (S2 - S1) / (m ** 2 - k ** 2)

        # dob_result = reduce_and_mask(dob_result)
        return dob_result

    @staticmethod
    def apply_filter(image, filter):
        """Applies a given filter to the image using convolution."""
        return cv2.filter2D(image, -1, filter)

    @staticmethod
    def do_b_kernel(image, size1, size2):
        """
        Applies the DoB kernel to an image.

        Parameters:
        - image: Grayscale input image
        - size1: Size of the smaller box filter
        - size2: Size of the larger box filter

        Returns:
        - Result of the DoB kernel application
        """
        box_filter1 = Dob.create_box_filter(size1)
        box_filter2 = Dob.create_box_filter(size2)

        convolved1 = Dob.apply_filter(image, box_filter1)
        convolved2 = Dob.apply_filter(image, box_filter2)

        do_b_result = convolved1 - convolved2

        return do_b_result

    @staticmethod
    def create_box_filter(size):
        """Creates a box filter of given size."""
        return np.ones((size, size), dtype=np.float32) / (size * size)
