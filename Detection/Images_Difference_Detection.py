import numpy as np
import cv2
from skimage import exposure, filters
from skimage.restoration import denoise_tv_bregman as tv_denoising
from skimage.restoration import denoise_tv_chambolle
from scipy import sparse
from Utils.utils_registration import UtilsRegistration as ur
from Spectro_MWIR.show_pair import *
import pickle
from scipy.special import expit
import matplotlib.pyplot as plt
import matplotlib
from scipy.signal import convolve2d
from scipy import ndimage
matplotlib.use('TkAgg')
plt.ion()


class ImageDifferenceAnalyzer:
    def __init__(self):
        pass

    @staticmethod
    def normalize_image(image):
        """
        Normalizes image by its mean intensity.
        Args:
            image: A numpy array representing the image.
        Returns:
            A numpy array representing the normalized image.
        """
        return image.astype(np.float32) / np.mean(image)

    @staticmethod
    def match_filter(main_image):
        # Create a 3x3 Gaussian template
        gaussian_template = np.array([[0, 1, 0],
                                      [1, 2, 1],
                                      [0, 1, 0]], dtype=np.float32)
        gaussian_template /= np.sum(gaussian_template)

        # Apply matchTemplate to find the Gaussian pattern in the main image
        result = cv2.matchTemplate(main_image, gaussian_template, cv2.TM_CCOEFF_NORMED)

        # Normalize the result for better visualization
        result = cv2.normalize(result, None, 0, 1, cv2.NORM_MINMAX, -1)

        # threshold = 0.5  # You can adjust this threshold value
        # _, thresholded_result = cv2.threshold(result, threshold, 1.0, cv2.THRESH_BINARY)

        # # Visualize the results
        # plt.figure(figsize=(12, 6))
        #
        # plt.subplot(1, 2, 1)
        # plt.title('Main Image')
        # plt.imshow(main_image, cmap='gray')
        #
        # plt.subplot(1, 2, 2)
        # plt.title('Gaussian Template')
        # plt.imshow(gaussian_template, cmap='gray')
        #
        # plt.show()

        return result

    @staticmethod
    def median_filter(image, kernel_size=5):
        """
        Applies median filter to remove noise.
        To reduce noise while preserving edges by using the median of the pixel values in the neighborhood defined by `kernel_size`
        Args:
            image: A numpy array representing the image.
            kernel_size: Size of the median filter kernel (default 5).
        Returns:
            A numpy array representing the filtered image.
        """
        return cv2.medianBlur(image, kernel_size)

    @staticmethod
    def morphological_opening(image, kernel_size=3):
        """
        Performs morphological opening to remove small differences.
        To smooth the image by removing small foreground details
        Args:
            image: A numpy array representing the image.
            kernel_size: Size of the structuring element (default 3).
        Returns:
            A numpy array representing the image after opening.
        """
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

    @staticmethod
    def local_adaptive_threshold(image, block_size=41, C=2):
        """
        Applies local adaptive thresholding for change detection.
        To segment the image based on local variations in pixel intensities
        Args:
            image: A numpy array representing the image.
            block_size: Neighborhood size for threshold calculation (default 41).
            C: Constant subtracted from the mean (default 2).
        Returns:
            A numpy array representing the thresholded image.
        """
        return cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size, C)

    # Image subtraction with absolute difference and thresholding
    @staticmethod
    def abs_diff_threshold(image1, image2, threshold=10):
        """
        Subtracts images with absolute difference and applies thresholding.
        To highlight differences between two images by computing the absolute difference and applying a threshold
        Args:
            image1: A numpy array representing the first image.
            image2: A numpy array representing the second image.
            threshold: Threshold value for change detection (default 10).
        Returns:
            A numpy array representing the change mask.
        """
        diff = cv2.absdiff(image1, image2)
        _, mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
        return mask

    @staticmethod
    # Image subtraction with block-matching (Sum of Absolute Differences - SAD)
    def block_matching_sad(image1, image2, block_size=8):
        """
        Performs block-matching using Sum of Absolute Differences (SAD).
        To compare blocks of pixels between two images to identify differences
        Args:
            image1: A numpy array representing the first image.
            image2: A numpy array representing the second image.
            block_size: Size of the block for comparison (default 8).
        Returns:
            A numpy array representing the change mask.
        """
        height, width = image1.shape[:2]
        result = np.zeros_like(image1, dtype=np.float32)
        for y in range(block_size, height, block_size):
            for x in range(block_size, width, block_size):
                ref_block = image1[y - block_size:y, x - block_size:x]
                diff = np.sum(np.abs(ref_block - image2[y - block_size:y, x - block_size:x]))
                result[y, x] = diff
        return result > np.mean(result) * 0.8  # Threshold based on mean

    @staticmethod
    def histogram_matching(image1, image2):
        '''To adjust the pixel intensities of one image to match the histogram of another, enhancing similarity'''
        # matched = exposure.match_histograms(image1, image2, multichannel=True)
        matched = exposure.match_histograms(image1, image2)
        return matched

    @staticmethod
    def bilateral_filter(image, d=9, sigmaColor=75, sigmaSpace=75):
        '''To smooth the image while preserving edges by filtering the pixel values based on spatial and intensity differences'''
        return cv2.bilateralFilter(image, d, sigmaColor, sigmaSpace)

    @staticmethod
    def difference_of_gaussians(image, low_sigma=1, high_sigma=2):
        '''To detect edges by subtracting a blurred version of the image (low sigma) from a more blurred version (high sigma)'''
        low_freq = filters.gaussian(image, sigma=low_sigma)
        high_freq = filters.gaussian(image, sigma=high_sigma)
        return low_freq - high_freq

    @staticmethod
    def adaptive_threshold(image, block_size=11, C=2):
        '''To segment the image into foreground and background based on local intensity variations'''
        # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, block_size, C)

    def process_images(self, image1, image2, method='all'):
        """
        To apply different image processing methods to analyze differences between two images
        To use this class:

        Create an instance of ImageDifferenceAnalyzer
        Load your two images using OpenCV's imread function
        Call the process_images method with your two images

        # Usage example:
        # analyzer = ImageDifferenceAnalyzer()
        # img1 = cv2.imread('image1.jpg')
        # img2 = cv2.imread('image2.jpg')
        # result = analyzer.process_images(img1, img2)
        # cv2.imshow('Difference', result)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        """

        if method == 'histogram_matching' or method == 'all':
            image1 = self.histogram_matching(image1, image2)

        if method == 'bilateral_filter' or method == 'all':
            image1 = self.bilateral_filter(image1)
            image2 = self.bilateral_filter(image2)

        if method == 'difference_of_gaussians' or method == 'all':
            image1 = self.difference_of_gaussians(image1)
            image2 = self.difference_of_gaussians(image2)

        difference = cv2.absdiff(image1, image2)

        if method == 'adaptive_threshold' or method == 'all':
            difference = self.adaptive_threshold((difference*255).astype(np.uint8))

        return difference

    def tv_denoising_diff(self,image1, image2, weight=0.1):
        """
        To denoise images using Total Variation and then identify differences
        Applies TV denoising to both images and subtracts the results:
        We import tv_denoising from skimage.restoration.
        The function takes two images and a weight parameter.
        Both images are converted to float32 for numerical stability.
        tv_denoising is applied to each image with the specified weight. This weight controls the trade-off between noise removal and edge preservation.
        The denoised images are subtracted, and absolute difference is calculated.
        Thresholding is applied to obtain a binary mask highlighting significant changes.
        Note: This is a simplified example. More sophisticated variational methods can be implemented for change detection, often involving solving minimization problems. It's recommended to explore libraries like scikit-image for more advanced variational tools.

        Args:
            image1: A numpy array representing the first image.
            image2: A numpy array representing the first image.
            weight: Regularization weight for TV denoising (default 0.1).
        Returns:
            A numpy array representing the change mask.
        """
        denoised1 = tv_denoising(image1.astype(np.float32), weight=weight)
        denoised2 = tv_denoising(image2.astype(np.float32), weight=weight)
        diff = cv2.absdiff(denoised1, denoised2)
        # diff_matched = self.match_filter(diff) # doesn't look good

        _, mask = cv2.threshold(diff, 10, 255, cv2.THRESH_BINARY)
        return mask

    @staticmethod
    def abs_diff_and_tv_denoise(image1, image2, threshold=0.1, weight=0.1):
        """
        To highlight significant differences between two images after denoising
        This function first subtract the images using absolute diff and then performs total variation denoising.
        the result is then threshold by a pre-defined threshold
        :param image1: A numpy array representing the first image.
        :param image2: A numpy array representing the first image.
        :param threshold: Threshold to highlight significant differences
        :param weight: Regularization weight for TV denoising (default 0.1).
        :return: Threshold image (binary)
        """
        # Compute the initial difference image
        difference_image = cv2.absdiff(image1, image2)

        # Apply Total Variation denoising
        tv_denoised = denoise_tv_chambolle(difference_image, weight=0.1)

        # Threshold to highlight significant differences
        thresholded = np.where(tv_denoised > 0.1, 1, 0)

        # # Display or save the thresholded image
        # plt.figure(), plt.imshow(thresholded)
        # plt.close()
        return thresholded

    @staticmethod
    def variational_bayesian_denoising(difference_image, alpha=1.0, beta=0.1, max_iter=100):
        """
        To estimate a clean version of a difference image using Bayesian inference
        Performs variational Bayesian denoising on a difference image.

        This function implements a variational inference approach for denoising an image
        representing the difference between two registered images. It assumes that the
        difference is corrupted by additive Gaussian noise. The denoising process
        estimates the mean (`mu`) and variance (`sigma_sq`) of the underlying clean image.

        Args:
            difference_image: A numpy array representing the difference image.
            alpha: Hyperparameter controlling the prior belief about the noise variance
              (default 1.0). Higher values imply stronger belief in low noise.
            beta: Hyperparameter controlling the prior belief about the image intensity
              (default 0.1). Higher values favor smoother images.
            max_iter: Maximum number of iterations for the variational inference
              algorithm (default 100).

        Returns:
            A tuple containing two numpy arrays:
                - mu: The estimated mean (denoised image) after variational inference.
                - sigma_sq: The estimated variance of the denoised image.
        """

        mu = np.zeros_like(difference_image)  # Initialize mean (denoised image)
        sigma_sq = np.ones_like(difference_image)  # Initialize variance

        for i in range(max_iter):
            # E-step: Update mean and variance estimates
            mu_new = (alpha * difference_image + beta * mu) / (alpha + beta)
            sigma_sq_new = 1 / (alpha + beta)

            # M-step: Update hyperparameters based on new estimates
            alpha_new = np.mean(mu_new ** 2 + sigma_sq_new)
            beta_new = 1 / np.var(difference_image - mu_new)

            # Update for next iteration
            mu, sigma_sq = mu_new, sigma_sq_new
            alpha, beta = alpha_new, beta_new

        return mu, sigma_sq

    def abs_diff_and_variational_bayes_denoise(self, image1, image2):
        '''To identify and highlight anomalies between two images using advanced denoising techniques'''
        # Compute the initial difference image
        difference_image = cv2.absdiff(image1, image2)

        # Apply Variational Bayesian Inference
        vb_denoised, vb_sigma_sq = self.variational_bayesian_denoising(difference_image)

        # Apply a sigmoid function to threshold the anomalies
        anomalies = expit(vb_denoised - 0.5)

        # # Display or save the anomalies image
        # plt.figure(), plt.imshow(anomalies)
        # plt.close()
        return anomalies

    @staticmethod
    def block_adaptive_thresholding(image, block_size, C):
        """
        To detect changes locally by adjusting thresholds based on local blocks
        Performs adaptive thresholding on an image for local change detection.

        This function implements adaptive thresholding to separate foreground objects
        from the background in an image. It calculates a local threshold for each pixel
        based on its surrounding neighborhood (block) and a constant value (C).

        Args:
            image: A numpy array representing the grayscale image.
            block_size: The size of the neighborhood (block) used for local mean and
              standard deviation calculation (must be odd and greater than 1).
            C: A constant value that multiples the local standard deviation to define
              the adaptive threshold (default 2). Higher values lead to stricter
              thresholding.

        Returns:
            A numpy array representing the thresholded image, where pixels exceeding
            the adaptive threshold are set to 1 (foreground) and others to 0 (background).
        """

        mean = cv2.boxFilter(image, cv2.CV_32F, (block_size, block_size))
        mean_sq = cv2.boxFilter(image ** 2, cv2.CV_32F, (block_size, block_size))
        std = np.sqrt(mean_sq - mean ** 2)
        thresholded = np.where(image > mean + C * std, 1, 0)
        return thresholded

    def adaptive_threshold_with_tv_denoise(self, image1, image2, block_size=15, C=3):
        '''To detect changes after denoising using Total Variation'''
        # Compute the initial difference image
        difference_image = cv2.absdiff(image1, image2)

        # Apply Total Variation denoising
        tv_denoised = denoise_tv_chambolle(difference_image, weight=0.1)

        # Apply adaptive thresholding
        thresholded = self.block_adaptive_thresholding(tv_denoised, block_size, C)

        # # Display or save the anomalies image
        # plt.figure(), plt.imshow(thresholded)
        # plt.close()
        return thresholded


    def adaptive_threshold_with_bayes_denoise(self, image1, image2, block_size=15, C=3):
        '''To highlight differences after denoising using Bayesian methods'''
        # Compute the initial difference image
        difference_image = cv2.absdiff(image1, image2)

        # Apply Variational Bayesian Inference
        vb_denoised, vb_sigma_sq = self.variational_bayesian_denoising(difference_image)

        # Apply adaptive thresholding
        thresholded = self.block_adaptive_thresholding(vb_denoised, block_size, C)

        # # Display or save the anomalies image
        # plt.figure(), plt.imshow(thresholded)
        # plt.close()
        return thresholded

    def mumford_shah_difference(self, image1, image2, alpha=0.1, beta=0.05, num_iter=100):
        """
        This method is good at segmenting images while preserving edges,
        which makes it well-suited for detecting significant differences between two images.
        Mumford-Shah Difference Detection:
        The Mumford-Shah functional aims to approximate an image with a piecewise smooth function.
        The simplified Mumford-Shah functional can be formulated as:
        minimize ∫(u - (I1 - I2))² dx + α∫|∇u|² dx + β|Γ|
        Where:

        u is the piecewise smooth approximation of the difference we're trying to estimate
        I1 and I2 are the input images
        Γ represents the set of discontinuities (edges) in u
        α and β are parameters controlling the smoothness and the length of the edges respectively

        This model effectively segments the difference image into smooth regions separated by sharp boundaries,
        which can help in identifying significant differences while suppressing noise and small variations.

        # Usage:
        analyzer = ImageVariationalDifferenceAnalyzer()
        result = analyzer.mumford_shah_difference(img1, img2)

        This implementation uses a gradient descent approach to approximate the Mumford-Shah functional.

        :param self:
        :param image1: first image
        :param image2: second image (aligned to the first image)
        :param alpha: controls the smoothness of the solution within regions
        :param beta: controls the sensitivity to edges
        :param num_iter: number of iterations for gradient descent
        :return: difference image (before thresholding - not binary)
        """
        # Convert images to float and compute initial difference
        I1 = image1.astype(float) / 255.0
        I2 = image2.astype(float) / 255.0
        u = (I1 - I2)

        # Get image dimensions
        rows, cols = I1.shape[:2]

        # Iterate to solve the Mumford-Shah problem
        for _ in range(num_iter):
            # Compute gradients
            ux, uy = np.gradient(u)
            uxx, _ = np.gradient(ux)
            _, uyy = np.gradient(uy)

            # Update u
            delta_u = (I1 - I2 - u) - alpha * (uxx + uyy)

            # Edge stopping function
            g = 1 / (1 + beta * (ux ** 2 + uy ** 2))

            u += 0.1 * g * delta_u  # 0.1 is a step size for gradient descent

        # Normalize and convert back to uint8
        u = (u - u.min()) / (u.max() - u.min())
        return (u * 255).astype(np.uint8)

    def tv_l1_difference(self, image1, image2, lambda_param=1.0, num_iter=500):
        """
        Variational methods in image processing provide a powerful framework for image analysis and restoration.
        For detecting differences between images, we can use a variational approach based on the Total Variation (TV) model.
        This method is particularly effective at preserving edges while smoothing out noise and small variations.
        Total Variation-L1 (TV-L1) Difference Detection:
        This method combines the Total Variation regularization with L1 data fidelity.
        It's particularly good at preserving sharp edges in the difference image while suppressing noise and small artifacts.
        The TV-L1 model for difference detection can be formulated as:
        minimize |∇u| + λ|u - (I1 - I2)|
        Where:

        u is the difference image we're trying to estimate
        I1 and I2 are the input images
        |∇u| is the Total Variation of u (L1 norm of the gradient)
        λ is a regularization parameter that balances the data fidelity term with the regularization term

        This model effectively denoises the naive difference (I1 - I2) while preserving significant structural differences.

        # Usage
        analyzer = ImageVariationalDifferenceAnalyzer()
        result = analyzer.tv_l1_difference(img1, img2)

        The TV-L1 method is particularly good at:

        Preserving sharp edges in the difference image
        Suppressing noise and small artifacts
        Producing piecewise constant regions, which can help in segmenting the differences

        :param image1: first image
        :param image2: second image (aligned to the first image)
        :param lambda_param: lambda_param controls the balance between the Total Variation regularization and the data fidelity term
                            A higher value will result in more denoising but might also remove some actual differences
        :param num_iter: number of iteration until convergence
        :return: difference image (before thresholding - not binary)
        """

        # Convert images to float and compute initial difference
        I1 = image1.astype(float) / 255.0
        I2 = image2.astype(float) / 255.0
        u = I1 - I2

        # Get image dimensions
        rows, cols = I1.shape[:2]
        size = rows * cols

        # Compute gradient operators
        dx = sparse.diags([-1, 1], [0, 1], shape=(size, size))
        dy = sparse.diags([-1, 1], [0, rows], shape=(size, size))

        # Iterate to solve the TV-L1 problem
        for _ in range(num_iter):
            # Compute gradients
            ux = dx.dot(u.flatten())
            uy = dy.dot(u.flatten())

            # Compute gradient magnitude
            unorm = np.sqrt(ux ** 2 + uy ** 2 + 1e-6)

            # Update u
            v = dx.T.dot(ux / unorm) + dy.T.dot(uy / unorm)
            u = (v + lambda_param * (I1.flatten() - I2.flatten())) / (1 + lambda_param)

            u = u.reshape(rows, cols)

        return np.clip(u * 255, 0, 255).astype(np.uint8)

    """
    Comparison of variational TV-L1 and Mumford-Shau functional:

    Underlying Principles:
    TV-L1:
    Based on Total Variation (TV) regularization
    Uses L1 norm for data fidelity
    Aims to preserve edges while promoting piecewise constant regions

    Mumford-Shah:
    Based on piecewise smooth approximation
    Uses L2 norm for data fidelity
    Aims to segment the image into smooth regions separated by sharp boundaries


    Regularization:
    TV-L1:

    Uses the L1 norm of the gradient (|∇u|) as regularization
    Promotes sparsity in the gradient domain

    Mumford-Shah:

    Uses the L2 norm of the gradient (|∇u|²) within regions
    Explicitly penalizes the length of boundaries between regions


    Results Characteristics:
    TV-L1:

    Tends to produce piecewise constant results
    Very good at preserving sharp edges
    Can sometimes create "staircase" artifacts in smooth gradients

    Mumford-Shah:

    Produces piecewise smooth results
    Preserves significant edges while allowing smooth variations within regions
    Can better handle gradual changes


    Noise Handling:
    TV-L1:

    Very effective at removing noise
    Can sometimes remove small details along with noise

    Mumford-Shah:

    Also effective at noise reduction
    May preserve more fine details, depending on parameter settings


    Parameter Sensitivity:
    TV-L1:

    Usually has one main parameter (λ) to balance data fidelity and regularization
    Relatively straightforward to tune

    Mumford-Shah:

    Has two main parameters (α and β) controlling smoothness and edge preservation
    Can require more careful tuning to balance these aspects


    Computational Aspects:
    TV-L1:

    Generally faster to compute
    Has well-established efficient algorithms for solving

    Mumford-Shah:

    Can be more computationally intensive
    Our implementation uses a simple gradient descent approach, which might converge slower


    Applicability:
    TV-L1:

    Well-suited for cases where the differences are expected to be sharp and distinct
    Good for scenarios with potential illumination changes between images

    Mumford-Shah:

    Well-suited for cases where differences might include both sharp changes and gradual variations
    Good for scenarios where you want to segment the difference image into distinct regions


    In practice, the choice between TV-L1 and Mumford-Shah often depends on the specific characteristics of your images 
    and the type of differences you're trying to detect:

    If you're primarily interested in sharp, distinct changes and can tolerate some loss of smooth gradients, TV-L1 might be preferable.
    If you need to preserve both sharp changes and smooth variations in the difference image, Mumford-Shah could be a better choice.
    """


class MultiScaleAnomalyDetector:
    """

    # Usage

    # detector = MultiScaleAnomalyDetector()

    # img = cv2.imread('your_image.png', 0)  # Read as grayscale

    # anomalies, response = detector.detect_anomalies(img)

    # cv2.imshow('Detected Anomalies', anomalies.astype(np.uint8) * 255)

    # cv2.imshow('Multi-scale Response', response / response.max())

    # cv2.waitKey(0)



    Key components:

    a) DoG Pyramid:



    We create a pyramid of DoG images by subtracting Gaussian-blurred versions of the original image.

    Each level of the pyramid represents a different scale.



    b) build_dog_pyramid method:



    Iterates through different scales.

    For each scale, it creates two Gaussian-blurred versions of the image with different sigma values.

    Subtracts these to create a DoG image.



    c) detect_anomalies method:



    Builds the DoG pyramid.

    Combines responses from all scales by summing the absolute values.

    Thresholds the combined response to identify anomalies.



    The principle behind this approach is that anomalies will show up as strong responses in the DoG images, potentially at different scales depending on their size.

    """

    def __init__(self, num_scales=5, start_sigma=1, k=1.6):
        self.num_scales = num_scales

        self.start_sigma = start_sigma

        self.k = k

    def build_dog_pyramid(self, image):
        pyramid = []

        for i in range(self.num_scales - 1):
            sigma1 = self.start_sigma * (self.k ** i)

            sigma2 = self.start_sigma * (self.k ** (i + 1))

            g1 = ndimage.gaussian_filter(image, sigma1)

            g2 = ndimage.gaussian_filter(image, sigma2)

            dog = g1 - g2

            pyramid.append(dog)

        return pyramid

    def detect_anomalies(self, image, threshold=2.5):
        dog_pyramid = self.build_dog_pyramid(image)

        # Combine responses from all scales

        combined_response = np.sum(np.abs(np.array(dog_pyramid)), axis=0)

        # Threshold to get binary mask

        mean = np.mean(combined_response)

        std = np.std(combined_response)

        anomalies = combined_response > (mean + threshold * std)

        return anomalies, combined_response


class match_Filter():
    """

    # Usage

    # img = cv2.imread('your_image.png', 0)  # Read as grayscale

    # detected, response = detect_bright_anomaly(img)

    # cv2.imshow('Detected Bright Anomalies', detected.astype(np.uint8) * 255)

    # cv2.imshow('Filter Response', (response - response.min()) / (response.max() - response.min()))

    # cv2.waitKey(0)



    # Usage

    # img = cv2.imread('your_image.png', 0)  # Read as grayscale

    # detected, response = detect_dark_anomaly(img)

    # cv2.imshow('Detected Dark Anomalies', detected.astype(np.uint8) * 255)

    # cv2.imshow('Filter Response', (response - response.min()) / (response.max() - response.min()))

    # cv2.waitKey(0)



    Key points about these filters:



    The filters are designed as 2D Gaussian functions, which work well for detecting blob-like anomalies.

    The filters are made zero-sum by subtracting the mean.

    This ensures that they respond strongly to local variations but not to uniform areas.

    The sigma parameter controls the size of the anomaly we're looking for.

    Smaller sigma values will detect smaller anomalies.

    The threshold parameter in the detection functions controls how strong a response needs to be to be considered an anomaly.

    You may need to adjust this based on your specific images.

    The filter for dark anomalies is simply an inverted version of the bright anomaly filter.



    To use these filters effectively:



    Adjust the filter_size and sigma parameters to match the expected size of your anomalies.

    Experiment with different threshold values to balance between detecting all anomalies and avoiding false positives.

    You can visualize the filter response (the result returned by the detection functions) to see how strongly different

    parts of the image are responding to the filter.



    These matched filters can be very effective for detecting small, localized anomalies in images.

    However, they may struggle with very large anomalies or those with complex shapes.

    For more complex scenarios, you might need to consider more advanced techniques like multi-scale analysis or

    machine learning-based approaches.



    """

    @staticmethod
    def create_bright_anomaly_filter(size=7, sigma=1):
        """Create a 2D Gaussian filter for detecting bright anomalies."""

        x, y = np.mgrid[-size // 2 + 1:size // 2 + 1, -size // 2 + 1:size // 2 + 1]

        g = np.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2)))

        return g - g.mean()  # Subtract mean to make the filter zero-sum

    @staticmethod
    def create_dark_anomaly_filter(size=7, sigma=1):
        """Create an inverted 2D Gaussian filter for detecting dark anomalies."""

        x, y = np.mgrid[-size // 2 + 1:size // 2 + 1, -size // 2 + 1:size // 2 + 1]

        g = np.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2)))

        return (g.mean() - g)  # Invert the filter

    def detect_bright_anomaly(self, image, filter_size=7, sigma=1, threshold=0.1):
        """Detect bright anomalies in the image."""

        # Create the filter

        filt = self.create_bright_anomaly_filter(filter_size, sigma)

        # Apply the filter
        pad_width = ((1, 1), (1, 1))  # Padding for a 3x3 kernel
        padded_image = np.pad(image, pad_width, mode='reflect')
        result = convolve2d(padded_image, filt, mode='valid')

        loc_max = np.unravel_index(np.argmax(result),result.shape)
        target_roi = result[loc_max[0]-9:loc_max[0]+9,loc_max[1]-9:loc_max[1]+9]
        mean_std_thresh = np.mean(target_roi) + 1.5*np.std(target_roi)

        # Threshold the result
        detected = result > mean_std_thresh

        return detected, result

    def detect_dark_anomaly(self, image, filter_size=7, sigma=1, threshold=0.1):
        """Detect dark anomalies in the image."""

        # Create the filter

        filt = self.create_dark_anomaly_filter(filter_size, sigma)

        # Apply the filter
        pad_width = ((1, 1), (1, 1))  # Padding for a 3x3 kernel
        padded_image = np.pad(image, pad_width, mode='reflect')
        result = np.abs(convolve2d(padded_image, filt, mode='valid'))

        loc_max = np.unravel_index(np.argmax(result),result.shape)
        target_roi = result[loc_max[0]-9:loc_max[0]+9,loc_max[1]-9:loc_max[1]+9]
        mean_std_thresh = np.mean(target_roi) + 1.5*np.std(target_roi)

        # Threshold the result
        detected = result > mean_std_thresh

        return detected, result

    def detect_anomaly(self, image,mode, curr_template, filter_type, filter_size=7, sigma=1, threshold=0.1):
        """Detect bright anomalies in the image."""

        # Create the filter

        val_max = None
        filt = self.create_bright_anomaly_filter(filter_size, sigma)

        if filter_type=='match_filter':
            if mode=='acquistion_OneShot':
                # Calculate the required padding
                pad_height = (filt.shape[0] - 1) // 2
                pad_width = (filt.shape[1] - 1) // 2
                padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), mode='reflect')

                # Perform convolution
                result = convolve2d(padded_image, filt, mode='same')

                # Crop the result to the original image size
                result = result[pad_height:pad_height + image.shape[0], pad_width:pad_width + image.shape[1]]

                # abs to find dark anomaly
                result = np.abs(result)
            elif 'Track':
                result = convolve2d(image, filt, mode='valid')
                pad_with = int((image.shape[0]-result.shape[0])/2)
                result = np.pad(result, pad_with, mode='constant', constant_values=0) # pad it back to original size
        elif filter_type=='template_filter':
            result_before = cv2.matchTemplate(image.astype(np.float32), curr_template.astype(np.float32),
                                       cv2.TM_CCOEFF_NORMED)
            result = np.zeros((image.shape[0], image.shape[0]), dtype=np.float32)
            start_x = (result.shape[1] - result_before.shape[1]) // 2
            start_y = (result.shape[0] - result_before.shape[0]) // 2
            result[start_y:start_y + result_before.shape[0], start_x:start_x + result_before.shape[1]] = result_before
            val_max = np.max(result_before)

        loc_max = np.unravel_index(np.argmax(result),result.shape)

        if filter_type=='match_filter' or mode=='acquistion_OneShot':
            roi_size = 9
            # Calculate the start and end indices for the ROI, with boundary checks
            start_x = max(loc_max[0] - roi_size, 0)
            end_x = min(loc_max[0] + roi_size, result.shape[0])
            start_y = max(loc_max[1] - roi_size, 0)
            end_y = min(loc_max[1] + roi_size, result.shape[1])
            # Extract the ROI
            target_roi = result[start_x:end_x, start_y:end_y]
            mean_std_thresh = np.mean(target_roi) + 1.5*np.std(target_roi)
            # Threshold the result
            detected = result > mean_std_thresh
        elif filter_type=='template_filter':
            detected = np.zeros_like(result)
            detected[int(loc_max[0]),int(loc_max[1])] = 1

        return detected, result, loc_max, val_max

    @staticmethod
    def op_ca_cfar(image, thr=7, window_size=13, guard_s=11):
        # guard_s = 5
        # window_size = 9
        cell_size = (window_size, window_size)
        guard_size = (guard_s, guard_s)
        # detection_map = np.full((image.shape[0], image.shape[1]), False)
        cent = cell_size[0] // 2
        kernel = np.ones(cell_size)
        kernel[cent - guard_size[0] // 2:cent + guard_size[0] // 2 + 1,
        cent - guard_size[1] // 2:cent + guard_size[1] // 2 + 1] = 0
        # kernel = kernel/np.nansum(kernel)
        # avg_image = convolve2d(image, kernel, mode='same') / convolve2d(~np.isnan(frame), kernel, mode='same')

        avg_image = convolve2d(image, kernel, mode='same') / np.sum(kernel)
        sqr_pixel_mean = convolve2d(image ** 2, kernel, mode='same') / np.sum(kernel)
        std_img = np.sqrt(sqr_pixel_mean - avg_image ** 2)
        cell_thr = avg_image + thr * std_img
        detection_map = image > cell_thr

        return detection_map, cell_thr

    @staticmethod
    def create_box_filter(size):
        """Creates a box filter of given size."""
        return np.ones((size, size), dtype=np.float32) / (size * size)

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
        box_filter1 = ImageDifferenceAnalyzer.create_box_filter(size1)
        box_filter2 = ImageDifferenceAnalyzer.create_box_filter(size2)

        convolved1 = ImageDifferenceAnalyzer.apply_filter(image, box_filter1)
        convolved2 = ImageDifferenceAnalyzer.apply_filter(image, box_filter2)

        do_b_result = convolved1 - convolved2

        return do_b_result

    @staticmethod
    def dob_filter(image, k, m):
        if k >= m:
            raise ValueError("Size of W1 (k) must be smaller than size of W2 (m)")

        # Compute the sum of pixels in the k x k and m x m boxes
        box_filter1 = ImageDifferenceAnalyzer.create_box_filter(k) * (k ** 2)
        box_filter2 = ImageDifferenceAnalyzer.create_box_filter(m) * (m ** 2)

        S1 = ImageDifferenceAnalyzer.apply_filter(image, box_filter1)
        S2 = ImageDifferenceAnalyzer.apply_filter(image, box_filter2)

        # Calculate the DoB result
        dob_result = S1 / (k ** 2) - (S2 - S1) / (m ** 2 - k ** 2)

        return dob_result

# # Example usage
# folder_path = '/mnt/nfs_storage/Public/MDSL/MDS_movies/outputs_before_10x_on_signal/'
# valid_area = [384, 480]
# mat_files = [f for f in os.listdir(folder_path) if f.endswith('.mat')]
# mat_files = [mat_files[9]]
# for mat_file in mat_files:
#     mat_path = os.path.join(folder_path, mat_file)
#     data = loadmat(mat_path)
#     if 'planted_mov' in data:
#         images = data['planted_mov']
#     else:
#         print(f"No 'images' key found in {mat_file}")
#
#     debug = False
#     # Parameters for DoB kernel
#     size1 = 2  # Size of the smaller box filter
#     size2 = 11  # Size of the larger box filter
#     num_images = images.shape[2]
#     psnr_list = []
#     # Read and display the images in the sorted order
#     for idx in tqdm(range(num_images)):
#         img1 = images[1:384, 1:480, idx]
#
#         # Apply DoB kernel
#         dob_result = dob_filter(img1, size1, size2)
#
#         if debug:
#             # Create a figure and two axes
#             fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
#             # Display the images
#             im1 = ax1.imshow(img1, cmap='gray')
#             im2 = ax2.imshow(dob_result, cmap='gray')
#
#             # Function to update the zoom and pan
#             def update_view(event):
#                 if event is not None and event.inaxes in [ax1, ax2]:
#                     if event.inaxes == ax1:
#                         xlim = ax1.get_xlim()
#                         ylim = ax1.get_ylim()
#                         ax2.set_xlim(xlim)
#                         ax2.set_ylim(ylim)
#                     elif event.inaxes == ax2:
#                         xlim = ax2.get_xlim()
#                         ylim = ax2.get_ylim()
#                         ax1.set_xlim(xlim)
#                         ax1.set_ylim(ylim)
#                     fig.canvas.draw_idle()
#
#
#             # Connect the zoom and pan events to the function
#             fig.canvas.mpl_connect('button_release_event', update_view)
#             fig.canvas.mpl_connect('scroll_event', update_view)
#             fig.canvas.mpl_connect('motion_notify_event', update_view)
#
#             # Enable interactive pan and zoom
#             ax1.set_xlim([0, img1.shape[1]])
#             ax1.set_ylim([img1.shape[0], 0])
#             ax2.set_xlim([0, img1.shape[1]])
#             ax2.set_ylim([img1.shape[0], 0])
#
#             plt.show()
#
#         pnsr = (dob_result.max() - np.mean(dob_result)) / np.std(dob_result)
#         psnr_list.append(pnsr)
#
#
# plt.figure(figsize=(10, 5))
# plt.title('psnr')
# plt.plot(psnr_list)
# plt.show()
#
#
#
#
