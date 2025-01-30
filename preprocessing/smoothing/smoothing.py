import cv2
import numpy as np


def smooth(image: np.ndarray, sigma: float) -> np.ndarray:
    """
    Applies Gaussian smoothing to the input image.

    Parameters:
        image (ndarray): Input image as an n*m*1 ndarray.
        sigma (float): Standard deviation for Gaussian smoothing.

    Returns:
        ndarray: Smoothed image.
    """
    return cv2.GaussianBlur(image, (0, 0), sigmaX=sigma, sigmaY=sigma)
