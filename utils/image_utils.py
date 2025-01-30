import cv2
import numpy as np


def read_image(filepath):
    """
    Reads a grayscale image and converts it to a n*m*1 ndarray.

    Parameters:
        filepath (str): Path to the grayscale PNG image.

    Returns:
        ndarray: Image as an n*m*1 array.
    """
    image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Image file not found: {filepath}")
    return image[:, :, np.newaxis]
