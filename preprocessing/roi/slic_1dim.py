import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.segmentation import slic
from skimage.color import label2rgb

def read_grayscale_image(filepath):
    """
    Reads a grayscale image from a file and converts it to a 2D array.

    Parameters:
        filepath (str): Path to the grayscale image file.

    Returns:
        ndarray: Grayscale image as an array of shape (n, m, 1).
    """
    image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Image file not found: {filepath}")
    return image[:, :, np.newaxis]

def slic_segmentation(image, n_segments=100, compactness=10):
    """
    Performs SLIC (Simple Linear Iterative Clustering) segmentation on a grayscale image.

    Parameters:
        image (ndarray): Input grayscale image of shape (n, m, 1).
        n_segments (int): Number of superpixels to generate.
        compactness (float): Compactness parameter for SLIC algorithm.

    Returns:
        tuple:
            - ndarray: Segmentation labels array of shape (n, m).
            - ndarray: Grayscale image with segmentation overlay.
    """
    image_2d = image[:, :, 0]
    labels = slic(image_2d, n_segments=n_segments, compactness=compactness, start_label=1, channel_axis=None)
    segmented_image = label2rgb(labels, image_2d, kind='avg', bg_label=0, alpha=0.4)
    return labels, segmented_image


def display_results(original_image, segmented_image):
    """
    Displays the original image and the segmented image side by side.

    Parameters:
        original_image (ndarray): Original grayscale image of shape (n, m, 1).
        segmented_image (ndarray): Segmented grayscale image.
    """
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(original_image[:, :, 0], cmap='gray')
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(segmented_image, cmap='gray')
    plt.title("SLIC Segmentation")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SLIC Segmentation on Grayscale Image")
    parser.add_argument("filepath", type=str, help="Path to the grayscale image file")
    parser.add_argument("--n_segments", type=int, default=2, help="Number of superpixels (default: 100)")
    parser.add_argument("--compactness", type=float, default=0.1, help="Compactness parameter (default: 10)")

    args = parser.parse_args()

    image = read_grayscale_image(args.filepath)

    labels, segmented_image = slic_segmentation(
        image,
        n_segments=args.n_segments,
        compactness=args.compactness
    )

    display_results(image, segmented_image)
