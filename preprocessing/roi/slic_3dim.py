import cv2
import cv2
import matplotlib.pyplot as plt
from skimage.color import label2rgb
from skimage.segmentation import slic


def read_rgb_image(filepath):
    """
    Reads an RGB image from a file.

    Parameters:
        filepath (str): Path to the RGB image file.

    Returns:
        ndarray: RGB image as an array of shape (n, m, 3).
    """
    image = cv2.imread(filepath)
    if image is None:
        raise FileNotFoundError(f"Image file not found: {filepath}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def slic_segmentation(image, n_segments=100, compactness=10):
    """
    Performs SLIC (Simple Linear Iterative Clustering) segmentation on an RGB image.

    Parameters:
        image (ndarray): Input RGB image of shape (n, m, 3).
        n_segments (int): Number of superpixels to generate.
        compactness (float): Compactness parameter for SLIC algorithm.

    Returns:
        tuple:
            - ndarray: Segmentation labels array of shape (n, m).
            - ndarray: RGB image with segmentation overlay.
    """
    labels = slic(image, n_segments=n_segments, compactness=compactness, start_label=1)
    segmented_image = label2rgb(labels, image, kind='avg')
    return labels, segmented_image


def display_results(original_image, segmented_image):
    """
    Displays the original image and the segmented image side by side.

    Parameters:
        original_image (ndarray): Original RGB image.
        segmented_image (ndarray): Segmented RGB image.
    """
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(original_image)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(segmented_image)
    plt.title("SLIC Segmentation")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SLIC Segmentation on RGB Image")
    parser.add_argument("filepath", type=str, help="Path to the RGB image file")
    parser.add_argument("--n_segments", type=int, default=4, help="Number of superpixels (default: 100)")
    parser.add_argument("--compactness", type=float, default=0.1, help="Compactness parameter (default: 10)")

    args = parser.parse_args()

    image = read_rgb_image(args.filepath)

    labels, segmented_image = slic_segmentation(
        image,
        n_segments=args.n_segments,
        compactness=args.compactness
    )

    display_results(image, segmented_image)
