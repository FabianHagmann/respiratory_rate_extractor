import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider


def read_image_to_ndarray(filepath):
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


def flood_fill_segmentation(image, tolerance=5) -> np.ndarray:
    """
    Performs flood fill segmentation on the image with a seed point at the center.

    Parameters:
        image (ndarray): Input image as an n*m*1 ndarray.
        tolerance (int): Tolerance for flood filling.

    Returns:
        mask (ndarray): Binary mask of the segmented region.
    """
    h, w = image.shape
    seed_point = (h // 2, w // 2)

    mask = np.zeros((h + 2, w + 2), np.uint8)
    flags = 4 | cv2.FLOODFILL_MASK_ONLY | (255 << 8)

    flood_fill_image = image.copy()
    cv2.floodFill(flood_fill_image, mask, seed_point, newVal=0, loDiff=(tolerance,), upDiff=(tolerance,), flags=flags)

    return mask[1:-1, 1:-1]


def display_images_with_slider(original_image):
    """
    Displays the original image with an overlay of the resulting mask and a slider to adjust tolerance.

    Parameters:
        original_image (ndarray): Original grayscale image as an n*m*1 ndarray.
    """

    def update(val):
        tolerance = slider.val
        mask = flood_fill_segmentation(original_image, tolerance=int(tolerance))
        overlay = np.copy(original_image[:, :, 0])
        overlay[mask == 255] = 255
        ax1.imshow(overlay, cmap='gray')
        fig.canvas.draw_idle()

    fig, ax1 = plt.subplots(1, 1, figsize=(8, 8))
    plt.subplots_adjust(bottom=0.25)

    ax1.set_title("Original Image with Segmentation Overlay")
    initial_tolerance = 100
    mask = flood_fill_segmentation(original_image, tolerance=initial_tolerance)
    overlay = np.copy(original_image[:, :, 0])
    overlay[mask == 255] = 255
    ax1.imshow(overlay, cmap='gray')
    ax1.axis("off")

    ax_slider = plt.axes([0.2, 0.1, 0.65, 0.03])
    slider = Slider(ax_slider, 'Tolerance', 0, 255, valinit=initial_tolerance, valstep=1)
    slider.on_changed(update)

    plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Flood Fill Segmentation on Grayscale Image")
    parser.add_argument("filepath", type=str, help="Path to the grayscale PNG image")

    args = parser.parse_args()

    image = read_image_to_ndarray(args.filepath)

    display_images_with_slider(image)
