import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.segmentation import active_contour
from skimage.filters import gaussian

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

def preprocess_image(image, sigma=2):
    """
    Applies preprocessing to smooth the image.

    Parameters:
        image (ndarray): Input grayscale image as an n*m*1 ndarray.
        sigma (float): The standard deviation for Gaussian smoothing.

    Returns:
        ndarray: Smoothed image.
    """
    return gaussian(image[:, :, 0], sigma=sigma)

def initialize_snake(image):
    """
    Initializes a circular snake around the center of the image.

    Parameters:
        image (ndarray): Input grayscale image as an n*m*1 ndarray.

    Returns:
        ndarray: Initial snake coordinates.
    """
    h, w = image.shape[:2]
    s = np.linspace(0, 2 * np.pi, 400)
    x = (w - 1) * (0.5 + 0.5 * np.cos(s))
    y = (h - 1) * (0.5 + 0.5 * np.sin(s))
    return np.array([x, y]).T

def active_contour_segmentation(image, snake, alpha=0.1, beta=0.1, gamma=0.01):
    """
    Performs active contour segmentation on the input image.

    Parameters:
        image (ndarray): Input grayscale image as an n*m*1 ndarray.
        snake (ndarray): Initial snake coordinates.
        alpha (float): Snake length shape parameter.
        beta (float): Snake smoothness shape parameter.
        gamma (float): Time step in the energy minimization.
        max_iterations (int): Maximum iterations for the optimization.

    Returns:
        ndarray: Final snake coordinates after convergence.
    """
    return active_contour(image, snake, alpha=alpha, beta=beta, gamma=gamma)

def display_results(original_image, snake, final_snake):
    """
    Displays the original image with the initial and final snakes overlaid.

    Parameters:
        original_image (ndarray): Original grayscale image as an n*m*1 ndarray.
        snake (ndarray): Initial snake coordinates.
        final_snake (ndarray): Final snake coordinates after segmentation.
    """
    plt.figure(figsize=(8, 8))
    plt.imshow(original_image[:, :, 0], cmap='gray')
    plt.plot(snake[:, 0], snake[:, 1], '-r', lw=2, label='Initial Snake')
    plt.plot(final_snake[:, 0], final_snake[:, 1], '-b', lw=2, label='Final Snake')
    plt.legend()
    plt.axis("off")
    plt.title("Active Contour Segmentation")
    plt.show()


if __name__ == "__main__":
    import argparse

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Active Contours Segmentation on Grayscale Image")
    parser.add_argument("filepath", type=str, help="Path to the grayscale PNG image")
    parser.add_argument("--sigma", type=float, default=2, help="Gaussian smoothing sigma (default: 2)")
    parser.add_argument("--alpha", type=float, default=0.02, help="Snake length shape parameter (default: 0.1)")
    parser.add_argument("--beta", type=float, default=0.1, help="Snake smoothness shape parameter (default: 0.1)")
    parser.add_argument("--gamma", type=float, default=0.01, help="Time step in energy minimization (default: 0.01)")
    parser.add_argument("--max_iterations", type=int, default=2500, help="Maximum iterations for optimization (default: 2500)")

    args = parser.parse_args()

    # Read the image and preprocess
    image = read_image_to_ndarray(args.filepath)
    smoothed_image = preprocess_image(image, sigma=args.sigma)

    # Initialize snake and perform active contour segmentation
    snake = initialize_snake(image)
    final_snake = active_contour_segmentation(
        smoothed_image,
        snake,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
    )

    # Display the results
    display_results(image, snake, final_snake)
