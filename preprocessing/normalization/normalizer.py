import os
import shutil
from tkinter import Tk
from tkinter.filedialog import askdirectory

import cv2
import numpy as np


def normalize(image: np.ndarray) -> np.ndarray:
    """
    Normalize pixel values in the image to range 0-255.
    :param image: grayscale image
    :return: normalized grayscale image
    """
    return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)


def normalize_by_path(image_path):
    """
    Normalize pixel values in the image to range 0-255.
    :param image_path: path to grayscale image
    :return: normalized grayscale image
    """
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    return normalize(image)


def process_files(input_dir, output_dir):
    """Process files from input_dir and save them to output_dir."""
    for root, _, files in os.walk(input_dir):
        for file in files:
            input_path = os.path.join(root, file)

            relative_path = os.path.relpath(root, input_dir)
            output_path_dir = os.path.join(output_dir, relative_path)
            os.makedirs(output_path_dir, exist_ok=True)

            output_path = os.path.join(output_path_dir, file)

            try:
                if (file.startswith("thermal_") and file.endswith(".png")) or \
                        (file.startswith("depth_") and file.endswith(".png")):
                    norm_image = normalize(input_path)
                    cv2.imwrite(output_path, norm_image)

                elif file.endswith(".txt") or file.endswith(".csv"):
                    shutil.copy2(input_path, output_path)

                else:
                    shutil.copy2(input_path, output_path)
            except Exception as e:
                print(f"Error processing file {input_path}: {e}")


if __name__ == "__main__":
    Tk().withdraw()
    print("Select the input directory:")
    input_dir = askdirectory(title="Select Input Directory")

    if not input_dir:
        print("No input directory selected. Exiting.")
        exit()

    print("Select the output directory:")
    output_dir = askdirectory(title="Select Output Directory")

    if not output_dir:
        print("No output directory selected. Exiting.")
        exit()

    # Process the files
    process_files(input_dir, output_dir)
    print("Processing completed.")
