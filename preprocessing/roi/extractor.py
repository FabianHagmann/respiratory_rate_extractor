import numpy as np

from preprocessing.roi.flood_fill_1dim import flood_fill_segmentation


def extract_roi_flood_fill(data: [np.ndarray], tolerance: int = 100):
    """
    Extracts the region of interest from all depth images of the data sample using flood fill segmentation.
    :param data: (ndarray): Input sample data with at least one depth image of dimensions n*m*1 ndarray.
    :param tolerance:  Tolerance for flood filling.
    :return: mask (ndarray): Binary mask of the segmented region.
    """

    pixel_counter = np.zeros_like(data[0], dtype=int)

    for image in data:
        roi_mask = flood_fill_segmentation(image, tolerance)
        pixel_counter += roi_mask

    final_mask = (pixel_counter >= 3).astype(np.uint8)

    return final_mask
