from datetime import datetime

import numpy as np

import analysis


def histogram_difference(image_series: [np.ndarray], roi_mask=None, bins=256, smoothing_factor=None) -> [float]:
    """
    Calculate histogram differences between consecutive frames,
    considering only regions specified by the ROI mask.

    Parameters:
        image_series (list of np.ndarray): List of images with dimensions (m, n, 3).
        roi_mask (np.ndarray): ROI mask with dimensions (m, n, 1) or (m, n).
        bins (int): Number of bins for the histograms.

    Returns:
        list of float: Histogram difference values for each pair of consecutive frames.
    """
    start_time = datetime.now()

    if roi_mask is None:
        roi_mask = np.ones(image_series[0].shape[:2], dtype=bool)
    else:
        roi_mask = roi_mask.squeeze()

    changes = []
    for i in range(1, len(image_series)):
        masked_image1 = image_series[i-1] * roi_mask[..., None]
        masked_image2 = image_series[i] * roi_mask[..., None]

        hist1 = np.histogramdd(masked_image1.reshape(-1, 3), bins=bins, range=[(0, 256)] * 3)[0]
        hist2 = np.histogramdd(masked_image2.reshape(-1, 3), bins=bins, range=[(0, 256)] * 3)[0]

        hist1 /= hist1.sum()
        hist2 /= hist2.sum()

        diff = np.linalg.norm(hist1 - hist2)
        changes.append(diff)
    analysis.smooth_time_series(changes, sigma=1 if smoothing_factor is None else smoothing_factor)

    print(f"histogram comparison: {datetime.now() - start_time}")
    return analysis.scale_to_unit_interval(changes)


def histogram_difference_greyscale(image_series: [np.ndarray], roi_mask=None, bins=256, smoothing_factor=None) -> [float]:
    """
    Calculate histogram differences between consecutive frames,
    considering only regions specified by the ROI mask.

    Parameters:
        image_series (list of np.ndarray): List of images with dimensions (m, n, 1).
        roi_mask (np.ndarray): ROI mask with dimensions (m, n, 1) or (m, n).
        bins (int): Number of bins for the histograms.

    Returns:
        list of float: Histogram difference values for each pair of consecutive frames.
    """
    start_time = datetime.now()

    if roi_mask is None:
        roi_mask = np.ones(image_series[0].shape[:2], dtype=bool)
    else:
        roi_mask = roi_mask.squeeze()

    changes = []
    for i in range(1, len(image_series)):
        # Ensure 2D images by squeezing
        masked_image1 = image_series[i - 1].squeeze() * roi_mask
        masked_image2 = image_series[i].squeeze() * roi_mask

        # Compute histograms for grayscale images
        hist1 = np.histogram(masked_image1[roi_mask].flatten(), bins=bins, range=(0, 256))[0]
        hist2 = np.histogram(masked_image2[roi_mask].flatten(), bins=bins, range=(0, 256))[0]

        # Normalize histograms
        hist1 = hist1 / hist1.sum() if hist1.sum() > 0 else hist1
        hist2 = hist2 / hist2.sum() if hist2.sum() > 0 else hist2

        # Compute histogram difference
        diff = np.linalg.norm(hist1 - hist2)
        changes.append(diff)

    # Apply smoothing
    analysis.smooth_time_series(changes, sigma=1 if smoothing_factor is None else smoothing_factor)

    print(f"histogram comparison: {datetime.now() - start_time}")
    return analysis.scale_to_unit_interval(changes)
