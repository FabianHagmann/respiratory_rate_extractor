from datetime import datetime

import numpy as np

import analysis


def pixel_difference(image_series: [np.ndarray], roi_mask=None, smoothing_factor=1) -> [float]:
    """
    Calculate pixel-wise absolute differences between consecutive frames,
    considering only regions specified by the ROI mask.

    Parameters:
        image_series (list of np.ndarray): List of images with dimensions (m, n, 3).
        roi_mask (np.ndarray): ROI mask with dimensions (m, n, 1) or (m, n).
                               Non-zero values indicate the region of interest.

    Returns:
        list of float: Normalized difference values for each pair of consecutive frames.
    """
    start_time = datetime.now()

    if roi_mask is None:
        roi_mask = np.ones(image_series[0].shape[:2], dtype=bool)
    else:
        roi_mask = roi_mask.squeeze()

    changes = []
    for i in range(1, len(image_series)):
        diff = np.abs(image_series[i] - image_series[i - 1])
        diff_roi = diff * roi_mask[..., None]

        roi_pixel_count = np.sum(roi_mask)
        if roi_pixel_count > 0:
            norm_diff = np.sum(diff_roi) / (roi_pixel_count * 3)
        else:
            norm_diff = 0.0

        changes.append(norm_diff)
    analysis.smooth_time_series(changes, sigma=smoothing_factor)

    print(f"pixel_difference: {datetime.now() - start_time}")
    return analysis.scale_to_unit_interval(changes)


def pixel_difference_greyscale(image_series: [np.ndarray], roi_mask=None, smoothing_factor=1) -> [float]:
    """
    Calculate pixel-wise absolute differences between consecutive frames,
    considering only regions specified by the ROI mask.

    Parameters:
        image_series (list of np.ndarray): List of images with dimensions (m, n, 1).
        roi_mask (np.ndarray): ROI mask with dimensions (m, n, 1) or (m, n).
                               Non-zero values indicate the region of interest.

    Returns:
        list of float: Normalized difference values for each pair of consecutive frames.
    """
    start_time = datetime.now()

    if roi_mask is None:
        roi_mask = np.ones(image_series[0].shape[:2], dtype=bool)
    else:
        roi_mask = roi_mask.squeeze()

    changes = []

    for i in range(1, len(image_series)):
        curr_gray = image_series[i].squeeze()
        prev_gray = image_series[i - 1].squeeze()
        diff = np.abs(curr_gray - prev_gray)
        diff_roi = diff * roi_mask

        roi_pixel_count = np.sum(roi_mask)
        if roi_pixel_count > 0:
            norm_diff = np.sum(diff_roi) / roi_pixel_count
        else:
            norm_diff = 0.0

        changes.append(norm_diff)

    analysis.smooth_time_series(changes, sigma=smoothing_factor)

    print(f"pixel_difference: {datetime.now() - start_time}")
    return analysis.scale_to_unit_interval(changes)
