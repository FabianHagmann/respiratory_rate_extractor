from datetime import datetime

import cv2
import numpy as np

import analysis


def optical_flow_difference(image_series: [np.ndarray], roi_mask=None, smoothing_factor=None) -> [float]:
    """
    Calculate optical flow magnitude differences between consecutive frames,
    considering only regions specified by the ROI mask.

    Parameters:
        image_series (list of np.ndarray): List of images with dimensions (m, n, 3).
        roi_mask (np.ndarray): ROI mask with dimensions (m, n, 1) or (m, n).

    Returns:
        list of float: Average optical flow magnitude values for each pair of consecutive frames.
    """
    start_time = datetime.now()

    if roi_mask is None:
        roi_mask = np.ones(image_series[0].shape[:2], dtype=bool)
    else:
        roi_mask = roi_mask.squeeze()

    prev_gray = cv2.cvtColor(image_series[0], cv2.COLOR_RGB2GRAY)
    changes = []

    for i in range(1, len(image_series)):
        curr_gray = cv2.cvtColor(image_series[i], cv2.COLOR_RGB2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        magnitude_roi = magnitude * roi_mask

        roi_pixel_count = np.sum(roi_mask)
        mean_magnitude = np.sum(magnitude_roi) / roi_pixel_count if roi_pixel_count > 0 else 0.0

        changes.append(mean_magnitude)
        prev_gray = curr_gray
    analysis.smooth_time_series(changes, sigma=1 if smoothing_factor is None else smoothing_factor)

    print(f"flow_magnitude: {datetime.now() - start_time}")
    return analysis.scale_to_unit_interval(changes)


def optical_flow_difference_greyscale(image_series: [np.ndarray], roi_mask=None, smoothing_factor=None) -> [float]:
    """
    Calculate optical flow magnitude differences between consecutive frames,
    considering only regions specified by the ROI mask.

    Parameters:
        image_series (list of np.ndarray): List of greyscale images with dimensions (m, n, 1).
        roi_mask (np.ndarray): ROI mask with dimensions (m, n, 1) or (m, n).

    Returns:
        list of float: Average optical flow magnitude values for each pair of consecutive frames.
    """
    start_time = datetime.now()

    if roi_mask is None:
        roi_mask = np.ones(image_series[0].shape[:2], dtype=bool)
    else:
        roi_mask = roi_mask.squeeze()

    prev_gray = image_series[0].squeeze()
    changes = []

    for i in range(1, len(image_series)):
        curr_gray = image_series[i].squeeze()
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        magnitude_roi = magnitude * roi_mask

        roi_pixel_count = np.sum(roi_mask)
        mean_magnitude = np.sum(magnitude_roi) / roi_pixel_count if roi_pixel_count > 0 else 0.0

        changes.append(mean_magnitude)
        prev_gray = curr_gray
    analysis.smooth_time_series(changes, sigma=1 if smoothing_factor is None else smoothing_factor)

    print(f"flow_magnitude_greyscale: {datetime.now() - start_time}")
    return analysis.scale_to_unit_interval(changes)
