from datetime import datetime

import numpy as np
from skimage.metrics import structural_similarity as ssim

import analysis


def ssim_difference(image_series: [np.ndarray], roi_mask=None, smoothing_factor=None) -> [float]:
    """
    Calculate the Structural Similarity Index (SSIM) between consecutive frames,
    considering only regions specified by the ROI mask.

    Parameters:
        image_series (list of np.ndarray): List of images with dimensions (m, n, 3).
        roi_mask (np.ndarray): ROI mask with dimensions (m, n, 1) or (m, n).
                               Non-zero values indicate the region of interest.

    Returns:
        list of float: 1 - SSIM values for each pair of consecutive frames (higher = more difference).
    """
    start_time = datetime.now()

    if roi_mask is None:
        roi_mask = np.ones(image_series[0].shape[:2], dtype=bool)
    else:
        roi_mask = roi_mask.squeeze()

    changes = []
    for i in range(1, len(image_series)):
        score, ssim_map = ssim(
            image_series[i - 1], image_series[i], full=True, multichannel=True, channel_axis=2
        )
        ssim_map_roi = ssim_map * roi_mask[..., None]

        roi_pixel_count = np.sum(roi_mask)
        if roi_pixel_count > 0:
            mean_ssim = np.sum(ssim_map_roi) / (roi_pixel_count * 3)
        else:
            mean_ssim = 1.0

        changes.append(1 - mean_ssim)
    analysis.smooth_time_series(changes, sigma=1 if smoothing_factor is None else smoothing_factor)

    print(f"ssim: {datetime.now() - start_time}")
    return analysis.scale_to_unit_interval(changes)


def ssim_difference_greyscale(image_series: [np.ndarray], roi_mask=None, smoothing_factor=None) -> [float]:
    """
    Calculate the Structural Similarity Index (SSIM) between consecutive frames,
    considering only regions specified by the ROI mask.

    Parameters:
        image_series (list of np.ndarray): List of images with dimensions (m, n, 1).
        roi_mask (np.ndarray): ROI mask with dimensions (m, n, 1) or (m, n).
                               Non-zero values indicate the region of interest.

    Returns:
        list of float: 1 - SSIM values for each pair of consecutive frames (higher = more difference).
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
        score, ssim_map = ssim(prev_gray, curr_gray, full=True, multichannel=False)

        ssim_map_roi = ssim_map * roi_mask  # No extra dimension needed here

        roi_pixel_count = np.sum(roi_mask)
        if roi_pixel_count > 0:
            mean_ssim = np.sum(ssim_map_roi) / roi_pixel_count
        else:
            mean_ssim = 1.0

        changes.append(1 - mean_ssim)

    analysis.smooth_time_series(changes, sigma=1 if smoothing_factor is None else smoothing_factor)

    print(f"ssim: {datetime.now() - start_time}")
    return analysis.scale_to_unit_interval(changes)
