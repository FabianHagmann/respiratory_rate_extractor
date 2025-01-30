from datetime import datetime

import cv2
import numpy as np

import analysis


def sift_difference(image_series: [np.ndarray], roi_mask=None, smoothing_factor=None) -> [float]:
    """
    Calculate differences using SIFT feature matching between consecutive frames,
    considering only regions specified by the ROI mask.

    Parameters:
        image_series (list of np.ndarray): List of images with dimensions (m, n, 3).
        roi_mask (np.ndarray): ROI mask with dimensions (m, n, 1) or (m, n).

    Returns:
        list of float: Fraction of unmatched features for each pair of consecutive frames.
    """

    start_time = datetime.now()

    if roi_mask is None:
        roi_mask = np.ones(image_series[0].shape[:2], dtype=bool)
    else:
        roi_mask = roi_mask.squeeze()

    sift = cv2.SIFT_create()
    changes = []

    for i in range(1, len(image_series)):
        mask1 = (roi_mask * 255).astype(np.uint8)
        mask2 = (roi_mask * 255).astype(np.uint8)

        kp1, des1 = sift.detectAndCompute(image_series[i-1], mask1)
        kp2, des2 = sift.detectAndCompute(image_series[i], mask2)

        if des1 is not None and des2 is not None:
            bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
            matches = bf.match(des1, des2)
            unmatched = 1 - len(matches) / max(len(kp1), len(kp2)) if max(len(kp1), len(kp2)) > 0 else 1.0
        else:
            unmatched = 1.0

        changes.append(unmatched)
    analysis.smooth_time_series(changes, sigma=1 if smoothing_factor is None else smoothing_factor)

    print(f"sift: {datetime.now() - start_time}")
    return analysis.scale_to_unit_interval(changes)
