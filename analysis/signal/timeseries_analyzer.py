from datetime import time

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks


def scale_to_unit_interval(series: [float]) -> [float]:
    """
    Scale a series of floats to the interval [0, 1].

    :param series: List of values to be scaled.
    :return: Scaled list
    """
    min_val = min(series)
    max_val = max(series)

    if max_val == min_val:
        return [0.5] * len(series)

    scaled_arr = [(x - min_val) / (max_val - min_val) for x in series]
    return scaled_arr


def smooth_time_series(series: [float], sigma: int) -> [float]:
    """
    Smooths the input data using a Gaussian filter.

    :param series: List of float values to be smoothed.
    :param sigma: Standard deviation of the Gaussian filter.
    :return: The smoothed data as a list.
    """
    return gaussian_filter1d(series, sigma=sigma)


def combine_time_series(series: [[float]], smoothing_factor=1) -> [float]:
    """
    Combine multiple time series by averaging them and then smoothing the result.

    :param series: list of [float] all having the same length
    :param smoothing_factor: factor with witch smooth_time_series is called for the final time_series
    :return: averaged and smoothed combined time_series
    """

    combined = [sum(x) / len(x) for x in zip(*series)]
    return smooth_time_series(combined, smoothing_factor)


def find_local_extrema_idxs(series: [float]):
    """
    Find the local maxima and minima of a time series.
    :param series: time series
    :return: 2 lists of indices, one for the peaks and one for the valleys
    """
    time_series = np.array(series)
    peaks, _ = find_peaks(time_series)
    valleys, _ = find_peaks(-time_series)

    return peaks, valleys


def calculate_respiratory_rate(peeks: [int], valleys: [int], duration: time) -> float | None:
    """
    Calculate the respiratory rate from the number of peeks and valleys and the duration of the measurement.
    :param peeks: peak indices
    :param valleys: valley indices
    :param duration: duration of the measurement
    :return: respiratory rate
    """
    if duration is None:
        return None
    duration_ms = duration.hour * 3600000 + duration.minute * 60000 + duration.second * 1000 + duration.microsecond / 1000
    num_breaths = max(len(peeks) / 2, len(valleys) / 2)
    return (60000 / duration_ms) * num_breaths
