import copy
import os.path

import numpy as np

import analysis
import preprocessing
import utils
import visualizer.sample_visualizer
from utils import SampleData

"""
Interactive development tool. All necessary values are entered by the user via the command line.
"""


def load(sample_dir: str, scale_factor=1.0) -> SampleData:
    return utils.pipeline_utils.load_data_from_directory(sample_dir, scale_factor)


def preprocess(data_sample: SampleData) -> None:
    utils.pipeline_utils.normalize_thermal_data(data_sample)
    utils.pipeline_utils.normalize_depth_data(data_sample)


def extract_roi(data_sample: SampleData, use_roi: bool, tolerance: int) -> np.ndarray | None:
    if not use_roi:
        return None

    roi_sample = copy.deepcopy(data_sample)
    utils.pipeline_utils.normalize_depth_data(roi_sample)
    return preprocessing.roi.extract_roi_flood_fill(roi_sample.depth, tolerance)


def query_hyperparameters() -> tuple[int, int, float, bool, int]:
    smoothing_factor = utils.input_utils.query_int("Smoothing factor: ", 1, 5)
    combination_smoothing_factor = utils.input_utils.query_int("Combination smoothing factor: ", 1, 5)
    scale_factor = utils.input_utils.query_float("Scale factor: ", 0.1, 1.0)
    use_roi = utils.input_utils.query_bool("Use ROI extraction: ")
    tolerance = utils.input_utils.query_int("Tolerance: ", 1, 255)
    return smoothing_factor, combination_smoothing_factor, scale_factor, use_roi, tolerance


def query_sample() -> tuple[int, int]:
    subject_number = utils.input_utils.query_int("Subject number: ", 1, 17)
    sample_idx = utils.input_utils.query_int("Sample index: ", 0, 2)
    return subject_number, sample_idx


def query_metrics() -> tuple[bool, bool, bool, bool, bool, bool]:
    use_flow_magnitude = utils.input_utils.query_bool("Use flow magnitude: ")
    use_ssim = utils.input_utils.query_bool("Use SSIM: ")
    use_pixel_diff = utils.input_utils.query_bool("Use pixel difference: ")
    use_orb = utils.input_utils.query_bool("Use ORB: ")
    use_sift = utils.input_utils.query_bool("Use SIFT: ")
    use_histogram = utils.input_utils.query_bool("Use histogram: ")
    return use_flow_magnitude, use_ssim, use_pixel_diff, use_orb, use_sift, use_histogram


def process_sample(data_sample: SampleData, smoothing_factor: int, uses_roi: bool, flow_mag: bool, ssim: bool,
                   pixel_diff: bool, orb: bool, sift: bool, histogram: bool) -> [[float]]:
    series = []

    series.extend(process_specific_sample(data_sample.rgb, "RGB", smoothing_factor, True,
                                          None if not uses_roi else data_sample.roi, flow_mag, ssim, pixel_diff, orb,
                                          sift, histogram))
    series.extend(process_specific_sample(data_sample.depth, "Depth", smoothing_factor, False,
                                          None if not uses_roi else data_sample.roi, flow_mag, ssim, pixel_diff, orb,
                                          sift, histogram))
    series.extend(process_specific_sample(data_sample.thermal, "Thermal", smoothing_factor, False,
                                          None if not uses_roi else data_sample.roi, flow_mag, ssim, pixel_diff, orb,
                                          sift, histogram))

    return series


def process_specific_sample(data: [np.ndarray], data_type: str, smoothing_factor: int, multi_dim: bool,
                            roi: [np.ndarray], flow_mag: bool, ssim: bool, pixel_diff: bool, orb: bool,
                            sift: bool, histogram: bool) -> [[float]]:
    series = []

    if sift:
        series.append((analysis.sift_difference(data, smoothing_factor=smoothing_factor, roi_mask=roi),
                       f"SIFT {data_type}", 'blue'))
    if orb:
        series.append(
            (analysis.orb_difference(data, smoothing_factor=smoothing_factor, roi_mask=roi), f"ORB {data_type}", 'red'))

    if multi_dim:
        if pixel_diff:
            series.append((analysis.pixel_difference(data, smoothing_factor=smoothing_factor, roi_mask=roi),
                           f"Pixel Diff. {data_type}", 'green'))
        if flow_mag:
            series.append((analysis.optical_flow_difference(data, roi, smoothing_factor=smoothing_factor),
                           f"Optical Flow {data_type}", 'magenta'))
        if ssim:
            series.append((analysis.ssim_difference(data, smoothing_factor=smoothing_factor, roi_mask=roi),
                           f"SSIM {data_type}", 'cyan'))
        if histogram:
            series.append((analysis.histogram_difference(data, smoothing_factor=smoothing_factor, roi_mask=roi),
                           f"Histogram Diff. {data_type}", 'yellow'))
    else:
        if pixel_diff:
            series.append((analysis.pixel_difference_greyscale(data, smoothing_factor=smoothing_factor, roi_mask=roi),
                           f"Pixel Diff. {data_type}", 'green'))
        if flow_mag:
            series.append((analysis.optical_flow_difference_greyscale(data, roi, smoothing_factor=smoothing_factor),
                           f"Optical Flow {data_type}", 'magenta'))
        if ssim:
            series.append((analysis.ssim_difference_greyscale(data, smoothing_factor=smoothing_factor, roi_mask=roi),
                           f"SSIM {data_type}", 'cyan'))
        if histogram:
            series.append((analysis.histogram_difference_greyscale(data, smoothing_factor=smoothing_factor,
                                                                   roi_mask=roi), f"Histogram Diff. {data_type}",
                           'yellow'))

    return series


if __name__ == "__main__":
    # query parameters
    smoothing_factor, combination_smoothing_factor, scale_factor, use_roi, tolerance = query_hyperparameters()
    subject_number, sample_idx = query_sample()
    use_flow_magnitude, use_ssim, use_pixel_diff, use_orb, use_sift, use_histogram = query_metrics()

    # load data
    dataset_base_dir = os.path.join("..", "final")
    sample_dir = utils.find_sample_dir(subject_number, sample_idx, dataset_base_dir)
    sample = load(sample_dir, scale_factor)

    # preprocessing
    roi_mask = extract_roi(sample, use_roi, tolerance)
    preprocess(sample)
    sample.roi = roi_mask

    # signal processing
    series = process_sample(sample, smoothing_factor, use_roi, use_flow_magnitude, use_ssim, use_pixel_diff, use_orb,
                            use_sift, use_histogram)
    series.append((analysis.combine_time_series([item[0] for item in series],
                                                smoothing_factor=combination_smoothing_factor), '_Combined', 'black'))
    peeks, valleys = analysis.find_local_extrema_idxs(series[len(series) - 1][0])

    # visualization
    annotation_frames = visualizer.find_annotation_frames_for_sample(sample_dir)
    breaths_out, breaths_in, annotated_rate, duration = utils.input_utils.load_annotation_data(sample_dir)
    calculated_rate = analysis.calculate_respiratory_rate(peeks, valleys, duration)
    roi_viewer = visualizer.InteractiveSampleViewer(sample.depth, roi_mask, series, annotation_frames, peeks, valleys,
                                                    breaths_out, breaths_in, annotated_rate, calculated_rate, duration)
