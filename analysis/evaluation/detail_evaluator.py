import copy
import os
import sys

import numpy as np

import analysis
import preprocessing
import utils
from utils import SampleData

"""
Evaluate the dataset with a specific combination of hyperparameters and methods.
They can be set below
"""

def evaluate(smoothing_factor: int, combination_smoothing_factor: int, scale_factor: float, roi: bool, tolerance: int,
             dataset_base_dir: str, results_file: str, use_flow_magnitude: bool, use_ssim: bool, use_pixel_diff: bool,
             use_orb: bool, use_sift: bool, use_histogram: bool):
    for subject_dir_name in os.listdir(dataset_base_dir):
        if not os.path.isdir(os.path.join(dataset_base_dir, subject_dir_name)):
            continue
        subject_dir = os.path.join(dataset_base_dir, subject_dir_name)
        subject_number = int(subject_dir_name.split('-')[0].strip())

        sample_idx = -1
        for sample_dir_name in os.listdir(subject_dir):
            if not os.path.isdir(os.path.join(subject_dir, sample_dir_name)):
                continue
            sample_dir = os.path.join(subject_dir, sample_dir_name)
            sample_idx += 1

            if not os.path.isfile(os.path.join(sample_dir, "annotations.txt")) or not os.path.isfile(
                    os.path.join(sample_dir, "annotations_evaluated.txt")):
                print(f"Skipping {sample_dir} as annotation files are missing", file=sys.stderr)
                continue

            result_series = evaluate_sample_dir(subject_number, sample_idx, roi, sample_dir, smoothing_factor,
                                                combination_smoothing_factor, scale_factor, use_flow_magnitude,
                                                use_ssim, use_pixel_diff, use_orb, use_sift)
            peeks, valleys = analysis.find_local_extrema_idxs(result_series)
            _, _, annotated_rate, _ = utils.input_utils.load_annotation_data(sample_dir)
            duration = utils.input_utils.find_duration(sample_dir)
            calculated_rate = analysis.calculate_respiratory_rate(peeks, valleys, duration)
            rate_difference = abs(annotated_rate - calculated_rate)

            write_result(results_file, subject_number, sample_idx, use_flow_magnitude, use_ssim, use_pixel_diff,
                         use_orb, use_sift, use_histogram, annotated_rate, calculated_rate, rate_difference, duration)


def evaluate_sample_dir(subject_number: int, sample_idx: int, roi: bool, sample_dir_path: str, smoothing_factor: int,
                        combination_smoothing_factor: int, down_sampling_modifier: float, flow_mag: bool, ssim: bool,
                        pixel_diff: bool, orb: bool, sift: bool) -> [float]:
    # load data
    sample = load(sample_dir_path, down_sampling_modifier)

    # preprocessing
    roi_mask = extract_roi(sample) if roi else None
    preprocess(sample)
    sample.roi = roi_mask

    # signal processing
    series = process_sample(sample, smoothing_factor, roi, flow_mag, ssim, pixel_diff, orb, sift)
    return analysis.combine_time_series([item for item in series],
                                        smoothing_factor=combination_smoothing_factor)


def write_result(results_file, subject_number, sample_idx, use_flow_magnitude, use_ssim, use_pixel_diff, use_orb,
                 use_sift, use_histogram, annotated_rate, calculated_rate, rate_difference, duration):
    sanitized_path = results_file.replace(':', '-')

    with open(sanitized_path, 'a') as f:
        if os.stat(sanitized_path).st_size == 0:
            f.write("Subject Number;Sample Index;Flow Magnitude;SSIM;Pixel Difference;ORB;SIFT;Histogram;Duration;"
                    "Annotated Rate;Calculated Rate;Rate Difference\n")
        # write csv results
        f.write(f"{subject_number};{sample_idx};{use_flow_magnitude};{use_ssim};{use_pixel_diff};{use_orb};{use_sift};"
                f"{use_histogram};{duration};{annotated_rate};{calculated_rate};{rate_difference}\n")


def load(sample_dir: str, down_sampling_modifier: float) -> SampleData:
    data = utils.pipeline_utils.load_data_from_directory(sample_dir, down_sampling_modifier)
    return data


def preprocess(data_sample: SampleData) -> None:
    utils.pipeline_utils.normalize_thermal_data(data_sample)
    utils.pipeline_utils.normalize_depth_data(data_sample)


def extract_roi(data_sample: SampleData) -> np.ndarray:
    roi_sample = copy.deepcopy(data_sample)
    utils.pipeline_utils.normalize_depth_data(roi_sample)
    return preprocessing.roi.extract_roi_flood_fill(roi_sample.depth, 25)


def process_sample(data_sample: SampleData, smoothing_factor: int, uses_roi: bool, flow_mag: bool, ssim: bool,
                   pixel_diff: bool, orb: bool, sift: bool) -> [[float]]:
    series = []

    series.extend(process_specific_sample(data_sample.rgb, smoothing_factor, True,
                                          None if not uses_roi else data_sample.roi, flow_mag, ssim, pixel_diff, orb,
                                          sift))
    series.extend(process_specific_sample(data_sample.depth, smoothing_factor, False,
                                          None if not uses_roi else data_sample.roi, flow_mag, ssim, pixel_diff, orb,
                                          sift))
    series.extend(process_specific_sample(data_sample.thermal, smoothing_factor, False,
                                          None if not uses_roi else data_sample.roi, flow_mag, ssim, pixel_diff, orb,
                                          sift))

    return series


def process_specific_sample(data: [np.ndarray], smoothing_factor: int, multi_dim: bool,
                            roi: [np.ndarray], flow_mag: bool, ssim: bool, pixel_diff: bool, orb: bool,
                            sift: bool) -> [[float]]:
    series = []

    if sift:
        series.append(analysis.sift_difference(data, smoothing_factor=smoothing_factor, roi_mask=roi))
    if orb:
        series.append(analysis.orb_difference(data, smoothing_factor=smoothing_factor, roi_mask=roi))

    if multi_dim:
        if pixel_diff:
            series.append(analysis.pixel_difference(data, smoothing_factor=smoothing_factor, roi_mask=roi))
        if flow_mag:
            series.append(analysis.optical_flow_difference(data, roi, smoothing_factor=smoothing_factor))
        if ssim:
            series.append(analysis.ssim_difference(data, smoothing_factor=smoothing_factor, roi_mask=roi))
    else:
        if pixel_diff:
            series.append(analysis.pixel_difference_greyscale(data, smoothing_factor=smoothing_factor, roi_mask=roi))
        if flow_mag:
            series.append(analysis.optical_flow_difference_greyscale(data, roi, smoothing_factor=smoothing_factor))
        if ssim:
            series.append(analysis.ssim_difference_greyscale(data, smoothing_factor=smoothing_factor, roi_mask=roi))

    return series


if __name__ == "__main__":
    # hyperparameters:
    smoothing_factor = 1
    combination_smoothing_factor = 2
    scale_factor = 1.0 / 3.0
    roi = False
    tolerance = 50

    # dataset parameters
    dataset_base_dir = os.path.join("..", "..", "final")
    results_file = os.path.join("..", "..", "results", "details_metrics", f"evaluation-{smoothing_factor}-"
                                                                          f"{combination_smoothing_factor}-"
                                                                          f"{scale_factor}{"-ROI" if roi else ""}-"
                                                                          f"{tolerance}.csv")

    # evaluation parameters
    use_flow_magnitude = False
    use_ssim = True
    use_pixel_diff = True
    use_orb = True
    use_sift = False
    use_histogram = False

    evaluate(smoothing_factor, combination_smoothing_factor, scale_factor, roi, tolerance, dataset_base_dir,
             results_file, use_flow_magnitude, use_ssim, use_pixel_diff, use_orb, use_sift, use_histogram)
