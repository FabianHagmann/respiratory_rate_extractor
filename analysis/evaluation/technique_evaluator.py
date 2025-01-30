import copy
import os
import sys
from datetime import datetime

import numpy as np

import analysis
import preprocessing
import utils
from utils import SampleData

"""
Evaluate all combinations of methods for a fixed set of hyperparameters.
"""

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


def write_results(result_file_path: str, flow_mag: bool, ssim: bool, pixel_diff: bool, orb: bool, sift: bool,
                  avg_diff: float):
    sanitized_path = result_file_path.replace(':', '-')

    with open(sanitized_path, 'a') as f:
        # if file is empty write header line with names
        if os.stat(sanitized_path).st_size == 0:
            f.write("Flow Magnitude;SSIM;Pixel Difference;ORB;SIFT;Avg.Diff.\n")
        # write csv results
        f.write(f"{flow_mag};{ssim};{pixel_diff};{orb};{sift};{avg_diff}\n")


def evaluate(smoothing_factor: int, combi_smoothing_factor: int, down_sample_factor: int, roi: bool, flow_mag: bool,
             ssim: bool, pixel_diff: bool, orb: bool, sift: bool):
    results_base_dir = os.path.join("..", "..", "results", "difference_metrics")
    dataset_base_dir = os.path.join("..", "..", "final")

    result_file_name = (
        f"metrics-{smoothing_factor}-{combi_smoothing_factor}-{1.0 / down_sample_factor}"
        f"{"-ROI" if roi else ""}.csv")
    result_file_path = os.path.join(results_base_dir, result_file_name)
    differences = []

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
                                                combi_smoothing_factor, 1.0 / down_sample_factor, flow_mag, ssim,
                                                pixel_diff, orb, sift)
            peeks, valleys = analysis.find_local_extrema_idxs(result_series)
            _, _, annotated_rate, duration = utils.input_utils.load_annotation_data(
                sample_dir)
            calculated_rate = analysis.calculate_respiratory_rate(peeks, valleys, duration)
            differences.append(abs(annotated_rate - calculated_rate))

    difference_avg = sum(differences) / len(differences)
    write_results(result_file_path, flow_mag, ssim, pixel_diff, orb, sift, difference_avg)


if __name__ == "__main__":
    smoothing_factor = 1
    combi_smoothing_factor = 2
    down_sample_factor = 3
    use_roi = False

    for use_flow_magnitude in range(0, 2):
        for use_ssim in range(0, 2):
            for use_pixel_diff in range(0, 2):
                for use_orb in range(0, 2):
                    for use_sift in range(0, 2):
                        if (use_flow_magnitude + use_ssim + use_pixel_diff + use_orb + use_sift) == 0:
                            continue
                        print(f"{datetime.now()}: Starting evaluation for flow_magnitude "
                              f"{use_flow_magnitude}, ssim {use_ssim}, pixel_diff {use_pixel_diff}, "
                              f"orb {use_orb}, sift {use_sift}")
                        flow_mag = bool(use_flow_magnitude)
                        ssim = bool(use_ssim)
                        pixel_diff = bool(use_pixel_diff)
                        orb = bool(use_orb)
                        sift = bool(use_sift)
                        evaluate(smoothing_factor, combi_smoothing_factor, down_sample_factor, use_roi, flow_mag, ssim,
                                 pixel_diff, orb, sift)
