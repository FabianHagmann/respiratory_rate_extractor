import copy
import os
import sys
from datetime import datetime, time

import numpy as np

import analysis
import preprocessing
import utils
from utils import SampleData

"""
Evaluate different combinations of hyperparameters with fixed methods.
All hyperparameters are iterated within reasonable rangers
"""


def load(sample_dir: str, down_sampling_modifier: float) -> SampleData:
    data = utils.pipeline_utils.load_data_from_directory(sample_dir, down_sampling_modifier)
    print(f"Data sample loaded\n{data}")
    return data


def preprocess(data_sample: SampleData) -> None:
    utils.pipeline_utils.normalize_thermal_data(data_sample)
    utils.pipeline_utils.normalize_depth_data(data_sample)


def extract_roi(data_sample: SampleData) -> np.ndarray:
    roi_sample = copy.deepcopy(data_sample)
    utils.pipeline_utils.normalize_depth_data(roi_sample)
    return preprocessing.roi.extract_roi_flood_fill(roi_sample.depth, 25)


def process_sample(data_sample: SampleData, smoothing_factor: int, uses_roi: bool) -> [[float]]:
    series = []

    series.extend(process_specific_sample(data_sample.rgb, smoothing_factor, True,
                                          None if not uses_roi else data_sample.roi))
    series.extend(process_specific_sample(data_sample.depth, smoothing_factor, False,
                                          None if not uses_roi else data_sample.roi))
    series.extend(process_specific_sample(data_sample.thermal, smoothing_factor, False,
                                          None if not uses_roi else data_sample.roi))

    return series


def process_specific_sample(data: [np.ndarray], smoothing_factor: int, multi_dim: bool,
                            roi: [np.ndarray]) -> [[float]]:
    series = []

    series.append(analysis.sift_difference(data, smoothing_factor=smoothing_factor, roi_mask=roi))
    series.append(analysis.orb_difference(data, smoothing_factor=smoothing_factor, roi_mask=roi))

    if multi_dim:
        series.append(analysis.pixel_difference(data, smoothing_factor=smoothing_factor, roi_mask=roi))
        series.append(analysis.optical_flow_difference(data, roi, smoothing_factor=smoothing_factor))
        series.append(analysis.ssim_difference(data, smoothing_factor=smoothing_factor, roi_mask=roi))
        # series.append(analysis.histogram_difference(data, smoothing_factor=smoothing_factor, roi_mask=roi))
    else:
        series.append(analysis.pixel_difference_greyscale(data, smoothing_factor=smoothing_factor, roi_mask=roi))
        series.append(analysis.optical_flow_difference_greyscale(data, roi, smoothing_factor=smoothing_factor))
        series.append(analysis.ssim_difference_greyscale(data, smoothing_factor=smoothing_factor, roi_mask=roi))
        # series.append(analysis.histogram_difference_greyscale(data, smoothing_factor=smoothing_factor, roi_mask=roi))

    return series


def evaluate_sample_dir(subject_number: int, sample_idx: int, roi: bool, sample_dir_path: str, smoothing_factor: int,
                        combination_smoothing_factor: int, down_sampling_modifier: float) -> [float]:
    print(
        f"{subject_number}:{sample_idx} - Evaluation started for subject-number {subject_number}, sample-index {sample_idx}")

    # load data
    print(f"{subject_number}:{sample_idx} - Loading sample data")
    sample = load(sample_dir_path, down_sampling_modifier)

    # preprocessing
    print(f"{subject_number}:{sample_idx} - Preprocessing sample data")
    roi_mask = extract_roi(sample) if roi else None
    preprocess(sample)
    sample.roi = roi_mask

    # signal processing
    print(f"{subject_number}:{sample_idx} - Starting Signal Processing")
    series = process_sample(sample, smoothing_factor, roi)
    return analysis.combine_time_series([item for item in series],
                                        smoothing_factor=combination_smoothing_factor)


def write_results(result_file_path: str, subject_number: int, sample_idx: int, annotated_breaths_out: int,
                  annotated_breaths_in: int, peeks: int, valleys: int, annotated_rate: float, calculated_rate: float,
                  duration: time):
    sanitized_path = result_file_path.replace(':', '-')

    with open(sanitized_path, 'a') as f:
        # if file is empty write header line with names
        if os.stat(sanitized_path).st_size == 0:
            f.write("Subject Number;Sample Index;Annotated Breaths Out;Annotated Breaths In;Peeks,Valleys;"
                    "Annotated Rate;Calculated Rate;Duration\n")
        # write csv results
        f.write(f"{subject_number};{sample_idx};{annotated_breaths_out};{annotated_breaths_in};{peeks},{valleys};"
                f"{annotated_rate};{calculated_rate};{duration}\n")


def evaluate(smoothing_factor: int, combi_smoothing_factor: int, down_sample_factor: int, roi: bool):
    results_base_dir = os.path.join("..", "..", "results")
    dataset_base_dir = os.path.join("..", "..", "final")
    result_file_name = (
        f"{datetime.now()}-{smoothing_factor}-{combi_smoothing_factor}-{1.0 / down_sample_factor}"
        f"{"-ROI" if roi else ""}.csv")
    result_file_path = os.path.join(results_base_dir, result_file_name)

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
                                                combi_smoothing_factor, 1.0 / down_sample_factor)
            peeks, valleys = analysis.find_local_extrema_idxs(result_series)
            annotated_breaths_out, annotated_breaths_in, annotated_rate, duration = utils.input_utils.load_annotation_data(
                sample_dir)
            calculated_rate = analysis.calculate_respiratory_rate(peeks, valleys, duration)
            write_results(result_file_path, subject_number, sample_idx, annotated_breaths_out,
                          annotated_breaths_in,
                          len(peeks), len(valleys), annotated_rate, calculated_rate, duration)


if __name__ == "__main__":
    for smoothing_factor in range(1, 5):
        for combi_smoothing_factor in range(2, 5):
            for down_sample_factor in range(2, 6):
                print(
                    f"{datetime.now()}: Starting evaluation for smoothing factor {smoothing_factor}, combination smoothing factor "
                    f"{combi_smoothing_factor}, down sample factor {down_sample_factor}")
                evaluate(smoothing_factor, combi_smoothing_factor, down_sample_factor, True)
                evaluate(smoothing_factor, combi_smoothing_factor, down_sample_factor, False)
