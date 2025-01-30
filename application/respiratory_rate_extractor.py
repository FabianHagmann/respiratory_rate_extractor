import copy
from tkinter import Tk, filedialog

import matplotlib.pyplot as plt
import numpy as np

import analysis
import preprocessing
import utils
import visualizer
from utils import SampleData

"""
Final application. Implements the entire pipeline
- read sample
- preprocess sample
- evaluate sample
- display interactive visulization

Execute from root directory
-----------------------------------------------------------------------
python -m application.respiratory_rate_extractor
-----------------------------------------------------------------------

Hyperparameters and methods can be configured directly below
"""

SMOOTHING_FACTOR = 1
COMBINATION_SMOOTHING_FACTOR = 2
SCALE_FACTOR = 1.0 / 3.0
USE_ROI = False
TOLERANCE = 50

USE_FLOW_MAGNITUDE = False
USE_SSIM = True
USE_PIXEL_DIFF = True
USE_ORB = True
USE_SIFT = False
USE_HISTOGRAM = False


class RespiratoryRateExtractor:
    def __init__(self):
        """
        Initialize the application.
        """
        self.current_vline = None
        self.text_box = None
        self.mask_display = None
        self.fig = None
        self.ax_img = None
        self.im_display = None

        self.sample = None
        self.roi_mask = None
        self.time_series = []

        self.annotation_idxs = []
        self.peek_idxs = []
        self.valley_idxs = []
        self.out_breaths = 0
        self.in_breaths = 0
        self.annotated_rate = 0.0
        self.calculated_rate = 0.0
        self.duration = None

        self.current_index = 0

        self.start_application()

    def start_application(self):
        """
        Start the application by allowing the user to select a directory.
        """
        root = Tk()
        root.withdraw()

        directory = filedialog.askdirectory(title="Select a Directory")
        if directory:
            print(f"Directory selected: {directory}")
            self.extract_parameters_from_directory(directory)

            self.initialize_visualization()
        else:
            print("No directory selected. Exiting application.")

    def extract_parameters_from_directory(self, sample_dir):
        """
        Extract input parameters from the selected directory.

        :param sample_dir: Path to the selected sample directory.
        """

        self.sample = load(sample_dir, SCALE_FACTOR)
        self.duration = utils.input_utils.find_duration(sample_dir)
        self.roi_mask = extract_roi(self.sample, TOLERANCE)
        preprocess(self.sample)
        self.sample.roi = self.roi_mask

        self.time_series = process_sample(self.sample, SMOOTHING_FACTOR, USE_ROI, USE_FLOW_MAGNITUDE, USE_SSIM,
                                          USE_PIXEL_DIFF, USE_ORB, USE_SIFT, USE_HISTOGRAM)
        self.time_series.append((analysis.combine_time_series([item[0] for item in self.time_series],
                                                              smoothing_factor=COMBINATION_SMOOTHING_FACTOR),
                                 '_Combined', 'black'))
        self.peek_idxs, self.valley_idxs = analysis.find_local_extrema_idxs(
            self.time_series[len(self.time_series) - 1][0])
        try:
            self.annotation_idxs = visualizer.find_annotation_frames_for_sample(sample_dir)
            self.out_breaths, self.in_breaths, self.annotated_rate, _ = utils.input_utils.load_annotation_data(
                sample_dir)
        except FileNotFoundError:
            self.annotation_idxs = []
            self.out_breaths = 0
            self.in_breaths = 0
            self.annotated_rate = -1.0
            print("No annotation files found")
        self.calculated_rate = analysis.calculate_respiratory_rate(self.peek_idxs, self.valley_idxs, self.duration)

    def initialize_visualization(self):
        """
        Set up the visualization based on the extracted parameters.
        """

        self.fig, (self.ax_top, self.ax_ts) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})
        self.ax_img = self.ax_top
        self.fig.canvas.mpl_connect("key_press_event", self.on_key_press)

        self.im_display = self.ax_img.imshow(self.sample.depth[self.current_index], cmap="gray")
        self.mask_display = self.ax_img.imshow(
            self.roi_mask, cmap="Reds", alpha=0.3, interpolation="none"
        )
        self.ax_img.set_title(f"Image 1 of {len(self.sample.depth)}")

        self.text_box = self.ax_img.text(
            1.05, 0.5, self.get_text_info(), transform=self.ax_img.transAxes, fontsize=20,
            verticalalignment='center', horizontalalignment='left',
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5)
        )

        self.ax_ts.set_title("Time Series")
        self.ax_ts.set_xlabel("Time Index")
        self.ax_ts.set_ylabel("Value")
        for i, series in enumerate(self.time_series):
            if i == len(self.time_series) - 1:
                self.ax_ts.plot(series[0], color=series[2], label=series[1])
            else:
                self.ax_ts.plot(series[0], color=series[2], label=series[1], alpha=0.075)
        self.ax_ts.set_ylim(0, 1)
        self.ax_ts.legend()

        self.current_vline = None
        self.update_peeks_and_valleys()

        plt.tight_layout()
        plt.show()

    def get_text_info(self):
        """Generate the text information to display beside the image."""
        return (f"Annotated Rate: {-1.0 if self.annotated_rate is None else self.annotated_rate:.2f}\n"
                f"Calculated Rate: {-1.0 if self.calculated_rate is None else self.calculated_rate:.2f}\n"
                f"Duration: {-1.0 if self.duration is None else self.duration}")

    def on_key_press(self, event):
        """
        Handle key press events for navigation.

        Left arrow: Show the previous image.
        Right arrow: Show the next image.
        """
        if event.key == "right":
            self.current_index = (self.current_index + 1) % len(self.sample.depth)
        elif event.key == "left":
            self.current_index = (self.current_index - 1) % len(self.sample.depth)

        self.update_display()

    def update_display(self):
        """Update the display with the current image, ROI mask, and time series."""
        self.im_display.set_data(self.sample.depth[self.current_index])
        self.mask_display.set_data(self.roi_mask)
        self.ax_img.set_title(f"Image {self.current_index + 1} of {len(self.sample.depth)}")

        if self.current_vline:
            self.current_vline.remove()

        self.current_vline = self.ax_ts.axvline(self.current_index, color='red', linestyle='--', label="Current Image")

        self.text_box.set_text(self.get_text_info())

        self.fig.canvas.draw_idle()

    def update_peeks_and_valleys(self):
        """
        Draw vertical lines at each peek and valley on the time series plot.
        """
        self.ax_ts.axvline(self.peek_idxs[0], color='darkblue', linestyle='--', label='Peeks')
        for peek in self.peek_idxs[1:]:
            self.ax_ts.axvline(peek, color='darkblue', linestyle='--')

        self.ax_ts.axvline(self.valley_idxs[0], color='darkred', linestyle='--', label='Valleys')
        for valley in self.valley_idxs[1:]:
            self.ax_ts.axvline(valley, color='darkred', linestyle='--')

        self.fig.canvas.draw_idle()


def load(sample_dir: str, scale_factor=1.0) -> SampleData:
    return utils.pipeline_utils.load_data_from_directory(sample_dir, scale_factor)


def preprocess(data_sample: SampleData) -> None:
    utils.pipeline_utils.normalize_thermal_data(data_sample)
    utils.pipeline_utils.normalize_depth_data(data_sample)


def extract_roi(data_sample: SampleData, tolerance: int) -> np.ndarray | None:
    roi_sample = copy.deepcopy(data_sample)
    utils.pipeline_utils.normalize_depth_data(roi_sample)
    return preprocessing.roi.extract_roi_flood_fill(roi_sample.depth, tolerance)


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
    RespiratoryRateExtractor()
