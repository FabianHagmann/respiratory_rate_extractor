import os
from enum import Enum

import numpy as np
from PIL import Image

import preprocessing


class DataType(Enum):
    THERMAL = 1
    DEPTH = 2
    RGB = 3


class SampleData:
    def __init__(self):
        self.thermal = []
        self.depth = []
        self.rgb = []
        self.roi = None

    def add_data(self, data_type: DataType, data: np.ndarray):
        """
        Add data to the container based on the data type.
        :param data_type: datatype to be added to
        :param data (ndarray) to be added
        """

        if data_type == DataType.THERMAL:
            self.thermal.append(data)
        elif data_type == DataType.DEPTH:
            self.depth.append(data)
        elif data_type == DataType.RGB:
            self.rgb.append(data)
        else:
            raise ValueError("Invalid data type")

    def __str__(self):
        """Return a string representation showing the count of elements in each list."""
        return (f"Thermal data: {len(self.thermal)} elements\n"
                f"Depth data: {len(self.depth)} elements\n"
                f"RGB data: {len(self.rgb)} elements")


def load_data_from_directory(sample_dir: str, scale_factor=1.0) -> SampleData:
    data = SampleData()
    skippedThermal = False
    skippedDepth = False
    skippedRGB = False

    for f in os.listdir(sample_dir):
        if not os.path.isfile(os.path.join(sample_dir, f)) or not f.endswith(".png"):
            continue

        image = Image.open(os.path.join(sample_dir, f))
        if scale_factor != 1.0:
            image = image.resize((int(image.width * scale_factor), int(image.height * scale_factor)))
        image_array = np.array(image)
        if "thermal" in f:
            if not skippedThermal:
                skippedThermal = True
                continue
            data.add_data(DataType.THERMAL, image_array)
        elif "depth" in f:
            if not skippedDepth:
                skippedDepth = True
                continue
            data.add_data(DataType.DEPTH, image_array)
        elif "rgb" in f:
            if not skippedRGB:
                skippedRGB = True
                continue
            data.add_data(DataType.RGB, image_array)

    return data


def smooth_depth_data(sample: SampleData) -> None:
    for i, data in enumerate(sample.depth):
        sample.depth[i] = preprocessing.smooth(data, 2)


def normalize_depth_data(sample: SampleData) -> None:
    for i, data in enumerate(sample.depth):
        sample.depth[i] = preprocessing.normalize(data)


def normalize_thermal_data(sample: SampleData) -> None:
    for i, data in enumerate(sample.thermal):
        sample.thermal[i] = preprocessing.normalize(data)
