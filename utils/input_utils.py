import os
import sys
from datetime import datetime, time


def input_sample_number(dataset_dir: str) -> tuple[int, int]:
    """
    Utility function for inputing the subject number and sample index

    :param dataset_dir: directory base path of dataset
    :return: subject number, sample index
    """

    subject_dirs = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
    subject_number = int(input(f"Subject number [1,{len(subject_dirs)}]: "))

    subject_dir = __find_subject_directory__(dataset_dir, subject_number)
    if subject_dir is None:
        print("subject directory not found in dataset", file=sys.stderr)
        return -1, -1
    sample_dirs = [d for d in os.listdir(subject_dir) if os.path.isdir(os.path.join(subject_dir, d))]
    sample_idx = int(input(f"Sample index [0,{len(sample_dirs) - 1}]: "))

    return subject_number, sample_idx


def find_sample_dir(subject_number: int, sample_idx: int, dataset_dir: str) -> str:
    """
    Find the sample directory with the given subject number and sample index in the dataset directory
    :param subject_number: valid subject number within the dataset dir
    :param sample_idx: valid sample index within the subject dir
    :param dataset_dir: base dir of the dataset
    :return: path of the valid sample dir
    """
    subject_dir = __find_subject_directory__(dataset_dir, subject_number)
    if subject_dir is None:
        print("subject directory not found in dataset", file=sys.stderr)
        return ""

    samples = os.listdir(subject_dir)
    if sample_idx >= len(samples):
        print("sample index out of range", file=sys.stderr)
        return ""

    return os.path.join(subject_dir, samples[sample_idx])


def load_annotation_data(sample_dir: str) -> tuple[int, int, float, time]:
    """
    Load the annotation data from the given sample directory
    :param sample_dir: path of the sample directory
    :return: tuple of (out_breaths, in_breaths, rate, duration)
    """
    global duration, rate, in_breaths, out_breaths
    annotation_file = os.path.join(sample_dir, "annotations_evaluated.txt")
    with open(annotation_file, 'r') as f:
        lines = f.readlines()

    for line in lines:
        if line.startswith("Total annotated out-breaths"):
            out_breaths = int(line.split(':')[1].strip())
        elif line.startswith("Total annotated in-breaths"):
            in_breaths = int(line.split(':')[1].strip())
        elif line.startswith("Respiratory rate"):
            rate = float(line.split(':')[1].strip())
        elif line.startswith("Sample duration"):
            duration = datetime.strptime(line.split(':')[1].strip() + ':' + line.split(':')[2].strip(), '%M:%S').time()
    return out_breaths, in_breaths, rate, duration


def query_int(prompt: str, min_val: int, max_val: int) -> int:
    """
    Query the user for an integer input within the given range
    :param prompt: prompt message
    :param min_val: minimum value
    :param max_val: maximum value
    :return: integer input within the range
    """
    while True:
        try:
            val = int(input(prompt))
            if min_val <= val <= max_val:
                return val
            else:
                print(f"Value out of range [{min_val},{max_val}]", file=sys.stderr)
        except ValueError:
            print("Invalid input, please enter an integer", file=sys.stderr)


def query_float(prompt: str, min_val: float, max_val: float) -> float:
    """
    Query the user for a float input within the given range
    :param prompt: prompt message
    :param min_val: minimum value
    :param max_val: maximum value
    :return: float input within the range
    """
    while True:
        try:
            val = float(input(prompt))
            if min_val <= val <= max_val:
                return val
            else:
                print(f"Value out of range [{min_val},{max_val}]", file=sys.stderr)
        except ValueError:
            print("Invalid input, please enter a float", file=sys.stderr)


def query_bool(prompt: str) -> bool:
    """
    Query the user for a boolean input
    :param prompt: prompt message
    :return: boolean input
    """
    while True:
        val = input(prompt).strip().lower()
        if val == 'y' or val == 'yes':
            return True
        elif val == 'n' or val == 'no':
            return False
        else:
            print("Invalid input, please enter 'y' or 'n'", file=sys.stderr)


def __find_subject_directory__(dataset_dir: str, subject_number: int) -> str:
    """
    Find the directory of the subject with the given number in the dataset directory
    :param dataset_dir: base dir of the dataset
    :param subject_number: valid subject number within the dataset dir
    :return: path of the valid subject dir
    """
    for item in os.listdir(dataset_dir):
        item_path = os.path.join(dataset_dir, item)
        item_number = int(item.split('-')[0].strip())

        if os.path.isdir(item_path) and item_number == subject_number:
            return item_path

    return None


def find_duration(sample_dir) -> time:
    files = os.listdir(sample_dir)
    millis = []
    for file in files:
        if file.startswith("rgb_") and file.endswith(".png"):
            millis.append(int(file.split("_")[2].split(".")[0]))

    millis = (max(millis) - min(millis))
    return time(second=millis // 1000, microsecond=(millis % 1000) * 1000)
