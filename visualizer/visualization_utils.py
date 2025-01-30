import os.path


def find_annotation_frames_for_sample(sample_dir: str) -> [int]:
    annotation_path = os.path.join(sample_dir, "annotations.txt")
    annotation_frames = []

    with open(annotation_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            frame_index = line.split(';')[0]
            annotation_frames.append(int(frame_index))

    return annotation_frames
