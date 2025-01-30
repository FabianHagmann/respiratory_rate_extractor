import glob
import os
import subprocess
from tkinter import Tk
from tkinter.filedialog import askdirectory

import cv2

"""
Tool for manually annotating samples
- i - annotate inhalation maximum
- o - annotate exhalation maximum
- Left arrow key - move to the previous image
- Right arrow key - move to the next image
- ESC - exit the tool
"""


def annotate_images():
    Tk().withdraw()
    directory = askdirectory(title="Select Directory with Images")

    if not directory:
        print("No directory selected. Exiting.")
        return

    image_files = sorted(glob.glob(os.path.join(directory, "rgb_*.png")))
    if not image_files:
        print("No images found in the directory with the naming schema 'rgb_AAAAAA_BBBBBBBBBB.png'.")
        return

    annotations = {}
    current_index = 0

    def display_image(index, annotated_indices):
        """Display the image at the specified index and visually highlight nearby annotations."""
        img = cv2.imread(image_files[index])
        if img is None:
            print(f"Failed to load image: {image_files[index]}")
            return

        iteration, milliseconds = get_iteration_and_time(image_files[index])

        for annotated_idx in annotated_indices:
            if abs(annotated_idx - index) <= 5:
                img = cv2.rectangle(img, (0, 0), (img.shape[1], img.shape[0]), (0, 0, 255), 10)

        annotation = annotations.get(iteration, ("", ""))
        overlay_text = f"Image: {image_files[index]} | Iteration: {iteration} | Milliseconds: {milliseconds} | Annotation: {annotation[1]}"
        cv2.putText(img, overlay_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow("Annotation Tool", img)

    def get_iteration_and_time(filename):
        """Extract the iteration (counter) and milliseconds from the filename."""
        base = os.path.basename(filename)
        parts = base.split('_')
        iteration = int(parts[1])
        milliseconds = int(parts[2].split('.')[0])  # Remove the ".png" extension
        return iteration, milliseconds

    try:
        while True:
            annotated_indices = [image_files.index(f) for f in image_files if
                                 get_iteration_and_time(f)[0] in annotations]

            display_image(current_index, annotated_indices)

            key = cv2.waitKeyEx(0)

            if key == 27:  # ESC to exit
                break
            elif key == ord('i'):  # Annotate with 'i'
                iteration, milliseconds = get_iteration_and_time(image_files[current_index])
                annotations[iteration] = (milliseconds, 'i')
            elif key == ord('o'):  # Annotate with 'o'
                iteration, milliseconds = get_iteration_and_time(image_files[current_index])
                annotations[iteration] = (milliseconds, 'o')
            elif key == 37:  # Left arrow key
                current_index = max(0, current_index - 1)
            elif key == 39:  # Right arrow key
                current_index = min(len(image_files) - 1, current_index + 1)

    finally:
        with open(os.path.join(directory, "annotations.txt"), "w") as f:
            for iteration, (milliseconds, annotation) in sorted(annotations.items()):
                f.write(f"{iteration};{milliseconds};{annotation}\n")
        print(f"Annotations saved to {os.path.join(directory, 'annotations.txt')}.")

        try:
            print("Running evaluation script...")
            subprocess.run(['python', 'evaluate_annotations.py'], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error while running evaluation script: {e}")
        cv2.destroyAllWindows()


if __name__ == "__main__":
    annotate_images()
