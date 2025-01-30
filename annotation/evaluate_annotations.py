import os
from tkinter import Tk
from tkinter.filedialog import askopenfilename

"""
Calculate statistical measures from an annotation file
"""


def evaluate_annotations():
    Tk().withdraw()
    filepath = askopenfilename(title="Select annotations.txt file", filetypes=[("Text Files", "*.txt")])

    if not filepath:
        print("No file selected. Exiting.")
        return

    try:
        with open(filepath, "r") as file:
            lines = file.readlines()

        annotations = []
        for line in lines:
            parts = line.strip().split(";")
            if len(parts) == 3:
                iteration, milliseconds, annotation = parts
                annotations.append((int(milliseconds), annotation))

        if not annotations:
            print("No valid annotations found in the file.")
            return

        in_breaths = sum(1 for _, annotation in annotations if annotation == "i")
        out_breaths = sum(1 for _, annotation in annotations if annotation == "o")

        annotation_balance = "true" if in_breaths == out_breaths else "false"

        first_timestamp = annotations[0][0]
        last_timestamp = annotations[-1][0]
        duration_ms = last_timestamp - first_timestamp
        num_breaths = max(in_breaths, out_breaths)
        if duration_ms > 0:
            respiratory_rate = (60000 / duration_ms) * num_breaths  # 60000 ms in a minute
        else:
            respiratory_rate = 0

        minutes = duration_ms // 60000
        seconds = (duration_ms % 60000) // 1000
        sample_duration = f"{minutes:02}:{seconds:02}"

        evaluated_filepath = os.path.join(os.path.dirname(filepath), "annotations_evaluated.txt")
        with open(evaluated_filepath, "w") as eval_file:
            eval_file.write(f"Total annotated out-breaths: {out_breaths}\n")
            eval_file.write(f"Total annotated in-breaths: {in_breaths}\n")
            eval_file.write(f"Annotation numbers current: {annotation_balance}\n")
            eval_file.write(f"Respiratory rate: {respiratory_rate:.1f}\n")
            eval_file.write(f"Sample duration: {sample_duration}\n")

        print(f"Evaluation completed. File saved as {evaluated_filepath}.")

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    evaluate_annotations()
