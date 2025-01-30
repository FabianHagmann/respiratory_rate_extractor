import os
import re
import csv

def calculate_total_duration(root_dir):
    total_duration = 0

    file_pattern = re.compile(r"rgb_[0-9]+_([0-9]+)\.png")

    for subdir, dirs, files in os.walk(root_dir):
        if subdir[len(root_dir):].count(os.sep) > 2:
            continue

        if subdir[len(root_dir):].count(os.sep) == 2:
            milliseconds = []
            for file in files:
                match = file_pattern.match(file)
                if match:
                    milliseconds.append(int(match.group(1)))

            if milliseconds:
                duration = max(milliseconds) - min(milliseconds)
                total_duration += duration

    return total_duration

def generate_csv_from_annotations(root_dir, output_csv):
    rows = []
    column_headers = set()

    for subdir, dirs, files in os.walk(root_dir):
        relative_path = os.path.relpath(subdir, root_dir)
        depth = relative_path.count(os.sep)

        if depth == 1:
            parent_dir, current_dir = os.path.split(subdir)
            first_level_dir = os.path.basename(parent_dir)
            dir_name = f"{first_level_dir}-{current_dir}"

            annotation_file_path = os.path.join(subdir, "annotations_evaluated.txt")
            row_data = {"Directory": dir_name}

            if os.path.isfile(annotation_file_path):
                with open(annotation_file_path, "r") as annotation_file:
                    for line in annotation_file:
                        if ": " in line:
                            key, value = line.strip().split(": ", 1)
                            row_data[key] = value
                            column_headers.add(key)

            rows.append(row_data)

    all_headers = ["Directory"] + sorted(column_headers)
    with open(output_csv, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=all_headers)
        writer.writeheader()
        for row in rows:
            complete_row = {header: row.get(header, "") for header in all_headers}
            writer.writerow(complete_row)


if __name__ == "__main__":
    root_directory = os.path.join("..", "final")
    total_duration = calculate_total_duration(root_directory)
    print(f"Total duration across all subdirectories: {total_duration} milliseconds")

    output_csv_path = os.path.join(root_directory, "annotations_summary.csv")
    generate_csv_from_annotations(root_directory, output_csv_path)
    print(f"CSV file generated: {output_csv_path}")

