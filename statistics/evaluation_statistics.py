import csv
import os


def extract_parameters_from_filename(filename: str) -> tuple[int, int, float, bool]:
    parts = filename.split('-')
    smoothing_factor = int(parts[5])
    combi_smoothing_factor = int(parts[6])
    if len(parts) == 8:
        down_sample_factor = float(parts[7].split('.')[0] + '.' + parts[7].split('.')[1])
        roi = False
    else:
        down_sample_factor = float(parts[7])
        roi = True
    return smoothing_factor, combi_smoothing_factor, down_sample_factor, roi


def calculate_statistics(file_path: str) -> tuple[float, float, float, float]:
    with open(file_path, 'r') as file:
        reader = csv.DictReader(file, delimiter=';')
        differences = []
        for row in reader:
            annotated_rate = float(row["Annotated Rate"])
            calculated_rate = float(row["Calculated Rate"])
            differences.append(abs(annotated_rate - calculated_rate))
        if differences:
            return sum(differences) / len(differences), differences[len(differences) // 2], min(differences), max(
                differences)
        else:
            return 0.0, 0.0, 0.0, 0.0


results_dir = os.path.join("..", "results", "hyperparameters")
evaluated_results = []

for filename in os.listdir(results_dir):
    if filename.endswith(".csv"):
        file_path = os.path.join(results_dir, filename)
        smoothing_factor, combi_smoothing_factor, down_sample_factor, roi = extract_parameters_from_filename(filename)
        avg_diff, median_diff, min_diff, max_diff = calculate_statistics(file_path)
        evaluated_results.append([smoothing_factor, combi_smoothing_factor, down_sample_factor, roi, round(avg_diff, 2),
                                  round(median_diff, 2), round(min_diff, 2), round(max_diff, 2)])

        output_file = '../results/final_evaluations/hyperparameter_evaluation.csv'
        with open(output_file, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(
                ["Smoothing Factor", "Combi Smoothing Factor", "Down Sample Factor", "ROI", "Average Difference",
                 "Median Difference", "Min Difference", "Max Difference"])
            writer.writerows(evaluated_results)
