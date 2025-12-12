#tools/recalculate_sync_scores.py
import os
import json
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.evaluation import evaluate_sync_metrics

def recalculate_sync_scores(reports_dir, golden_dir):
    """
    Recalculate sync_scores for each model result in the reports directory based on the golden files.

    Args:
        reports_dir (str): Path to the directory containing report JSON files.
        golden_dir (str): Path to the directory containing golden JSON files.
    """
    for report_file in os.listdir(reports_dir):
        if report_file.endswith("stop_sign_non_stop_daylight_clear_analysis_20251210_211752.json"):
            report_path = os.path.join(reports_dir, report_file)

            # Extract the base name to find the corresponding golden file
            base_name = os.path.splitext(report_file)[0]
            base_name = "_".join(base_name.split("_")[:-3])  # Truncate suffix after 'analysis' and date
            golden_file = f"{base_name}.golden.json"
            golden_path = os.path.join(golden_dir, golden_file)

            if not os.path.exists(golden_path):
                print(f"Golden file {golden_file} not found for {report_file}, skipping...")
                continue

            # Load the report and golden data
            with open(report_path, "r") as report_f:
                report_data = json.load(report_f)

            with open(golden_path, "r") as golden_f:
                golden_data = json.load(golden_f)

            # Recalculate sync_scores for each model
            for model_name, model_data in report_data.get("models", {}).items():
                result = model_data.get("result")
                if result:
                    try:
                        # Ensure result is converted to the expected AnalysisResult format
                        from core.schemas import AnalysisResult
                        result_obj = AnalysisResult(**result)
                        sync_scores = evaluate_sync_metrics(result_obj, golden_data)
                        model_data["sync_scores"] = sync_scores
                        print(sync_scores)
                        print(f"Updated sync_scores for model {model_name} in {report_file}.")
                    except Exception as e:
                        print(f"Error recalculating sync_scores for model {model_name} in {report_file}: {e}")

            # Save the updated report
            with open(report_path, "w") as report_f:
                json.dump(report_data, report_f, indent=4)

if __name__ == "__main__":
    reports_directory = "reports"
    golden_files_directory = "benchmark/golden_files"

    recalculate_sync_scores(reports_directory, golden_files_directory)