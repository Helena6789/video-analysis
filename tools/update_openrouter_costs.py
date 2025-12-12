import os
import json
import csv
from datetime import datetime

# Paths to the directories and files
REPORTS_DIR = "reports/newrun1"  # Directory containing the JSON files
CSV_FILE = "openrouter_activity_qwen.csv"  # Path to the CSV file

MODEL_MAPPING = {
    "GPT-5.1": "openai/gpt-5.1-20251113",
    "GPT-5.2": "openai/gpt-5.2-20251211",
    "GPT-5.1": "openai/gpt-5.1-20251113",
    "GPT-5.2": "openai/gpt-5.2-20251211",
    "Qwen3 VL 8B Thinking": "qwen/qwen3-vl-8b-thinking",
    "Qwen3 VL 235B A22B Thinking": "qwen/qwen3-vl-235b-a22b-thinking",
}

def load_cost_data():
    """
    Load cost data for GPT-5.1 and GPT-5.2 from the CSV file.
    Returns a sorted list of dictionaries containing generation_id, model, and cost_total.
    """
    cost_data = []
    with open(CSV_FILE, "r") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            model = row["model_permaslug"]
            if model in MODEL_MAPPING.values():
                cost_data.append({
                    "generation_id": row["generation_id"],
                    "model": model,
                    "cost_total": float(row["cost_total"]),
                    "created_at": datetime.strptime(row["created_at"], "%Y-%m-%d %H:%M:%S.%f")
                })
    # Sort by creation time
    cost_data.sort(key=lambda x: x["created_at"])
    return cost_data

def update_json_files(cost_data):
    """
    Update the JSON files in the REPORTS_DIR with the cost data.
    """
    json_files = [f for f in os.listdir(REPORTS_DIR) if f.endswith(".json")]
    json_files.sort()  # Ensure files are processed in sorted order

    cost_index = 0
    for json_file in json_files:
        file_path = os.path.join(REPORTS_DIR, json_file)
        with open(file_path, "r") as f:
            data = json.load(f)

        # Update the performance cost for GPT-5.1 and GPT-5.2
        for model_name, model_data in data.get("models", {}).items():
            mapped_model = MODEL_MAPPING.get(model_name)
            if mapped_model and cost_index < len(cost_data) and cost_data[cost_index]["model"] == mapped_model:
                if model_data.get("performance"):
                    print(f"Processing {model_name} in {json_file} with cost data index {cost_index}.")
                    print(f"Current cost: {model_data['performance'].get('estimated_cost', 'N/A')}, New cost: {cost_data[cost_index]['cost_total']}")
                    model_data["performance"]["estimated_cost"] = cost_data[cost_index]["cost_total"]
                    cost_index += 1
                else:
                    print(f"Skipping {model_name} in {json_file} because 'performance' field is missing.")

        # Save the updated file
        with open(file_path, "w") as f:
            json.dump(data, f, indent=4)
        print(f"Updated file: {json_file}")

def main():
    cost_data = load_cost_data()
    update_json_files(cost_data)

if __name__ == "__main__":
    main()