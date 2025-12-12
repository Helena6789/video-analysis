# utils/aggregate_reports.py
import os
import json
import pandas as pd
from datetime import datetime

REPORTS_DIR = "reports"
SUMMARY_DIR = "summary"

def load_and_process_reports():
    """Loads all JSON reports and flattens them into a list of records."""
    all_records = []
    
    print(f"Scanning for reports in '{REPORTS_DIR}/'...")
    report_files = [f for f in os.listdir(REPORTS_DIR) if f.endswith(".json")]
    
    for report_file in report_files:
        filepath = os.path.join(REPORTS_DIR, report_file)
        with open(filepath, "r") as f:
            data = json.load(f)
            
            case_name = data.get("video_filename", "unknown_case").replace(".mp4", "")
            
            for model_name, model_data in data.get("models", {}).items():
                record = {
                    "case": case_name,
                    "model": model_name,
                }
                
                performance = model_data.get("performance", {})
                judge_performance = model_data.get("judge_performance", {})
                if performance:
                    record["latency"] = performance.get("latency", 0) + (judge_performance.get("latency", 0) if judge_performance else 0)
                    record["cost"] = performance.get("estimated_cost", 0) + (judge_performance.get("estimated_cost", 0) if judge_performance else 0)

                sync_scores = model_data.get("sync_scores", {})
                if sync_scores:
                    record.update(sync_scores)

                judge_scores = model_data.get("judge_scores", {})
                if judge_scores:
                    record.update(judge_scores)
                
                all_records.append(record)
                
    print(f"Found and processed {len(all_records)} records from {len(report_files)} report files.")
    return all_records

def generate_summary_html(by_case_df, by_model_df):
    """Generates a clean HTML report from the dataframes, highlighting the best scores."""
    
    def highlight_max_or_min(s, column_name):
        if column_name == "latency" or column_name == "cost":
            # Highlight the lowest value for latency
            is_min = s == s.min()
            return ['background-color: #d4edda' if v else '' for v in is_min]
        else:
            # Highlight the highest value for other columns
            is_max = s == s.max()
            return ['background-color: #d4edda' if v else '' for v in is_max]

    # Apply styling to both dataframes
    styled_by_model = by_model_df.style.apply(lambda s: highlight_max_or_min(s, s.name), axis=0)
    styled_by_case = by_case_df.style.apply(lambda s: highlight_max_or_min(s, s.name), axis=0)

    html_template = f"""
    <html>
    <head>
        <title>Performance Analysis Summary</title>
        <style>
            body {{ font-family: sans-serif; margin: 20px; }}
            h1, h2 {{ color: #1E88E5; }}
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
        </style>
    </head>
    <body>
        <h1>VLM Performance Analysis Summary</h1>
        <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <h2>Average Performance by Model</h2>
        {styled_by_model.to_html(float_format='%.2f')}
        
        <h2>Detailed Comparison by Case</h2>
        {styled_by_case.to_html(float_format='%.2f')}
    </body>
    </html>
    """
    return html_template

def main():
    """Main function to run the aggregation and save reports."""
    
    records = load_and_process_reports()
    if not records:
        print("No records found. Exiting.")
        return

    df = pd.DataFrame(records)
    
    # --- Create Summaries ---
    numeric_cols = df.select_dtypes(include='number').columns
    
    # 1. Detailed comparison: Group by case and model and average any duplicates.
    # This is the critical fix to ensure a unique index.
    by_case = df.groupby(['case', 'model'])[numeric_cols].mean().sort_index()
    
    # 2. Average performance by model
    by_model = df.groupby('model')[numeric_cols].mean()
    
    # --- Save Outputs ---
    if not os.path.exists(SUMMARY_DIR):
        os.makedirs(SUMMARY_DIR)
        
    by_case_csv_path = os.path.join(SUMMARY_DIR, "comparison_by_case.csv")
    by_model_csv_path = os.path.join(SUMMARY_DIR, "average_performance_by_model.csv")
    html_report_path = os.path.join(SUMMARY_DIR, "summary_report.html")
    
    by_case.to_csv(by_case_csv_path, float_format='%.4f')
    by_model.to_csv(by_model_csv_path, float_format='%.4f')
    
    html_content = generate_summary_html(by_case, by_model)
    with open(html_report_path, "w") as f:
        f.write(html_content)
        
    print(f"\nSummary reports generated successfully in '{SUMMARY_DIR}/'")
    print(f"- {by_case_csv_path}")
    print(f"- {by_model_csv_path}")
    print(f"- {html_report_path}")

if __name__ == "__main__":
    main()