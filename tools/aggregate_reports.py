# tools/aggregate_reports.py
import os
import json
import pandas as pd
from datetime import datetime
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description="Aggregate reports and generate summaries.")
    parser.add_argument("--reports-dir", default="reports", help="Directory containing the report JSON files.")
    parser.add_argument("--summary-dir", default="summary", help="Directory to save the summary outputs.")
    return parser.parse_args()

args = parse_arguments()
REPORTS_DIR = args.reports_dir
SUMMARY_DIR = args.summary_dir

def load_and_process_reports():
    """Loads all JSON reports and flattens them into a list of records."""
    all_records = []
    
    print(f"Scanning for reports in '{REPORTS_DIR}/'...")
    report_files = [f for f in os.listdir(REPORTS_DIR) if f.endswith(".json")]
    
    for report_file in report_files:
        filepath = os.path.join(REPORTS_DIR, report_file)
        print(f"Processing report: {report_file}")
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
    """Generates a clean HTML report from the dataframes, highlighting best scores and separating cases."""
    
    # --- 1. Styling Logic ---
    def highlight_global_best(s):
        if s.name in ["latency", "cost"]:
            is_best = s == s.min()
        else:
            is_best = s == s.max()
        return ['background-color: #d4edda' if v else '' for v in is_best]

    def highlight_best_per_case(s):
        if s.name in ["latency", "cost"]:
            best_in_group = s.groupby(level=0).transform('min')
        else:
            best_in_group = s.groupby(level=0).transform('max')
        is_best = s == best_in_group
        return ['background-color: #d4edda' if v else '' for v in is_best]

    def draw_group_separators(df):
        """Applies a thick top border to the first row of every new group (case)."""
        styles = pd.DataFrame('', index=df.index, columns=df.columns)
        cases = df.index.get_level_values(0)
        prev_case = cases[0]
        for i, case in enumerate(cases):
            if case != prev_case:
                styles.iloc[i] = 'border-top: 3px solid #555;'
            prev_case = case
        return styles

    # --- 2. Apply Styles ---
    styled_by_model = by_model_df.style.apply(highlight_global_best, axis=0)
    
    styled_by_case = (by_case_df.style
                      .apply(highlight_best_per_case, axis=0)
                      .apply(draw_group_separators, axis=None))

    # --- 3. HTML Generation (CSS FIX APPLIED BELOW) ---
    html_template = f"""
    <html>
    <head>
        <title>Performance Analysis Summary</title>
        <style>
            body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 30px; background-color: #f4f6f8; }}
            h1, h2 {{ color: #2c3e50; }}
            h1 {{ border-bottom: 2px solid #2c3e50; padding-bottom: 10px; }}
            
            table {{ 
                border-collapse: collapse; 
                width: 100%; 
                margin-bottom: 30px; 
                background-color: white; 
                box-shadow: 0 1px 3px rgba(0,0,0,0.1); 
            }}
            
            /* --- FIXED SECTIONS --- */
            
            /* 1. Main Header (Top Row) - White text on Blue */
            thead th {{ 
                background-color: #007bff; 
                color: white; 
                border: 1px solid #007bff; 
                padding: 12px 15px; 
                text-align: left;
            }}
            
            /* 2. Row Headers (The 'Case' column) - Dark text on Light Gray */
            tbody th {{ 
                background-color: #f8f9fa; 
                color: #333333; /* Dark text for visibility */
                border: 1px solid #dee2e6; 
                font-weight: bold; 
                padding: 12px 15px;
                text-align: left;
                vertical-align: top; /* Aligns text to top for long merged cells */
            }}
            
            /* 3. Data Cells */
            td {{ 
                border: 1px solid #e0e0e0; 
                padding: 12px 15px; 
                color: #333;
            }}
            
        </style>
    </head>
    <body>
        <h1>VLM Performance Analysis Summary</h1>
        <p style="color: #666;">Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
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