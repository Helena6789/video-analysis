# core/exporter.py
from fpdf import FPDF
from datetime import datetime
import json

class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'VLM Accident Analysis Report', 0, 1, 'C')
        self.set_font('Arial', '', 8)
        self.cell(0, 10, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 0, 1, 'C')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

    def section_title(self, title):
        self.set_font('Arial', 'B', 11)
        self.cell(0, 10, title, 0, 1, 'L')
        self.ln(2)

    def section_body(self, data):
        self.set_font('Arial', '', 10)
        if isinstance(data, dict):
            key_w = 40
            value_w = self.w - self.l_margin - self.r_margin - key_w
            for k, v in data.items():
                self.set_font('Arial', 'B', 10)
                self.cell(key_w, 5, f"{k.replace('_', ' ').title()}:", align='L')
                self.set_font('Arial', '', 10)
                self.multi_cell(value_w, 5, str(v), align='L')
                self.ln(2) # Add space after each key-value pair
        elif isinstance(data, list):
            for item in data:
                self.multi_cell(0, 5, f"- {item}")
        else:
            self.multi_cell(0, 5, str(data))
        self.ln(4)

    def metric_row(self, metrics, is_bold=False):
        self.set_font('Arial', 'B' if is_bold else '', 9)
        # Calculate width for each cell
        num_metrics = len(metrics)
        width = self.w / (num_metrics + 1) # Distribute width
        for metric in metrics:
            self.cell(width, 5, str(metric), 0, 0, 'C')
        self.ln()

def create_pdf_report(results_data: dict) -> bytes:
    """Generates a comprehensive PDF report from the analysis results."""
    pdf = PDF()
    pdf.add_page()

    # --- Report Header ---
    pdf.section_title("Analysis Overview")
    pdf.section_body({
        "Video File": results_data['video_filename'],
        "Report ID": results_data['report_id']
    })

    # --- Configuration ---
    pdf.section_title("Configuration Used")
    pdf.section_body({
        "Models Analyzed": ", ".join(results_data['config']['selected_models']),
        "Judge Model": results_data['config']['judge_model']
    })

    # --- Loop through each model's results ---
    for model_name, model_data in results_data['models'].items():
        pdf.add_page()
        pdf.section_title(f"Results for: {model_name}")

        result = model_data.get("result")
        if not result:
            pdf.section_body("Analysis failed for this model.")
            continue

        # --- Scorecards ---
        sync_scores = model_data.get("sync_scores")
        judge_scores = model_data.get("judge_scores")
        if sync_scores and judge_scores:
            pdf.section_title("Accuracy Scorecard")
            pdf.metric_row(["Struct. F1", "Injury Score", "Summary (BLEU)", "Summary (METEOR)", "Liability (BLEU)"], is_bold=True)
            pdf.metric_row([f"{sync_scores['categorical_f1']:.2f}", f"{sync_scores['injury_score']:.2f}", f"{sync_scores['summary_bleu']:.2f}", f"{sync_scores['summary_meteor']:.2f}", f"{sync_scores['liability_bleu']:.2f}"])
            pdf.ln(2)
            pdf.metric_row(["Dmg Precision", "Dmg Recall", "Beh Precision", "Beh Recall", "Summary Rating", "Liability Rating"], is_bold=True)
            pdf.metric_row([f"{judge_scores['damage_precision']:.2f}", f"{judge_scores['damage_recall']:.2f}", f"{judge_scores['behavior_precision']:.2f}", f"{judge_scores['behavior_recall']:.2f}", f"{judge_scores['summary_rating']}", f"{judge_scores['liability_rating']}"])
            pdf.ln(5)

        # --- Performance ---
        performance = model_data.get("performance", {})
        judge_performance = model_data.get("judge_performance", {})
        total_perf = performance.copy()
        if judge_performance:
            for key in total_perf:
                total_perf[key] += judge_performance.get(key, 0)

        pdf.section_title("Performance Metrics")
        pdf.metric_row(["Total Latency (s)", "Total Cost ($)", "Total Input Tokens", "Total Output Tokens"], is_bold=True)
        pdf.metric_row([f"{total_perf.get('latency', 0):.2f}", f"{total_perf.get('estimated_cost', 0):.4f}", f"{total_perf.get('input_tokens', 0):,}", f"{total_perf.get('output_tokens', 0):,}"])
        pdf.ln(5)

        # --- Dashboard ---
        pdf.section_title("Analysis Dashboard")
        pdf.section_body({"Summary": result['accident_summary']})
        pdf.section_body({"Liability": result['liability_indicator']})
        pdf.section_body({"Recommended Action": result['recommended_action']})
        pdf.section_body({"Injury Risk": result['injury_risk']})

        pdf.set_font('Arial', 'B', 10)
        pdf.cell(0, 5, "Vehicles Involved:")
        pdf.ln()
        pdf.set_font('Arial', '', 10)
        for v in result['vehicles_involved']:
            pdf.cell(0, 5, f"- Vehicle {v['vehicle_id']} ({v['description']}): {v['damage']}")
            pdf.ln()

        # ... Add other fields as needed ...

    return bytes(pdf.output())