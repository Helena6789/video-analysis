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

        # --- Dashboard ---
        pdf.section_title("Analysis Dashboard")
        pdf.section_body({"Summary": result['accident_summary']})

        liability = result['liability_indicator']
        pdf.section_body({"At-Fault Party": f"{liability['color']} {liability['type']}"})
        pdf.section_body({"Liability Reasoning": liability['driver_major_behavior']})

        pdf.section_body({"Recommended Action": result['recommended_action']})

        hf = result['human_factors']
        pdf.section_body({
            "Injury Risk": hf['injury_risk'],
            "Pedestrians Involved": hf['pedestrians_involved'],
            "Potential Witnesses": hf['potential_witnesses']
        })

        pdf.set_font('Arial', 'B', 10)
        pdf.cell(0, 5, "Vehicles Involved:")
        pdf.ln()
        pdf.set_font('Arial', '', 10)
        for v in result['vehicles_involved']:
            pdf.cell(0, 5, f"- {v['color']} {v['type']}: {v['damage_level']} damage to the {v['damage_direction']}.")
            pdf.ln()
        pdf.ln(4)

    return bytes(pdf.output())
