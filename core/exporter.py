# core/exporter.py
import markdown
from jinja2 import Environment, FileSystemLoader
from weasyprint import HTML
from datetime import datetime

def create_pdf_report(results_data: dict) -> bytes:
    """Generates a PDF report from the analysis results using an HTML template."""
    
    # Define the Markdown-to-HTML filter
    def markdown_filter(text):
        if not text:
            return ""
        return markdown.markdown(text, extensions=['fenced_code', 'tables', 'nl2br'])

    # Set up Jinja2 environment
    env = Environment(loader=FileSystemLoader('templates'))
    env.filters['markdown'] = markdown_filter
    template = env.get_template('report.html')
    
    # Prepare data for the template
    template_data = {
        "generation_date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        **results_data
    }
    
    # Render the HTML template with the data
    html_out = template.render(template_data)
    
    # Convert HTML to PDF
    pdf_bytes = HTML(string=html_out).write_pdf()
    
    return pdf_bytes