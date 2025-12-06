# utils/common.py
import re

def clean_response(text: str) -> str:
    """Removes Markdown formatting from the model's response."""
    match = re.search(r"```json\n(.*)\n```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()
        