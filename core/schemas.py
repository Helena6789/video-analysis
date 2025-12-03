# core/schemas.py
from pydantic import BaseModel
from typing import List, Dict

class AnalysisResult(BaseModel):
    accident_summary: str
    vehicles_involved: List[Dict[str, str]]
    liability_indicator: str
    injury_risk: str
    recommended_action: str
    reasoning_trace: List[str]
