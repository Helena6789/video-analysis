# analyzers/mock_analyzer.py
import time
from core.analyzers import AccidentAnalyzer
from core.schemas import AnalysisResult

class MockVLMAnalyzer(AccidentAnalyzer):
    def analyze_video(self, video_path: str) -> AnalysisResult:
        # Simulate processing time
        time.sleep(3)
        
        # Return a hardcoded, realistic result
        return AnalysisResult(
            accident_summary="A rear-end collision occurred at a signalized intersection. Vehicle A (Blue Sedan) failed to stop and collided with the rear of Vehicle B (White SUV), which was stationary at a red light.",
            vehicles_involved=[
                {"vehicle_id": "A", "description": "Blue Sedan", "damage": "Severe front-end damage"},
                {"vehicle_id": "B", "description": "White SUV", "damage": "Moderate rear-end damage"}
            ],
            liability_indicator="Vehicle A appears to be 100% at fault for failing to maintain a safe following distance and disobeying a traffic signal.",
            injury_risk="Medium. The impact speed appears moderate, but whiplash injuries are common in rear-end collisions.",
            recommended_action="Initiate claim against Vehicle A's policy. Recommend medical evaluation for occupants of Vehicle B. Flag for potential fraud review due to sudden stop claim by Vehicle A (if any).",
            reasoning_trace=[
                "Frame 45: Vehicle B is stationary.",
                "Frame 47: Traffic light is red.",
                "Frame 52: Vehicle A is approaching at speed, no brake lights visible.",
                "Frame 55: Impact occurs."
            ]
        )
