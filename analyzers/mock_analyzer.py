# analyzers/mock_analyzer.py
import asyncio
import time
from core.analyzers import AccidentAnalyzer
from core.schemas import AnalysisResult, EnvironmentalConditions, HumanFactors, VehicleDetails, LiabilityIndicator

class MockVLMAnalyzer(AccidentAnalyzer):
    async def analyze_video(self, video_path: str) -> tuple[AnalysisResult, dict]:
        # Simulate a non-blocking network call
        start_time = time.monotonic()
        await asyncio.sleep(2)
        end_time = time.monotonic()

        # Create a dummy performance dictionary
        performance = {
            "latency": end_time - start_time,
            "estimated_cost": 0.0,
            "input_tokens": 322,
            "output_tokens": 266
        }

        # Return a hardcoded, realistic result conforming to the new schema
        result = AnalysisResult(
            accident_summary="A rear-end collision occurred at a signalized intersection. Vehicle A (Blue Sedan) failed to stop and collided with the rear of Vehicle B (White SUV), which was stationary at a red light.",
            vehicles_involved=[
                VehicleDetails(
                    vehicle_id="A",
                    color="Blue",
                    type="Sedan",
                    damage_direction="Front-end",
                    damage_level="Severe",
                    dashcam_vehicle="Yes"
                ),
                VehicleDetails(
                    vehicle_id="B",
                    color="White",
                    type="SUV",
                    damage_direction="Rear-end",
                    damage_level="Moderate",
                    dashcam_vehicle="No"
                )
            ],
            liability_indicator=LiabilityIndicator(
                color="Blue",
                type="Sedan",
                driver_major_behavior="Inattentive Driving",
                dashcam_vehicle="Yes"
            ),
            environmental_conditions=EnvironmentalConditions(
                time_of_day="Daylight",
                weather="Clear",
                road_conditions="Dry",
                location_type="Intersection"
            ),
            human_factors=HumanFactors(
                pedestrians_involved="No",
                potential_witnesses="Yes",
                injury_risk="Medium"
            ),
            collision_type="Rear-End",
            traffic_controls_present=["Traffic light"],
            recommended_action="Initiate claim against Vehicle A's policy. Recommend medical evaluation for occupants of Vehicle B.",
            reasoning_trace=[
                "Frame 45: Vehicle B is stationary at a red light.",
                "Frame 52: Vehicle A is approaching at speed, driver appears to be looking down.",
                "Frame 55: Impact occurs. No brake lights visible on Vehicle A prior to impact."
            ]
        )
        return result, performance
