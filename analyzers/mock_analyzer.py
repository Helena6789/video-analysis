import asyncio
from core.analyzers import AccidentAnalyzer
from core.schemas import AnalysisResult, EnvironmentalConditions, HumanFactors, VehicleDetails

class MockVLMAnalyzer(AccidentAnalyzer):
    async def analyze_video(self, video_path: str) -> AnalysisResult:
        # Simulate a non-blocking network call
        await asyncio.sleep(2)

        # Return a hardcoded, realistic result conforming to the new schema
        return AnalysisResult(
            accident_summary="A rear-end collision occurred at a signalized intersection. Vehicle A (Blue Sedan) failed to stop and collided with the rear of Vehicle B (White SUV), which was stationary at a red light.",
            vehicles_involved=[
                VehicleDetails(vehicle_id="A", description="Blue Sedan", damage="Severe front-end damage"),
                VehicleDetails(vehicle_id="B", description="White SUV", damage="Moderate rear-end damage")
            ],
            liability_indicator="Vehicle A appears to be 100% at fault for failing to maintain a safe following distance and disobeying a traffic signal.",
            environmental_conditions=EnvironmentalConditions(
                time_of_day="Daylight",
                weather="Clear",
                road_conditions="Dry",
                location_type="Intersection"
            ),
            human_factors=HumanFactors(
                occupants_visible="Driver only in both vehicles",
                pedestrians_involved="No",
                driver_behavior_flags=["Appears Distracted"],
                potential_witnesses="Other stopped vehicles"
            ),
            collision_type="Rear-End",
            traffic_controls_present=["Traffic light"],
            injury_risk="Medium. The impact speed appears moderate, but whiplash injuries are common in rear-end collisions.",
            recommended_action="Initiate claim against Vehicle A's policy. Recommend medical evaluation for occupants of Vehicle B.",
            reasoning_trace=[
                "Frame 45: Vehicle B is stationary at a red light.",
                "Frame 52: Vehicle A is approaching at speed, driver appears to be looking down.",
                "Frame 55: Impact occurs. No brake lights visible on Vehicle A prior to impact."
            ]
        )
