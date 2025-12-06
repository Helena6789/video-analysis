# core/schemas.py
from pydantic import BaseModel, Field
from typing import List, Literal, Optional

class EnvironmentalConditions(BaseModel):
    time_of_day: Literal["Daylight", "Dusk", "Night", "Dawn", "Unknown"] = Field(..., description="The estimated time of day.")
    weather: str = Field(..., description="e.g., 'Clear', 'Rainy', 'Snowing', 'Foggy'")
    road_conditions: str = Field(..., description="e.g., 'Dry', 'Wet', 'Icy', 'Debris on road'")
    location_type: str = Field(..., description="e.g., 'Highway', 'Residential Street', 'Intersection', 'Parking Lot'")

class HumanFactors(BaseModel):
    pedestrians_involved: str = Field(..., description="A string: 'Yes', 'No', or 'Unknown'")
    potential_witnesses: str = Field(..., description="A string: 'Yes', 'No', or 'Unknown'")
    injury_risk: str = Field(..., description="A string: 'Low', 'Medium', 'High', or 'Unknown'")

class VehicleDetails(BaseModel):
    vehicle_id: Optional[str] = Field(None, description="An optional identifier for the vehicle, e.g., 'A', 'B'.")
    color: str = Field(..., description="e.g., 'Blue', 'White', 'Black'")
    type: str = Field(..., description="e.g., 'Sedan', 'SUV', 'Truck'")
    damage_direction: str = Field(..., description="e.g., 'Front-end', 'Rear-end', 'Driver-side', 'Unknown'")
    damage_level: str = Field(..., description="The severity of the damage. e.g., 'None', 'Minor', 'Moderate', 'Severe', 'Unknown'")

class LiabilityIndicator(BaseModel):
    color: str = Field(..., description="The color of the at-fault vehicle.")
    type: str = Field(..., description="The type of the at-fault vehicle.")
    driver_major_behavior: str = Field(..., description="The key action of the at-fault driver that caused the accident.")

class AnalysisResult(BaseModel):
    """The comprehensive analysis result schema."""
    accident_summary: str = Field(..., description="A concise, objective summary of the accident.")
    vehicles_involved: List[VehicleDetails]
    liability_indicator: LiabilityIndicator
    environmental_conditions: EnvironmentalConditions
    human_factors: HumanFactors
    collision_type: str = Field(..., description="Standard insurance term, e.g., 'Rear-End', 'T-Bone', 'Sideswipe'")
    traffic_controls_present: List[str] = Field(..., description="List of traffic controls, e.g., ['Traffic light', 'Stop sign']")
    recommended_action: str = Field(..., description="Next steps for the claims adjuster.")
    reasoning_trace: List[str] = Field(..., description="A step-by-step log of key frames or events.")