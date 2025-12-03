# core/schemas.py
from pydantic import BaseModel, Field
from typing import List, Literal

class EnvironmentalConditions(BaseModel):
    time_of_day: Literal["Daylight", "Dusk", "Night", "Dawn", "Unknown"] = Field(..., description="The estimated time of day.")
    weather: str = Field(..., description="e.g., 'Clear', 'Rainy', 'Snowing', 'Foggy'")
    road_conditions: str = Field(..., description="e.g., 'Dry', 'Wet', 'Icy', 'Debris on road'")
    location_type: str = Field(..., description="e.g., 'Highway', 'Residential Street', 'Intersection', 'Parking Lot'")

class HumanFactors(BaseModel):
    occupants_visible: str = Field(..., description="e.g., 'Driver only', 'Driver and passenger', 'Multiple occupants'")
    pedestrians_involved: str = Field(..., description="A string: 'Yes', 'No', or 'Unknown'")
    driver_behavior_flags: List[str] = Field(..., description="List of observed behaviors, e.g., ['Appears Distracted', 'Speeding']")
    potential_witnesses: str = Field(..., description="Description of potential witnesses, e.g., 'None visible', 'Pedestrians on sidewalk'")

class VehicleDetails(BaseModel):
    vehicle_id: str = Field(..., description="A unique identifier for the vehicle, e.g., 'A', 'B'.")
    description: str = Field(..., description="e.g., 'Blue Sedan', 'Red SUV'")
    damage: str = Field(..., description="e.g., 'Severe front-end damage', 'Minor rear bumper scratches'")
    
class AnalysisResult(BaseModel):
    """The comprehensive analysis result schema."""
    accident_summary: str = Field(..., description="A concise, objective summary of the accident.")
    vehicles_involved: List[VehicleDetails]
    liability_indicator: str = Field(..., description="A clear statement on who is likely at fault and why.")
    
    # New structured fields
    environmental_conditions: EnvironmentalConditions
    human_factors: HumanFactors
    collision_type: str = Field(..., description="Standard insurance term, e.g., 'Rear-End', 'T-Bone', 'Sideswipe'")
    traffic_controls_present: List[str] = Field(..., description="List of traffic controls, e.g., ['Traffic light', 'Stop sign']")
    
    # Existing fields
    injury_risk: str = Field(..., description="An assessment of the potential for occupant injury (e.g., 'Low', 'Medium', 'High').")
    recommended_action: str = Field(..., description="Next steps for the claims adjuster.")
    reasoning_trace: List[str] = Field(..., description="A step-by-step log of key frames or events.")