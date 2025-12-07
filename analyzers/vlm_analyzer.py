import os
import json
import re
import time
import asyncio
from dotenv import load_dotenv
import google.generativeai as genai
from pydantic import ValidationError
from moviepy import VideoFileClip
import tiktoken

from core.analyzers import AccidentAnalyzer
from core.schemas import AnalysisResult
from core.pricing import calculate_cost, video_token_per_second
from utils.llm_client import get_llm_client
from utils.common import clean_response

# Load environment variables from .env file
load_dotenv()

# Initialize tokenizer for output estimation
tokenizer = tiktoken.get_encoding("cl100k_base")

class VLMAnalyzer(AccidentAnalyzer):
    """
    An async analyzer that uses LLMClient and calculates performance metrics.
    """
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.client = get_llm_client(model_name)

    async def analyze_video(self, video_path: str, byte64_video = None, byte64_frames = None) -> tuple[AnalysisResult, dict]:
        """
        Handles the full async lifecycle for a single video analysis.
        """

        try:
            start_time = time.monotonic()

            prompt = self._build_prompt()
            raw_response_text = await asyncio.to_thread(
                self.client.invoke, self.model_name, prompt, byte64_video, byte64_frames
            )
            
            end_time = time.monotonic()

            # --- Performance Calculation ---
            # 1. Latency
            latency = end_time - start_time

            # 2. Input Tokens & Cost
            latency = end_time - start_time
            video_duration = await asyncio.to_thread(lambda: VideoFileClip(video_path).duration)
            video_tokens = int(video_duration * video_token_per_second(self.model_name))
            input_tokens = video_tokens + len(tokenizer.encode(prompt))
            output_tokens = len(tokenizer.encode(raw_response_text))

            # 3. Output Tokens & Cost
            output_tokens = len(tokenizer.encode(raw_response_text))

            # 4. Total Cost
            total_cost = calculate_cost(self.model_name, input_tokens, output_tokens)

            performance = {
                "latency": latency,
                "estimated_cost": total_cost,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens
            }

            # --- Result Validation ---
            json_part = clean_response(raw_response_text)
            analysis_data = json.loads(json_part)
            result = AnalysisResult(**analysis_data)

            return result, performance

        except (ValidationError, json.JSONDecodeError, Exception) as e:
            import traceback
            traceback.print_exc()
            raise ValueError(f"Model output validation failed for {self.model_name}: {e}. Raw output: {raw_response_text}") from e


    def _build_prompt(self) -> str:
        """Builds the detailed prompt for the VLM model."""
        return f"""
        You are an expert AI assistant for an insurance company, specializing in vehicle accident analysis from video footage.
        Your task is to analyze the provided video and return a structured JSON object with your findings.
        The JSON object must strictly adhere to the following detailed schema.

        **IMPORTANT**: For any field where the information is not available or cannot be determined from the video, you **must** use a specific string like 'Unknown', 'Not Visible', or 'N/A'. **Do not omit any fields.**

        {{
          "accident_summary": "A concise, objective summary of the sequence of events in the accident.",
          "vehicles_involved": [
            {{
              "vehicle_id": "An optional identifier, e.g., 'A', 'B'",
              "color": "e.g., 'Blue', 'White', 'Black', 'Unknown'",
              "type": "e.g., 'Sedan', 'SUV', 'Truck', "Bus", "Road vehicle", 'Unknown'",
              "damage_direction": "e.g., 'Front-end', 'Rear-end', 'Driver-side', 'Unknown'",
              "damage_level": "Choose one: 'None', 'Minor', 'Moderate', 'Severe', 'Unknown'"
            }}
          ],
          "liability_indicator": {{
            "color": "The color of the single at-fault vehicle.",
            "type": "The type of the single at-fault vehicle.",
            "driver_major_behavior": "The single key action of the at-fault driver that caused the accident, e.g., 'Speeding', 'Ran a red light'."
          }},
          "environmental_conditions": {{
            "time_of_day": "Choose one: 'Daylight', 'Dusk', 'Night', 'Dawn', 'Unknown'",
            "weather": "e.g., 'Clear', 'Rainy', 'Snowing', 'Foggy', or 'Not Visible'",
            "road_conditions": "e.g., 'Dry', 'Wet', 'Icy', or 'Not Visible'",
            "location_type": "e.g., 'Highway', 'Residential Street', 'Intersection', or 'Unknown'"
          }},
          "human_factors": {{
            "pedestrians_involved": "Choose one: 'Yes', 'No', 'Unknown'",
            "potential_witnesses": "Choose one: 'Yes', 'No', 'Unknown'",
            "injury_risk": "Choose one: 'Low', 'Medium', 'High', 'Unknown'"
          }},
          "collision_type": "Standard insurance term, e.g., 'Rear-End', 'T-Bone', or 'Unknown'",
          "traffic_controls_present": ["List of traffic controls observed. If none, return an empty list []"],
          "recommended_action": "Next steps for the claims adjuster. If no specific action, state 'Standard procedure'.",
          "reasoning_trace": [
            "A step-by-step log of key frames or events. If the video is unclear, provide a trace explaining why."
          ]
        }}
        """
