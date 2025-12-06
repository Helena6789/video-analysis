#analyzers/openrouter_analyzer.py
import base64
import os
import re
import json
import time
import asyncio
from dotenv import load_dotenv
from moviepy import VideoFileClip

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
import tiktoken

from core.analyzers import AccidentAnalyzer
from core.schemas import AnalysisResult
from core.pricing import calculate_cost, video_token_per_second
from utils.common import clean_response

# Load environment variables from .env file
load_dotenv()

# Initialize tokenizer for output estimation
tokenizer = tiktoken.get_encoding("cl100k_base")

class OpenRouterAnalyzer(AccidentAnalyzer):
    """
    An async analyzer that uses model from OpenRouter, sending the video as base64 string.
    """
    def __init__(self, model_name: str):
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY not found in environment variables.")
        self.model_name = model_name
        self.client = ChatOpenAI(
            model=self.model_name,
            openai_api_key=self.api_key,
            openai_api_base="https://openrouter.ai/api/v1",
        )

    async def analyze_video(self, video_path: str) -> tuple[AnalysisResult, dict]:
        """
        Handles the full async lifecycle for a single video analysis.
        """
        try:
            start_time = time.monotonic()
            
            with open(video_path, "rb") as video_file:
                video_base64 = base64.b64encode(video_file.read()).decode("utf-8")
            
            prompt = self._build_prompt()

            message = HumanMessage(
                content= [
                    {"type": "text", "text": prompt},
                    {
                        "type": "video_url",
                        "video_url" : { "url": f"data:video/mp4;base64,{video_base64}"}
                    }
                ]
            )

            response = await asyncio.to_thread(self.client.invoke, [message])
            raw_response_text = clean_response(response.content)
            end_time = time.monotonic()

            latency = end_time - start_time
            video_duration = await asyncio.to_thread(lambda: VideoFileClip(video_path).duration)
            video_tokens = int(video_duration * video_token_per_second(self.model_name))
            input_tokens = video_tokens + len(tokenizer.encode(prompt))
            output_tokens = len(tokenizer.encode(raw_response_text))

            total_cost = calculate_cost(self.model_name, input_tokens, output_tokens)

            performance = {
                "latency": latency,
                "estimated_cost": total_cost,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens
            }

            analysis_data = json.loads(raw_response_text)
            result = AnalysisResult(**analysis_data)
            return result, performance
        except (json.JSONDecodeError) as e:
            raise ValueError(F"Model output validation failed for {self.model_name}: {e}. Raw output: {raw_response_text}") from e

        
    def _build_prompt(self) -> str:
        """Builds the detailed prompt for the model."""
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
              "type": "e.g., 'Sedan', 'SUV', 'Truck', "Bus", "Road vehicle"",
              "damage_direction": "e.g., 'Front-end', 'Rear-end', 'Driver-side'",
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
