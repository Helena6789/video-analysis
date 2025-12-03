

import os
import json
import re
import time
from dotenv import load_dotenv
import google.generativeai as genai
from pydantic import ValidationError
from core.analyzers import AccidentAnalyzer
from core.schemas import AnalysisResult
import streamlit as st

# Load environment variables from .env file
load_dotenv()

class GeminiProAnalyzer(AccidentAnalyzer):
    """
    An analyzer that uses a Gemini model.
    It separates file management from the core analysis logic.
    """
    def __init__(self, model_name: str = 'gemini-2.5-pro'):
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables.")
        genai.configure(api_key=self.api_key)
        self.model_name = model_name
        self.model = genai.GenerativeModel(self.model_name)

    def analyze_video(self, video_path: str) -> AnalysisResult:
        """
        Handles the full lifecycle (upload, analyze, delete) for a single video analysis.
        This is a convenience wrapper for single-use cases.
        """
        st.info("Uploading video to Gemini... (This may take a moment)")
        video_file = self.upload_video(video_path)
        
        try:
            return self.perform_analysis_on_file(video_file)
        finally:
            self.delete_video(video_file)

    def perform_analysis_on_file(self, video_file) -> AnalysisResult:
        """
        Performs analysis on a pre-uploaded video file object.
        This is the core analysis logic, decoupled from file I/O.
        """
        st.info(f"Analyzing with {self.model_name}...")
        prompt = self._build_prompt()
        
        try:
            response = self.model.generate_content(
                [prompt, video_file],
                request_options={"timeout": 1000}
            )
            raw_response_text = self._clean_response(response.text)
            analysis_data = json.loads(raw_response_text)
            result = AnalysisResult(**analysis_data)
            st.info(f"Completed analysis with {self.model_name}.")
            return result
        except (ValidationError, json.JSONDecodeError) as e:
            st.error(f"Data Validation Error from {self.model_name}. See details below.")
            st.subheader("Raw Model Output:")
            st.code(raw_response_text, language='json')
            raise ValueError(f"Model output validation failed: {e}") from e

    @staticmethod
    def upload_video(video_path: str):
        """Static method to upload a video and return the file object."""
        video_file = genai.upload_file(path=video_path)
        while video_file.state.name == "PROCESSING":
            time.sleep(2)
            video_file = genai.get_file(video_file.name)
        if video_file.state.name == "FAILED":
            raise ValueError("Video processing failed on the server.")
        return video_file

    @staticmethod
    def delete_video(video_file):
        """Static method to delete an uploaded video file."""
        genai.delete_file(video_file.name)

    def _clean_response(self, text: str) -> str:
        """Removes Markdown formatting from the model's response."""
        match = re.search(r"```json\n(.*)\n```", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return text.strip()

    def _build_prompt(self) -> str:
        """Builds the detailed prompt for the Gemini model."""
        return f"""
        You are an expert AI assistant for an insurance company, specializing in vehicle accident analysis from video footage.
        Your task is to analyze the provided video and return a structured JSON object with your findings.
        The JSON object must strictly adhere to the following detailed schema.

        **IMPORTANT**: For any field where the information is not available or cannot be determined from the video, you **must** use a specific string like 'Unknown', 'Not Visible', or 'N/A'. **Do not omit any fields.**

        {{
          "accident_summary": "A concise, objective summary of the sequence of events in the accident.",
          "vehicles_involved": [
            {{
              "vehicle_id": "A",
              "description": "e.g., 'Blue Sedan', 'Red SUV'",
              "damage": "e.g., 'Severe front-end damage', 'Minor rear bumper scratches'"
            }}
          ],
          "liability_indicator": "A clear statement on who is likely at fault and why. If unclear, state 'Undetermined'.",
          "environmental_conditions": {{
            "time_of_day": "Choose one: 'Daylight', 'Dusk', 'Night', 'Dawn', 'Unknown'",
            "weather": "e.g., 'Clear', 'Rainy', 'Snowing', 'Foggy', or 'Not Visible'",
            "road_conditions": "e.g., 'Dry', 'Wet', 'Icy', or 'Not Visible'",
            "location_type": "e.g., 'Highway', 'Residential Street', 'Intersection', or 'Unknown'"
          }},
          "human_factors": {{
            "occupants_visible": "e.g., 'Driver only', 'Driver and passenger', or 'Not Visible'",
            "pedestrians_involved": "A string: 'Yes', 'No', or 'Unknown'",
            "driver_behavior_flags": ["List of observed behaviors. If none, return an empty list []"],
            "potential_witnesses": "Description of potential witnesses. If none, state 'None visible'."
          }},
          "collision_type": "Standard insurance term, e.g., 'Rear-End', 'T-Bone', or 'Unknown'",
          "traffic_controls_present": ["List of traffic controls observed. If none, return an empty list []"],
          "injury_risk": "An assessment of the potential for occupant injury (e.g., 'Low', 'Medium', 'High', 'Unknown').",
          "recommended_action": "Next steps for the claims adjuster. If no specific action, state 'Standard procedure'.",
          "reasoning_trace": [
            "A step-by-step log of key frames or events. If the video is unclear, provide a trace explaining why."
          ]
        }}
        """
