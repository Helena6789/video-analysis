
#utils/llm_client.py
import os
import abc
import time
import asyncio
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

# Load environment variables from .env file
load_dotenv()

class LLMClient(abc.ABC):
    """Abstract base class for a unified LLM client."""
    @abc.abstractmethod
    def invoke(self, model_name: str, prompt: str, base64_video = None, byte64_frames = None) -> str:
        pass


class GeminiClient(LLMClient):
    """Client for Google's Gemini Model"""
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables.")
        genai.configure(api_key=self.api_key)
        self.models = {}

    def invoke(self, model_name: str, prompt: str, base64_video = None, byte64_frames = None) -> str:
        if model_name not in self.models:
            self.models[model_name] = genai.GenerativeModel(model_name)
        
        content = [prompt]
        if base64_video:
            # append uploaded file via genai.upload_file().
            content.append(base64_video)

        response = self.models[model_name].generate_content(content)
        return response.text

    @staticmethod
    async def upload_video(video_path: str):
        """Static async method to upload a video and return the file object."""
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        def upload_sync():
            video_file = genai.upload_file(path=video_path)
            while video_file.state.name == "PROCESSING":
                time.sleep(2)
                video_file = genai.get_file(video_file.name)
            return video_file

        video_file = await asyncio.to_thread(upload_sync)

        if video_file.state.name == "FAILED":
            raise ValueError("Video processing failed on the server.")
        return video_file

    @staticmethod
    async def delete_video(video_file):
        """Static async method to delete an uploaded video file."""
        await asyncio.to_thread(genai.delete_file, video_file.name)


class OpenRouterClient(LLMClient):
    """Client for models hosted on OpenRouter"""
    def __init__(self):
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY not found in environment variables.")
        self.models = {}
    
    def invoke(self, model_name: str, prompt: str, base64_video = None, byte64_frames = None) -> str:
        if model_name not in self.models:
            self.models[model_name] = ChatOpenAI(
                model=model_name,
                openai_api_key=self.api_key,
                openai_api_base="https://openrouter.ai/api/v1",
            )
        
        content =  [{"type": "text", "text": prompt}]
        if base64_video and self._model_support_base64_video(model_name):
            print(f"Send video with video base64 encoding for {model_name}...")
            content.append({
                "type": "video_url",
                "video_url" : { "url": f"data:video/mp4;base64,{base64_video}"}
            })
        elif byte64_frames:
            print(f"Send video with image frames for {model_name}...")
            for base64_frame in byte64_frames:
                content.append({
                    "type": "image_url",
                    "image_url" : { "url": f"data:image/jpeg;base64,{base64_frame}"}
                })


        message = HumanMessage(content=content)
        response = self.models[model_name].invoke([message])
        return response.content

    def _model_support_base64_video(self, model_name):
        if "nemotron" in model_name or "gemini" in model_name:
            return True
        return False

def get_llm_client(model_name: str) -> LLMClient:
    """Factory function to get the correct client based on the model name."""
    if model_name.startswith("gemini"): # Official Gemini instead of OpenRouter
        return GeminiClient()
    else:
        return OpenRouterClient()


            
