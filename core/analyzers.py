# core/analyzers.py
from abc import ABC, abstractmethod
from .schemas import AnalysisResult

class AccidentAnalyzer(ABC):
    """Abstract base class for a VLM-based accident analyzer."""

    @abstractmethod
    def analyze_video(self, video_path: str) -> AnalysisResult:
        """
        Analyzes a video and returns a structured analysis result.
        """
        pass
