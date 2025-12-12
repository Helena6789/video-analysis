#tools/batch_video_analysis.py
import sys
import os
import asyncio
import json
from datetime import datetime
from dotenv import load_dotenv
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app import ANALYZER_CATALOG, get_analyzer, run_analysis_async
from core.evaluation import evaluate_sync_metrics, evaluate_llm_judge_metrics_async
from core.schemas import AnalysisResult
from utils.llm_client import GeminiClient
from utils.video_utils import extract_frames_as_base64, video_base64_encoding

# Load environment variables
load_dotenv()

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Batch video analysis tool.")
parser.add_argument("--video_dir", type=str, required=True, help="Path to the directory containing videos.")
parser.add_argument("--reports_dir", type=str, required=True, help="Path to save the analysis reports.")
args = parser.parse_args()

VIDEO_DIR = args.video_dir
REPORTS_DIR = args.reports_dir

SELECTED_MODELS = [
    "Gemini 3 Pro (Preview)",
    "Gemini 2.5 Pro",
    "Nemotron Nano 12B 2 VL (Free)",
    "Qwen3 VL 8B Thinking",
    "Qwen3 VL 235B A22B Thinking",
    "GPT-5.1",
    "GPT-5.2",
]  # Predefined list of models
JUDGE_MODEL = "gemini-2.5-flash"  # Predefined judge model

async def analyze_videos():
    """
    Analyze all videos in the configured directory using the predefined models and judge model.
    """
    video_files = [f for f in os.listdir(VIDEO_DIR) if f.endswith((".mp4", ".mov", ".avi"))]

    if not video_files:
        print("No videos found in the directory.")
        return

    # Ensure the REPORTS_DIR exists
    if not os.path.exists(REPORTS_DIR):
        os.makedirs(REPORTS_DIR)
        print(f"Created reports directory: {REPORTS_DIR}")

    for video_file in video_files:
        video_path = os.path.join(VIDEO_DIR, video_file)
        print(f"Processing video: {video_file}")

        golden_data = None
        base_filename, _ = os.path.splitext(video_file)
        golden_filename = f"{base_filename}.golden.json"
        golden_filepath = os.path.join("benchmark", "golden_files", golden_filename)
        if os.path.exists(golden_filepath):
            try:
                with open(golden_filepath, "r") as f:
                    golden_data = json.load(f)
                print(f"Loaded ground truth file: {golden_filename}")
            except Exception as e:
                print(f"Failed to load ground truth file: {e}")

        # Ensure status is defined before checking if it's None
        status = None
        if status is None:
            class DummyLogger:
                def write(self, message):
                    print(message)

                def markdown(self, message, unsafe_allow_html):
                    print(message)

            status = DummyLogger()

        # Run analysis asynchronously
        analysis_results = await run_analysis_async(SELECTED_MODELS, video_path, status=status)

        # Construct the full results object
        report_id = f"{base_filename}_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        results_obj = {
            "report_id": report_id,
            "video_filename": video_file,
            "config": {
                "selected_models": SELECTED_MODELS,
                "judge_model": JUDGE_MODEL
            },
            "models": {}
        }

        for model_name, (result, performance) in analysis_results.items():
            try:
                # Run evaluation on the fly before saving
                sync_scores = None
                judge_scores = None
                judge_performance = None
                if golden_data and isinstance(result, AnalysisResult):
                    print(f"Evaluating accuracy for {model_name}...")
                    sync_scores = evaluate_sync_metrics(result, golden_data)
                    judge_scores, judge_performance = await evaluate_llm_judge_metrics_async(result, golden_data, JUDGE_MODEL)
                    print(f"Completed accuracy evaluation for {model_name}")

                results_obj["models"][model_name] = {
                    "result": result.model_dump() if result else None,
                    "performance": performance,
                    "sync_scores": sync_scores,
                    "judge_scores": judge_scores,
                    "judge_performance": judge_performance
                }
            except Exception as e:
                print(f"Error processing model {model_name}: {e}")
                results_obj["models"][model_name] = {
                    "result": None,
                    "performance": {"error": str(e)},
                    "sync_scores": None,
                    "judge_scores": None,
                    "judge_performance": None
                }

        # Save the results to a JSON file
        report_filename = f"{results_obj['report_id']}.json"
        report_path = os.path.join(REPORTS_DIR, report_filename)
        with open(report_path, "w") as f:
            json.dump(results_obj, f, indent=4)
        print(f"Analysis saved to {report_path}")

if __name__ == "__main__":
    asyncio.run(analyze_videos())