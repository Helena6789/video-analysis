# core/evaluation.py
import json
import asyncio
import time
from sklearn.metrics import f1_score
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize
import nltk
import google.generativeai as genai
import tiktoken

from core.schemas import AnalysisResult
from .pricing import calculate_cost

# Ensure NLTK data is available.
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('wordnet')

# Initialize tokenizer for token counting
tokenizer = tiktoken.get_encoding("cl100k_base")

def calculate_categorical_f1(model_result: AnalysisResult, golden_data: dict) -> float:
    """Calculates the F1 score for the structured categorical fields."""
    golden_values = list(golden_data["categorical_fields"].values())

    model_values = [
        model_result.environmental_conditions.time_of_day,
        model_result.environmental_conditions.weather,
        model_result.environmental_conditions.road_conditions,
        model_result.environmental_conditions.location_type,
        model_result.human_factors.pedestrians_involved,
        model_result.collision_type
    ]

    return f1_score(golden_values, model_values, average='weighted', zero_division=0)

def calculate_text_similarity(model_text: str, golden_text: str) -> dict:
    """Calculates BLEU and METEOR scores for descriptive text fields."""
    reference = [word_tokenize(golden_text.lower())]
    candidate = word_tokenize(model_text.lower())

    bleu = sentence_bleu(reference, candidate, weights=(0.5, 0.5))
    meteor = meteor_score(reference, candidate)

    return {"bleu": bleu, "meteor": meteor}

async def evaluate_accuracy(model_result: AnalysisResult, golden_data: dict, judge_model_name: str) -> tuple[dict, dict]:
    """
    Generates a full accuracy scorecard and tracks the performance of the judge model.
    Returns (scorecard, judge_performance).
    """
    start_time = time.monotonic()

    # --- Step 1: Calculate synchronous metrics ---
    f1 = calculate_categorical_f1(model_result, golden_data)
    summary_scores = calculate_text_similarity(model_result.accident_summary, golden_data["descriptive_fields"]["accident_summary"])
    liability_scores = calculate_text_similarity(model_result.liability_indicator, golden_data["descriptive_fields"]["liability_indicator"])

    # --- Step 2: Prepare for the single async LLM judge call ---
    model_trace = model_result.reasoning_trace
    golden_trace = golden_data["list_fields"]["reasoning_trace"]
    model_behaviors = model_result.human_factors.driver_behavior_flags
    golden_behaviors = golden_data["list_fields"]["driver_behavior_flags"]

    judge_model = genai.GenerativeModel(judge_model_name)

    # --- Step 3: Build the combined prompt ---
    combined_prompt = f"""
    You are an AI assistant acting as an impartial judge. Your task is to evaluate two sets of lists (reasoning trace and driver behaviors) and return a single JSON object with four distinct evaluations.

    **Input Data:**
    - Golden Reasoning Trace: {json.dumps(golden_trace)}
    - Model Reasoning Trace: {json.dumps(model_trace)}
    - Golden Driver Behaviors: {json.dumps(golden_behaviors)}
    - Model Driver Behaviors: {json.dumps(model_behaviors)}

    **Your Task (Perform all four):**
    1.  **`trace_precision`**: For each item in "Model Reasoning Trace", is it semantically consistent with at least one item in "Golden Reasoning Trace"?
    2.  **`trace_recall`**: For each item in "Golden Reasoning Trace", is its core meaning captured by at least one item in "Model Reasoning Trace"?
    3.  **`behavior_precision`**: For each item in "Model Driver Behaviors", is it semantically consistent with at least one item in "Golden Driver Behaviors"?
    4.  **`behavior_recall`**: For each item in "Golden Driver Behaviors", is its core meaning captured by at least one item in "Model Driver Behaviors"?

    **Output Format:**
    Respond with a single JSON object containing four keys. The value for each key should be another JSON object mapping the items from the list being evaluated to a boolean (true/false).

    Example response format:
    ```json
    {{
      "trace_precision": {{ "Model step 1": true, "Model step 2": false }},
      "trace_recall": {{ "Golden step 1": true }},
      "behavior_precision": {{ "Model behavior 1": true }},
      "behavior_recall": {{ "Golden behavior 1": false }}
    }}
    ```
    """

    # --- Step 4: Make the single API call ---
    response = await asyncio.to_thread(judge_model.generate_content, combined_prompt)
    try:
        json_part = response.text.split('```json')[1].split('```')[0]
        judgments = json.loads(json_part)
    except (IndexError, json.JSONDecodeError):
        judgments = {}

    # --- Step 5: Calculate scores ---
    trace_precision = sum(1 for v in judgments.get("trace_precision", {}).values() if v) / len(model_trace) if model_trace else 1.0
    trace_recall = sum(1 for v in judgments.get("trace_recall", {}).values() if v) / len(golden_trace) if golden_trace else 1.0
    behavior_precision = sum(1 for v in judgments.get("behavior_precision", {}).values() if v) / len(model_behaviors) if model_behaviors else 1.0
    behavior_recall = sum(1 for v in judgments.get("behavior_recall", {}).values() if v) / len(golden_behaviors) if golden_behaviors else 1.0

    end_time = time.monotonic()

    # --- Step 6: Calculate Judge Performance ---
    judge_input_tokens = len(tokenizer.encode(combined_prompt))
    judge_output_tokens = len(tokenizer.encode(response.text))
    judge_cost = calculate_cost(judge_model_name, judge_input_tokens, judge_output_tokens)
    judge_latency = end_time - start_time

    judge_performance = {
        "latency": judge_latency,
        "estimated_cost": judge_cost,
        "input_tokens": judge_input_tokens,
        "output_tokens": judge_output_tokens
    }

    # --- Step 7: Assemble the final scorecard ---
    scorecard = {
        "categorical_f1": f1,
        "summary_bleu": summary_scores["bleu"],
        "summary_meteor": summary_scores["meteor"],
        "liability_bleu": liability_scores["bleu"],
        "liability_meteor": liability_scores["meteor"],
        "trace_precision": trace_precision,
        "trace_recall": trace_recall,
        "behavior_precision": behavior_precision,
        "behavior_recall": behavior_recall
    }

    return scorecard, judge_performance