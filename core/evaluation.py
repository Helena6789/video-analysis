# core/evaluation.py
import json
import asyncio
import time
import os
from sklearn.metrics import f1_score
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize
import nltk
import numpy as np
import tiktoken

from core.schemas import AnalysisResult
from core.pricing import calculate_cost
from utils.common import clean_response
from utils.llm_client import get_llm_client

# Ensure NLTK data is available.
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/wordnet')
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('punkt_tab')

tokenizer = tiktoken.get_encoding("cl100k_base")

# --- Part 1: Synchronous, Objective Metrics ---

def _calculate_attribute_similarity(model_data, golden_data: dict, weights: dict) -> float:
    """
    Calculates a weighted similarity score between model output (dict or object) and golden data (dict).

    Args:
        model_data: The prediction data (can be a Pydantic model or a dict).
        golden_data: The ground truth dictionary.
        weights: A dictionary mapping field names to their score weights.

    Returns:
        float: The accumulated weighted score.
    """
    score = 0.0

    for key, weight in weights.items():
        # Retrieve value from model_data (handle both Pydantic objects and dicts)
        if isinstance(model_data, dict):
            model_val = model_data.get(key, "")
        else:
            model_val = getattr(model_data, key, "")

        # Retrieve value from golden_data
        golden_val = golden_data.get(key, "")

        if str(model_val).strip().lower() == str(golden_val).strip().lower():
            score += weight
    return score

def evaluate_vehicle_score(model_vehicles: list, golden_vehicles: list) -> float:
    """
    Calculates an 'Average Match Quality' score for the vehicles_involved list.

    This logic iterates through each golden vehicle and finds its best semantic match
    in the model's output. This approach implicitly handles vehicle count mismatches:
    - If the model finds FEWER vehicles, the unmatched golden vehicles get a score of 0, lowering the average.
    - If the model finds MORE vehicles, the extra (hallucinated) vehicles are ignored, not affecting the score.
    """
    if not golden_vehicles:
        return 1.0 if not model_vehicles else 0.0
    if not model_vehicles:
        return 0.0

    match_scores = []
    model_vehicles_copy = [v.model_dump() for v in model_vehicles]

    # Define weights for vehicle attributes
    vehicle_weights = {
        "color": 0.3,
        "type": 0.3,
        "damage_direction": 0.1,
        "damage_level": 0.1,
        "dashcam_vehicle": 0.2
    }

    # For each ground truth vehicle, find its best match in the model's predictions.
    for golden_vehicle in golden_vehicles:
        if not model_vehicles_copy:
            # The model has run out of vehicles to match, so this golden vehicle was missed.
            match_scores.append(0)
            continue

        best_match_score = -1
        best_match_index = -1
        for i, model_vehicle in enumerate(model_vehicles_copy):
            score = _calculate_attribute_similarity(model_vehicle, golden_vehicle, vehicle_weights)
            if score > best_match_score:
                best_match_score = score
                best_match_index = i

        match_scores.append(best_match_score)

        # Remove the best match to prevent it from being matched to multiple golden vehicles.
        if best_match_index != -1:
            model_vehicles_copy.pop(best_match_index)

    # The final score is the average of the best-match scores for each golden vehicle.
    return np.mean(match_scores) if match_scores else 0.0

def evaluate_environment_score(model_env, golden_env: dict) -> float:
    """
    Calculates a weighted score for environmental conditions.
    Uses METEOR score for semantic fields (time, location) and exact match for others.
    """
    score = 0.0

    # 1. Exact Matches (Weather, Road Conditions)
    exact_weights = {
        "weather": 0.25,
        "road_conditions": 0.25
    }
    score += _calculate_attribute_similarity(model_env, golden_env, exact_weights)

    # 2. Semantic Matches (Time of Day, Location Type) using METEOR
    meteor_weights = {
        "time_of_day": 0.25,
        "location_type": 0.25
    }

    for field, weight in meteor_weights.items():
        # Retrieve normalized values
        golden_val = golden_env.get(field, "").strip().lower()

        if isinstance(model_env, dict):
            cand_val = model_env.get(field, "").strip().lower()
        else:
            cand_val = getattr(model_env, field, "").strip().lower()

        # Optimization: Check for exact equality before using METEOR for efficiency
        if cand_val == golden_val:
            score += weight
        else:
            ref = [word_tokenize(golden_val)]
            cand = word_tokenize(cand_val)
            score += weight * meteor_score(ref, cand)

    return score

def evaluate_liability_score(model_liability, golden_liability: dict) -> float:
    """Calculates a weighted similarity score for the liability indicator."""
    score = 0.0

    # 1. Calculate score for the complex text field (METEOR score)
    behavior_weight = 0.4
    ref = [word_tokenize(golden_liability.get("driver_major_behavior", "").lower())]
    cand = word_tokenize(model_liability.driver_major_behavior.lower())
    score += behavior_weight * meteor_score(ref, cand)

    # 2. Calculate score for simple categorical fields using the common function
    simple_weights = {
        "color": 0.2,
        "type": 0.2,
        "dashcam_vehicle": 0.2
    }
    score += _calculate_attribute_similarity(model_liability, golden_liability, simple_weights)

    return score

def evaluate_sync_metrics(model_result: AnalysisResult, golden_data: dict) -> dict:
    """Generates a domain-driven scorecard for all synchronous metrics."""

    # --- Environment Score (Weighted METEOR/Match) ---
    env_score = evaluate_environment_score(
        model_result.environmental_conditions,
        golden_data["environmental_conditions"]
    )

    # --- Human Factors Score (Weighted Accuracy) ---
    hf_weights = {
        "injury_risk": 0.6,
        "pedestrians_involved": 0.3,
        "potential_witnesses": 0.1
    }
    human_factors_score = _calculate_attribute_similarity(
        model_result.human_factors,
        golden_data["human_factors"],
        hf_weights
    )

    # --- Vehicle Score (Average Match Quality) ---
    vehicle_score = evaluate_vehicle_score(
        model_result.vehicles_involved,
        golden_data["vehicles_involved"]
    )

    # --- Liability Score (Weighted Similarity) ---
    liability_score = evaluate_liability_score(
        model_result.liability_indicator,
        golden_data["liability_indicator"]
    )

    # --- Summary Scores (BLEU/METEOR) ---
    summary_ref = [word_tokenize(golden_data["accident_summary"].lower())]
    summary_cand = word_tokenize(model_result.accident_summary.lower())
    summary_bleu = sentence_bleu(summary_ref, summary_cand, weights=(0.5, 0.5))
    summary_meteor = meteor_score(summary_ref, summary_cand)

    return {
        "environment_score": env_score,
        "human_factors_score": human_factors_score,
        "vehicle_score": vehicle_score,
        "liability_score": liability_score,
        "summary_bleu": summary_bleu,
        "summary_meteor": summary_meteor
    }

# --- Part 2: Asynchronous, Intelligent "LLM as a Judge" Metrics ---

async def evaluate_llm_judge_metrics_async(model_result: AnalysisResult, golden_data: dict, judge_model_name: str) -> tuple[dict, dict]:
    """Uses a single LLM call to get a 1-100 rating for the accident summary."""

    start_time = time.monotonic()

    judge_model = get_llm_client(judge_model_name)

    prompt = f"""
    You are an AI assistant acting as an impartial judge. Your task is to evaluate the quality of a model's accident summary against a "golden" ground truth.

    **EVALUATION CRITERIA:**
    - How accurately and completely does the model's summary capture the key events from the golden summary? Ignore minor differences in wording. Provide an integer rating from 1 to 100.

    **GROUND TRUTH SUMMARY:**
    "{golden_data['accident_summary']}"

    **MODEL'S SUMMARY:**
    "{model_result.accident_summary}"

    **OUTPUT FORMAT:**
    Respond with a single JSON object.
    ```json
    {{
      "summary_rating": 90
    }}
    ```
    """

    try:
        response_text = await asyncio.to_thread(judge_model.invoke, judge_model_name, prompt)
        json_part = clean_response(response_text)
        judgments = json.loads(json_part)

        end_time = time.monotonic()

        judge_input_tokens = len(tokenizer.encode(prompt))
        judge_output_tokens = len(tokenizer.encode(response_text))
        judge_cost = calculate_cost(judge_model_name, judge_input_tokens, judge_output_tokens)
        judge_latency = end_time - start_time

        judge_performance = {
            "latency": judge_latency,
            "estimated_cost": judge_cost,
            "input_tokens": judge_input_tokens,
            "output_tokens": judge_output_tokens
        }

        return judgments, judge_performance
    except (IndexError, json.JSONDecodeError, Exception) as e:
        print(F"Model evaluation output validation failed for {judge_model_name}: {e}.")
        return { "summary_rating": 0 }, {}