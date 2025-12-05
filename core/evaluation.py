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

tokenizer = tiktoken.get_encoding("cl100k_base")

# --- Part 1: Synchronous, Objective Metrics ---

def _calculate_attribute_similarity(model_vehicle: dict, golden_vehicle: dict) -> float:
    """Calculates a weighted similarity score between two vehicle attribute dicts."""
    score = 0.0
    weights = {"color": 0.4, "type": 0.4, "damage_direction": 0.1, "damage_level": 0.1}

    for key, weight in weights.items():
        if model_vehicle.get(key, "").lower() == golden_vehicle.get(key, "").lower():
            score += weight
    return score

def evaluate_vehicle_score(model_vehicles: list, golden_vehicles: list) -> dict:
    """Calculates precision and recall for the vehicles_involved list."""
    if not golden_vehicles:
        return {"precision": 1.0 if not model_vehicles else 0.0, "recall": 1.0}
    if not model_vehicles:
        return {"precision": 1.0, "recall": 0.0}

    true_positives = 0
    model_vehicles_copy = [v.model_dump() for v in model_vehicles]

    for golden_vehicle in golden_vehicles:
        best_match_score = -1
        best_match_index = -1
        for i, model_vehicle in enumerate(model_vehicles_copy):
            score = _calculate_attribute_similarity(model_vehicle, golden_vehicle)
            if score > best_match_score:
                best_match_score = score
                best_match_index = i

        if best_match_score >= 0.75:
            true_positives += 1
            model_vehicles_copy.pop(best_match_index)

    precision = true_positives / len(model_vehicles)
    recall = true_positives / len(golden_vehicles)
    return {"precision": precision, "recall": recall}

def evaluate_liability_score(model_liability, golden_liability: dict) -> float:
    """Calculates a weighted similarity score for the liability indicator."""
    score = 0.0
    weights = {"driver_major_behavior": 0.5, "color": 0.25, "type": 0.25}

    ref = [word_tokenize(golden_liability["driver_major_behavior"].lower())]
    cand = word_tokenize(model_liability.driver_major_behavior.lower())
    score += weights["driver_major_behavior"] * meteor_score(ref, cand)

    if model_liability.color.lower() == golden_liability["color"].lower():
        score += weights["color"]
    if model_liability.type.lower() == golden_liability["type"].lower():
        score += weights["type"]

    return score

def evaluate_sync_metrics(model_result: AnalysisResult, golden_data: dict) -> dict:
    """Generates a domain-driven scorecard for all synchronous metrics."""

    # --- Environment Score (F1) ---
    env_golden = golden_data["environmental_conditions"]
    env_model = model_result.environmental_conditions
    env_f1 = f1_score(
        list(env_golden.values()),
        [env_model.time_of_day, env_model.weather, env_model.road_conditions, env_model.location_type],
        average='weighted', zero_division=0
    )

    # --- Human Factors Score (Weighted Accuracy) ---
    hf_golden = golden_data["human_factors"]
    hf_model = model_result.human_factors
    injury_score = 1.0 if hf_model.injury_risk.lower() == hf_golden["injury_risk"].lower() else 0.0
    peds_score = 1.0 if hf_model.pedestrians_involved.lower() == hf_golden["pedestrians_involved"].lower() else 0.0
    witness_score = 1.0 if hf_model.potential_witnesses.lower() == hf_golden["potential_witnesses"].lower() else 0.0
    human_factors_score = (0.6 * injury_score) + (0.3 * peds_score) + (0.1 * witness_score)

    # --- Vehicle Score (Precision/Recall) ---
    vehicle_scores = evaluate_vehicle_score(
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
        "environment_score": env_f1,
        "human_factors_score": human_factors_score,
        "vehicle_precision": vehicle_scores["precision"],
        "vehicle_recall": vehicle_scores["recall"],
        "liability_score": liability_score,
        "summary_bleu": summary_bleu,
        "summary_meteor": summary_meteor
    }

# --- Part 2: Asynchronous, Intelligent "LLM as a Judge" Metrics ---

async def evaluate_llm_judge_metrics_async(model_result: AnalysisResult, golden_data: dict, judge_model_name: str) -> tuple[dict, dict]:
    """Uses a single LLM call to get a 1-100 rating for the accident summary."""

    start_time = time.monotonic()
    judge_model = genai.GenerativeModel(judge_model_name)

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
        response = await asyncio.to_thread(judge_model.generate_content, prompt)
        json_part = response.text.split('```json')[1].split('```')[0]
        judgments = json.loads(json_part)

        end_time = time.monotonic()

        judge_input_tokens = len(tokenizer.encode(prompt))
        judge_output_tokens = len(tokenizer.encode(response.text))
        judge_cost = calculate_cost(judge_model_name, judge_input_tokens, judge_output_tokens)
        judge_latency = end_time - start_time

        judge_performance = {
            "latency": judge_latency,
            "estimated_cost": judge_cost,
            "input_tokens": judge_input_tokens,
            "output_tokens": judge_output_tokens
        }

        return judgments, judge_performance
    except (IndexError, json.JSONDecodeError, Exception):
        return { "summary_rating": 0 }, {}
