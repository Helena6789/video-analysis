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

def calculate_injury_score(model_risk: str, golden_risk: str) -> float:
    model_risk_lower = model_risk.lower().strip()
    golden_risk_lower = golden_risk.lower().strip()
    classification_score = 0.0
    extracted_risk = "unknown"
    for risk_level in ["low", "medium", "high"]:
        if model_risk_lower.startswith(risk_level):
            extracted_risk = risk_level
            break
    if extracted_risk == golden_risk_lower:
        classification_score = 1.0
    elif golden_risk_lower in extracted_risk or extracted_risk in golden_risk_lower:
        classification_score = 0.5

    return classification_score

def evaluate_sync_metrics(model_result: AnalysisResult, golden_data: dict) -> dict:
    golden_cat = golden_data["categorical_fields"]
    model_cat_values = [
        model_result.environmental_conditions.time_of_day,
        model_result.environmental_conditions.weather,
        model_result.environmental_conditions.road_conditions,
        model_result.environmental_conditions.location_type,
        model_result.human_factors.pedestrians_involved,
        model_result.collision_type
    ]
    f1 = f1_score(list(golden_cat.values()), model_cat_values, average='weighted', zero_division=0)
    insurance_golden = golden_data["insurance_fields"]
    summary_ref = [word_tokenize(insurance_golden["accident_summary"].lower())]
    summary_cand = word_tokenize(model_result.accident_summary.lower())
    summary_bleu = sentence_bleu(summary_ref, summary_cand, weights=(0.5, 0.5))
    summary_meteor = meteor_score(summary_ref, summary_cand)
    liability_ref = [word_tokenize(insurance_golden["liability_indicator"].lower())]
    liability_cand = word_tokenize(model_result.liability_indicator.lower())
    liability_bleu = sentence_bleu(liability_ref, liability_cand, weights=(0.5, 0.5))
    liability_meteor = meteor_score(liability_ref, liability_cand)
    injury_score = calculate_injury_score(
        model_result.injury_risk,
        insurance_golden["injury_risk"]
    )
    return {
        "categorical_f1": f1,
        "summary_bleu": summary_bleu,
        "summary_meteor": summary_meteor,
        "liability_bleu": liability_bleu,
        "liability_meteor": liability_meteor,
        "injury_score": injury_score
    }

# --- Part 2: Asynchronous, Intelligent "LLM as a Judge" Metrics ---

async def evaluate_llm_judge_metrics_async(model_result: AnalysisResult, golden_data: dict, judge_model_name: str) -> tuple[dict, dict]:
    """Uses a single LLM call to get ratings and precision/recall for key fields."""

    start_time = time.monotonic()
    insurance_golden = golden_data["insurance_fields"]
    judge_model = genai.GenerativeModel(judge_model_name)

    model_damages = [f'{v.description}: {v.damage}' for v in model_result.vehicles_involved]
    model_behaviors = model_result.human_factors.driver_behavior_flags
    golden_damages = insurance_golden['vehicle_damages']
    golden_behaviors = insurance_golden['driver_behavior_flags']

    prompt = f"""
    You are an AI assistant acting as an impartial judge. Your task is to evaluate a model's analysis of a car accident against a "golden" ground truth. Provide a comprehensive evaluation in a single JSON object.

    **EVALUATION CRITERIA:**
    1.  **Ratings (1-100)**: For `summary` and `liability`, provide an integer rating from 1 to 100 on how well the model's text captures the semantic meaning and key details of the golden text.
    2.  **Precision/Recall**: For `damages` and `behaviors`, evaluate the semantic consistency between the model's list and the golden list. For each item in the list being evaluated, provide a boolean (true/false).

    **GROUND TRUTH DATA:**
    - Golden Summary: "{insurance_golden['accident_summary']}"
    - Golden Liability: "{insurance_golden['liability_indicator']}"
    - Golden Damages: {json.dumps(golden_damages)}
    - Golden Behaviors: {json.dumps(golden_behaviors)}

    **MODEL'S ANALYSIS:**
    - Model Summary: "{model_result.accident_summary}"
    - Model Liability: "{model_result.liability_indicator}"
    - Model Damages: {json.dumps(model_damages)}
    - Model Behaviors: {json.dumps(model_behaviors)}

    **OUTPUT FORMAT:**
    Respond with a single JSON object.
    ```json
    {{
      "summary_rating": 90,
      "liability_rating": 95,
      "damage_precision": {{ "Model damage 1": true, "Model damage 2": false }},
      "damage_recall": {{ "Golden damage 1": true }},
      "behavior_precision": {{ "Model behavior 1": true }},
      "behavior_recall": {{ "Golden behavior 1": false }}
    }}
    ```
    """

    try:
        response = await asyncio.to_thread(judge_model.generate_content, prompt)
        json_part = response.text.split('```json')[1].split('```')[0]
        judgments = json.loads(json_part)

        damage_precision = sum(1 for v in judgments.get("damage_precision", {}).values() if v) / len(model_damages) if model_damages else 1.0
        damage_recall = sum(1 for v in judgments.get("damage_recall", {}).values() if v) / len(golden_damages) if golden_damages else 1.0
        behavior_precision = sum(1 for v in judgments.get("behavior_precision", {}).values() if v) / len(model_behaviors) if model_behaviors else 1.0
        behavior_recall = sum(1 for v in judgments.get("behavior_recall", {}).values() if v) / len(golden_behaviors) if golden_behaviors else 1.0

        judge_scores = {
            "summary_rating": judgments.get("summary_rating", 0),
            "liability_rating": judgments.get("liability_rating", 0),
            "damage_precision": damage_precision,
            "damage_recall": damage_recall,
            "behavior_precision": behavior_precision,
            "behavior_recall": behavior_recall
        }

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

        return judge_scores, judge_performance
    except (IndexError, json.JSONDecodeError, Exception):
        return { "summary_rating": 0, "liability_rating": 0, "damage_precision": 0, "damage_recall": 0, "behavior_precision": 0, "behavior_recall": 0 }, None
