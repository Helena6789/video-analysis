# core/evaluation.py
import json
from sklearn.metrics import f1_score
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize
import nltk
import numpy as np

from core.schemas import AnalysisResult

# Ensure NLTK data is available. This is a one-time download.
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
try:
    # Add the missing punkt_tab resource
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

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

def calculate_semantic_list_metrics(model_list: list[str], golden_list: list[str]) -> dict:
    """
    Calculates semantic precision and recall for lists of strings using METEOR score.
    """
    if not model_list and not golden_list:
        return {"precision": 1.0, "recall": 1.0}
    if not model_list:
        return {"precision": 1.0, "recall": 0.0} # Model correctly said nothing, but missed everything.
    if not golden_list:
        return {"precision": 0.0, "recall": 1.0} # Model hallucinated items that weren't there.

    tokenized_model_list = [word_tokenize(s.lower()) for s in model_list]
    tokenized_golden_list = [word_tokenize(s.lower()) for s in golden_list]

    # --- Calculate Semantic Precision ---
    # For each item in the model's list, find its best match in the golden list.
    precision_scores = []
    for model_item in tokenized_model_list:
        best_match_score = max(meteor_score([ref], model_item) for ref in tokenized_golden_list)
        precision_scores.append(best_match_score)
    semantic_precision = np.mean(precision_scores) if precision_scores else 0.0

    # --- Calculate Semantic Recall ---
    # For each item in the golden list, find its best match in the model's list.
    recall_scores = []
    for golden_item in tokenized_golden_list:
        best_match_score = max(meteor_score([golden_item], cand) for cand in tokenized_model_list)
        recall_scores.append(best_match_score)
    semantic_recall = np.mean(recall_scores) if recall_scores else 0.0
    
    return {"precision": semantic_precision, "recall": semantic_recall}


def evaluate_accuracy(model_result: AnalysisResult, golden_data: dict) -> dict:
    """
    Generates a full accuracy scorecard by comparing the model's result
    to the golden (ground truth) data.
    """
    
    # 1. Categorical F1 Score
    f1 = calculate_categorical_f1(model_result, golden_data)
    
    # 2. Descriptive Text Scores
    summary_scores = calculate_text_similarity(
        model_result.accident_summary,
        golden_data["descriptive_fields"]["accident_summary"]
    )
    liability_scores = calculate_text_similarity(
        model_result.liability_indicator,
        golden_data["descriptive_fields"]["liability_indicator"]
    )
    
    # 3. Semantic List Metrics
    trace_metrics = calculate_semantic_list_metrics(
        model_result.reasoning_trace,
        golden_data["list_fields"]["reasoning_trace"]
    )
    behavior_metrics = calculate_semantic_list_metrics(
        model_result.human_factors.driver_behavior_flags,
        golden_data["list_fields"]["driver_behavior_flags"]
    )
    
    # Combine into a final scorecard
    scorecard = {
        "categorical_f1": f1,
        "summary_bleu": summary_scores["bleu"],
        "summary_meteor": summary_scores["meteor"],
        "liability_bleu": liability_scores["bleu"],
        "liability_meteor": liability_scores["meteor"],
        "trace_precision": trace_metrics["precision"],
        "trace_recall": trace_metrics["recall"],
        "behavior_precision": behavior_metrics["precision"],
        "behavior_recall": behavior_metrics["recall"]
    }
    
    return scorecard