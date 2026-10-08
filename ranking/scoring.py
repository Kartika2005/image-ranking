from __future__ import annotations
from utils.normalization import clamp


def calculate_final_score(features: dict, config: dict) -> dict:
    """Blend quality, coverage and composition with semantic/category evidence.

    Category importance modifies semantic evidence (rather than being added alone), so a
    technically poor image cannot win solely because it resembles an important scene.
    """
    weights = config["scoring"]["weights"]
    priority = config["scoring"]["category_importance"][features["predicted_category"]]
    mix = config["scoring"]["category_importance_mix"]
    semantic_relevance = (1 - mix) * features["semantic_score"] + mix * priority
    final_score = (weights["semantic_relevance"] * semantic_relevance + weights["image_quality"] * features["quality_score"] +
                   weights["event_coverage"] * features["event_coverage_score"] + weights["composition"] * features["composition_score"])
    features["category_importance"] = priority
    features["semantic_relevance"] = clamp(semantic_relevance)
    features["final_score"] = clamp(final_score)
    return features
