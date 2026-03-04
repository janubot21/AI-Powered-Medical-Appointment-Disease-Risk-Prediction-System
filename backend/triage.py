from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class PriorityRecommendation:
    priority: str
    recommended_slot: str
    badge_text: str
    badge_color: str


# Centralized mapping so new risk tiers can be added in one place.
PRIORITY_RULES: Dict[str, PriorityRecommendation] = {
    "high": PriorityRecommendation(
        priority="Immediate",
        recommended_slot="Within 1 hour",
        badge_text="Immediate Attention Required",
        badge_color="red",
    ),
    "medium": PriorityRecommendation(
        priority="Same Day",
        recommended_slot="Today",
        badge_text="Same Day Appointment",
        badge_color="orange",
    ),
    "low": PriorityRecommendation(
        priority="Normal",
        recommended_slot="Next Available Date",
        badge_text="Normal Appointment",
        badge_color="green",
    ),
}


def normalize_risk_level(value: Any) -> str:
    text = str(value or "").strip().lower()
    if "high" in text:
        return "high"
    if "medium" in text:
        return "medium"
    return "low"


def determine_priority(risk_level: Any) -> PriorityRecommendation:
    normalized = normalize_risk_level(risk_level)
    return PRIORITY_RULES.get(normalized, PRIORITY_RULES["low"])

