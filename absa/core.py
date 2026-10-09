"""Shared label, preprocessing, evidence, and inference functions."""
import html
from pathlib import Path
import re

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ASPECTS = ("service", "rooms", "location", "value", "cleanliness", "sleep_quality", "business_service")
LABELS = {0: "Negative", 1: "Neutral", 2: "Positive"}
KEYWORDS = {
    "service": r"\b(staff|service|reception|receptionist|concierge|front desk|check[ -]?in|manager|employees?|helpful|rude)\b",
    "rooms": r"\b(rooms?|suite|bathroom|shower|air conditioning|spacious|cramped|furniture|balcony)\b",
    "location": r"\b(location|located|central|downtown|nearby|subway|metro|station|airport|walking distance|beach)\b",
    "value": r"\b(value|price|cost|expensive|cheap|affordable|overpriced|money|budget|worth|paid)\b",
    "cleanliness": r"\b(clean|cleanliness|dirty|filthy|spotless|dust|dusty|stains?|mould|mold|hygiene|smell|smelled)\b",
    "sleep_quality": r"\b(sleep|slept|quiet|noisy|noise|bed|beds|mattress|pillow|pillows|soundproof|awake)\b",
    "business_service": r"\b(business cent(?:er|re)|meeting|conference|printer|printing|wi[ -]?fi|internet|workspace|work desk)\b",
}


def normalize(text):
    text = html.unescape(str(text))
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip().lower()


def polarity(ratings):
    """1-2 negative, 3 neutral, 4-5 positive. Missing stays missing."""
    x = np.asarray(ratings, dtype=float)
    return np.where(np.isnan(x), np.nan, np.where(x < 3, 0, np.where(x == 3, 1, 2)))


def evidence(text, aspect, limit=4):
    # Evidence is a heuristic, not a supervised aspect detector.
    clauses = re.split(r"(?<=[.!?;])\s+|\s+\b(?:but|however|although|whereas)\b\s+", html.unescape(text), flags=re.I)
    matches = [c.strip(" ,.;") for c in clauses if re.search(KEYWORDS[aspect], c, re.I)]
    return matches if limit is None else matches[:limit]


class Analyzer:
    def __init__(self, model_path=None):
        # Load only the model generated locally or included with this project.
        self.bundle = joblib.load(model_path or ROOT / "models" / "hotel_absa.joblib")

    def _predict(self, target, task, text):
        selected = self.bundle["models"][target][task]
        vectorizer = self.bundle["vectorizers"][selected["feature"]]
        prediction = selected["estimator"].predict(vectorizer.transform([normalize(text)]))[0]
        if task == "polarity":
            return LABELS[int(prediction)]
        return int(np.clip(np.floor(float(prediction) + 0.5), 1, 5))

    def analyze(self, text, include_unmentioned=False):
        if not isinstance(text, str) or len(text.strip()) < 3:
            raise ValueError("Enter a review with at least 3 characters.")
        if len(text) > 20000:
            raise ValueError("Please keep each review within 20,000 characters.")
        result = {
            "overall": {}, "aspects": [], "inference_version": 2,
            "method": "Aspect classifiers use all matching keyword excerpts. Overall and optional unmentioned-aspect estimates use full review text. Reported test metrics evaluate the original full-review baseline, not excerpt inference.",
        }
        if "overall" in self.bundle["models"]:
            result["overall"] = {"sentiment": self._predict("overall", "polarity", text), "rating": self._predict("overall", "rating", text)}
        for aspect in ASPECTS:
            # Separate opinions before classification so praise for another
            # aspect does not overwhelm a complaint. Keep every match for the
            # model even though the page displays at most four excerpts.
            matches = evidence(text, aspect, limit=None)
            row = {"aspect": aspect, "name": aspect.replace("_", " ").title(), "evidence": matches[:4], "prediction_source": None}
            if aspect not in self.bundle["models"]:
                row.update(status="unavailable", sentiment=None, rating=None)
            elif not matches and not include_unmentioned:
                row.update(status="not_mentioned", sentiment=None, rating=None)
            else:
                prediction_text = ". ".join(matches) if matches else text
                row.update(
                    status="predicted",
                    prediction_source="aspect_excerpts" if matches else "full_review",
                    sentiment=self._predict(aspect, "polarity", prediction_text),
                    rating=self._predict(aspect, "rating", prediction_text),
                )
            result["aspects"].append(row)
        return result
