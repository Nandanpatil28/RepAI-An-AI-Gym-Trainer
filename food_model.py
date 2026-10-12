"""Local food image classification helpers."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path


MODEL_ID = "nateraw/food"


class FoodModelUnavailable(RuntimeError):
    """Raised when local food model dependencies or weights are unavailable."""


def normalize_label(label: str) -> str:
    return " ".join(label.replace("_", " ").split()).title()


@lru_cache(maxsize=1)
def _load_pipeline():
    try:
        import torch
        from transformers import pipeline
    except ImportError as exc:
        raise FoodModelUnavailable(
            "Food scanner dependencies are not installed. Install torch and transformers before scanning."
        ) from exc

    device = 0 if torch.cuda.is_available() else -1
    try:
        return pipeline("image-classification", model=MODEL_ID, device=device), device
    except Exception as exc:  # Model files may not have been downloaded yet.
        raise FoodModelUnavailable(f"Local food model is not ready: {exc}") from exc


def classify_food(image_path: Path, top_k: int = 5) -> dict:
    pipe, device = _load_pipeline()
    try:
        raw_predictions = pipe(str(image_path), top_k=top_k)
    except Exception as exc:
        raise FoodModelUnavailable(f"Food image prediction failed: {exc}") from exc

    predictions = []
    for item in raw_predictions:
        label = item.get("label", "")
        score = float(item.get("score", 0.0))
        predictions.append({
            "label": label,
            "display_name": normalize_label(label),
            "confidence": round(score, 4),
            "confidence_percent": round(score * 100, 1),
        })

    return {
        "model": MODEL_ID,
        "device": "cuda" if device == 0 else "cpu",
        "predictions": predictions,
    }
