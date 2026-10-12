"""USDA FoodData Central lookup and serving-size nutrition math."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)


API_BASE = "https://api.nal.usda.gov/fdc/v1"
NUTRIENT_IDS = {
    "calories": {1008, 2047, 2048},
    "protein": {1003},
    "fat": {1004},
    "carbs": {1005},
}


class NutritionLookupError(RuntimeError):
    """Raised for expected nutrition lookup failures."""


@dataclass
class NutritionResult:
    name: str
    fdc_id: int | None
    source: str
    per_100g: dict
    serving_grams: float
    totals: dict
    approximate: bool
    note: str

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "fdc_id": self.fdc_id,
            "source": self.source,
            "per_100g": self.per_100g,
            "serving_grams": self.serving_grams,
            "totals": self.totals,
            "approximate": self.approximate,
            "note": self.note,
        }


def calculate_totals(per_100g: dict, serving_grams: float) -> dict:
    factor = serving_grams / 100.0
    return {key: round(float(value or 0) * factor, 1) for key, value in per_100g.items()}


def _api_key() -> str | None:
    value = os.environ.get("USDA_FDC_API_KEY", "").strip()
    return value or None


def _post_json(path: str, payload: dict, timeout: int = 10) -> dict:
    api_key = _api_key()
    if not api_key:
        raise NutritionLookupError("USDA_FDC_API_KEY is not configured on the server.")
    url = f"{API_BASE}{path}?{urlencode({'api_key': api_key})}"
    body = json.dumps(payload).encode("utf-8")
    request = Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        if exc.code == 429:
            logger.warning("USDA rate limit hit (HTTP 429)")
            raise NutritionLookupError("USDA FoodData Central rate limit was reached.") from exc
        logger.warning(f"USDA returned HTTP {exc.code}")
        raise NutritionLookupError(f"USDA FoodData Central returned HTTP {exc.code}.") from exc
    except URLError as exc:
        logger.warning(f"Network error connecting to USDA: {type(exc.reason).__name__}")
        raise NutritionLookupError("Unable to reach USDA FoodData Central right now.") from exc
    except TimeoutError as exc:
        logger.warning("USDA request timed out")
        raise NutritionLookupError("USDA FoodData Central request timed out.") from exc
    except json.JSONDecodeError as exc:
        logger.warning(f"Invalid JSON from USDA: {exc.msg}")
        raise NutritionLookupError("USDA FoodData Central returned invalid data.") from exc
    except Exception as exc:
        logger.error(f"Unexpected error in USDA lookup: {type(exc).__name__}: {exc}")
        raise NutritionLookupError("Unable to reach USDA FoodData Central right now.") from exc


def _nutrient_value(food: dict, nutrient_ids: set[int]) -> float | None:
    for nutrient in food.get("foodNutrients", []):
        nutrient_id = nutrient.get("nutrientId")
        if nutrient_id is None and isinstance(nutrient.get("nutrient"), dict):
            nutrient_id = nutrient["nutrient"].get("id")
        if nutrient_id in nutrient_ids:
            value = nutrient.get("value", nutrient.get("amount"))
            if value is not None:
                return float(value)
    return None


def _extract_per_100g(food: dict) -> dict | None:
    values = {name: _nutrient_value(food, ids) for name, ids in NUTRIENT_IDS.items()}
    if all(value is None for value in values.values()):
        return None
    return {name: round(value or 0, 1) for name, value in values.items()}


def lookup_nutrition(food_name: str, serving_grams: float = 100.0) -> NutritionResult:
    query = " ".join(food_name.replace("_", " ").split())
    if not query:
        raise NutritionLookupError("Food name is required for nutrition lookup.")

    payload = {
        "query": query,
        "pageSize": 8,
        "dataType": ["Survey (FNDDS)", "SR Legacy", "Foundation"],
    }
    data = _post_json("/foods/search", payload)
    for food in data.get("foods", []):
        per_100g = _extract_per_100g(food)
        if not per_100g:
            continue
        source = food.get("dataType") or "USDA FoodData Central"
        name = food.get("description") or query
        result = NutritionResult(
            name=name.title(),
            fdc_id=food.get("fdcId"),
            source=source,
            per_100g=per_100g,
            serving_grams=serving_grams,
            totals=calculate_totals(per_100g, serving_grams),
            approximate=False,
            note="",
        )
        return result

    raise NutritionLookupError("No suitable USDA nutrition match was found for this food.")


def manual_nutrition(food_name: str, per_100g: dict, serving_grams: float = 100.0) -> NutritionResult:
    cleaned = {key: round(float(per_100g.get(key) or 0), 1) for key in ("calories", "protein", "carbs", "fat")}
    return NutritionResult(
        name=food_name.strip() or "Manual food",
        fdc_id=None,
        source="Manual estimate",
        per_100g=cleaned,
        serving_grams=serving_grams,
        totals=calculate_totals(cleaned, serving_grams),
        approximate=True,
        note="Manually entered values. Treat this as an estimate.",
    )
