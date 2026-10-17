import json
import os
import re
import sys
import time
from pathlib import Path
from typing import List, Dict, Optional
from json_repair import repair_json

from openai import OpenAI
from dotenv import load_dotenv

# Import recipe catalog
sys.path.insert(0, str(Path(__file__).parent))
from recipes import RECIPES  # noqa: E402

load_dotenv()

client = OpenAI(
    base_url=os.getenv("LLM_BASE_URL", "http://localhost:11434/v1"),
    api_key=os.getenv("LLM_API_KEY", "ollama"),
    timeout=600.0,
)

DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")

DATASET_SCHEMA = """
Table ratings (movie ratings):
  user_id: int
  item_id: int
  rating: int - 1..5 (TARGET)
  timestamp: datetime

Table movies:
  item_id: int
  title: str
  genres: str - separated by `|`, e.g. "Action|Comedy"

Table users:
  user_id: int
  gender: str, age: int (categorical), occupation: int (categorical), zip: str
"""

SYSTEM_PROMPT = """You generate NON-STANDARD features for RecSys by selecting a recipe.

Available recipes (choose ONLY from these):

1. entropy(scope, key, window)
   params: {"scope": "user", "key": "genre", "window": "30d"}

2. unique_ratio(scope, key, window)
   params: {"scope": "user", "key": "genre", "window": "30d"}

3. slope(scope, value_col, window)    [value_col="rating"]
   params: {"scope": "item", "value_col": "rating", "window": "90d"}

4. drift(scope, value_col, w_short, w_long)    [value_col="rating"]
   params: {"scope": "user", "value_col": "rating", "w_short": "30d", "w_long": "90d"}

5. session_count(scope, gap_hours, window)
   params: {"scope": "user", "gap_hours": 1, "window": "30d"}

6. inter_event_time(scope, window)
   Average interval between adjacent events (in hours) in the window.
   params: {"scope": "user", "window": "30d"}

7. genre_match(scope, window)
   Share of events with the same genre as the target item in the scope history.
   params: {"scope": "user", "window": "30d"}

8. cross_ratio(num_col, den_col, eps)
   num_col / (den_col + eps). Both operands are ALREADY EXISTING columns from
   baseline: user_avg_rating_all, user_avg_rating_30d, user_rating_count_30d,
   item_avg_rating_all, item_rating_count_7d, item_rating_count_30d,
   user_genre_avg_rating_30d, rating_diff_user_item, user_activity_ratio_7_30,
   item_popularity_ratio_7_30.
   params: {"num_col": "user_avg_rating_all", "den_col": "item_avg_rating_all", "eps": 1.0}

9. trend_acceleration(scope, value_col, w_recent, w_older)    [value_col="rating"]
   (mean(recent) - mean(older)) / (|mean(older)| + eps)
   params: {"scope": "user", "value_col": "rating", "w_recent": "30d", "w_older": "180d"}

10. co_occurrence(scope, key, window)    [scope="item", key="genre"]
    Share of users with the target item's genre in history over the window.
    params: {"scope": "item", "key": "genre", "window": "30d"}

11. item_age_at_event(scope, window)    [scope="item"]
    Days since the first event of the item until target timestamp.
    params: {"scope": "item", "window": "180d"}

12. repeat_ratio(scope, window)
    Share of items encountered in the scope history more than once.
    params: {"scope": "user", "window": "365d"}

RULES:
- scope in {"user", "item"} - mandatory. user_item is NOT supported.
- window in {"7d","30d","90d","180d","365d","730d","none"}.
- value_col: "rating" for slope/drift.
- eps: float, usually 1.0.

Return ONLY a JSON array without markdown. Example:
[
  {"name": "user_inter_event_time_30d", "recipe": "inter_event_time",
   "params": {"scope": "user", "window": "30d"},
   "type": "user", "rationale": "User activity density"},
  {"name": "user_genre_match_30d", "recipe": "genre_match",
   "params": {"scope": "user", "window": "30d"},
   "type": "user", "rationale": "Match of target item with genre interests"},
  {"name": "user_item_rating_ratio_llm", "recipe": "cross_ratio",
   "params": {"num_col": "user_avg_rating_all",
              "den_col": "item_avg_rating_all", "eps": 1.0},
   "type": "user", "rationale": "Ratio of average ratings"}
]

DO NOT repeat: user_avg_rating_*, user_rating_count_*, item_avg_rating_*,
item_rating_count_*, item_popularity_*.

STIMULATED: recipes 6, 7, 8 - they provide a fundamentally new signal.

Generate EXACTLY <N> features, without duplicates by name."""


def build_system_prompt(n_features: int) -> str:
    return SYSTEM_PROMPT.replace("<N>", str(n_features))

def build_user_prompt(n_features: int) -> str:
    return (
        f"Generate {n_features} DIVERSE features using recipes. "
        f"RULES: "
        f"1) scope in {{user, item}} (no user_item); "
        f"2) value_col='rating' for slope/drift/trend_acceleration; "
        f"3) DO NOT repeat a single recipe more than 5 times; "
        f"4) vary windows (7d/30d/90d/180d/365d/730d/none); "
        f"5) mix user-scope and item-scope features equally. "
        f"STIMULATED: inter_event_time, genre_match, session_count, "
        f"entropy, drift, trend_acceleration, repeat_ratio, item_age_at_event."
    )

def _sanitize_field(s: str) -> str:
    if not isinstance(s, str):
        return s
    # Strip chat-template tokens
    for tok in ("<|im_start|>", "<|im_end|>", "<|endoftext|>"):
        s = s.replace(tok, "")
    # Strip continuation text (heuristic)
    s = re.sub(r"[\u4e00-\u9fff]+.*", "", s, flags=re.DOTALL)
    return s.strip()

def generate_features(
    n_features: int = 20,
    model: str = DEFAULT_MODEL,
    verbose: bool = True,
) -> List[Dict]:
    stream = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": build_system_prompt(n_features)},
            {"role": "user", "content": build_user_prompt(n_features)},
        ],
        temperature=0.2,
        top_p=0.9,
        seed=42,
        stream=True,
    )

    chunks = []
    t0 = time.time()
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            chunks.append(delta)
            if verbose and len(chunks) % 50 == 0:
                print(f"  ...{len(''.join(chunks))} chars, {time.time()-t0:.0f}s", flush=True)

    raw = "".join(chunks).strip()

    log_dir = Path("data/processed/_llm_raw_logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    (log_dir / f"{ts}_{model.replace(':', '_')}_{n_features}f.json").write_text(
        raw, encoding="utf-8"
    )

    if verbose:
        print(f"  {time.time()-t0:.0f}s, {len(raw)} chars")
    return _extract_list(raw)


def _extract_list(raw: str) -> List[Dict]:
    """Extracts feature list, repairing invalid JSON."""
    if raw.startswith("```"):
        parts = raw.split("```")
        raw = parts[1] if len(parts) >= 3 else parts[-1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    idx = raw.find("[")
    if idx > 0:
        raw = raw[idx:]

    data = None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        try:
            data = json.loads(repair_json(raw))
        except Exception as e:
            debug_path = Path("data/processed/_failed_llm_output.txt")
            debug_path.parent.mkdir(parents=True, exist_ok=True)
            debug_path.write_text(raw, encoding="utf-8")
            raise ValueError(
                f"Failed to parse even after repair_json: {e}. "
                f"Raw output -> {debug_path}"
            )

    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("features", "hypotheses", "result", "items", "data"):
            if key in data and isinstance(data[key], list):
                return data[key]
        for value in data.values():
            if isinstance(value, list) and value and isinstance(value[0], dict):
                return value

    for item in data:
     if isinstance(item, dict):
        if "rationale" in item:
            item["rationale"] = _sanitize_field(item["rationale"])
        if "name" in item:
            item["name"] = _sanitize_field(item["name"])

    raise ValueError(f"Not a list or dict with array: type={type(data)}")


# Validation of recipe-based features
def validate_features(features: list, max_per_recipe: int = 6) -> list:
    """Validates recipe, scope, params. Deduplicates by (recipe, params)
    and limits the number of features per recipe (max_per_recipe)."""
    valid = []
    seen_names = set()
    seen_signatures = set()
    recipe_counts = {}

    NO_SCOPE_RECIPES = {"cross_ratio"}

    for f in features:
        name = f.get("name")
        recipe = f.get("recipe")
        params = f.get("params", {})

        if not name or not recipe:
            print(f"WARNING: missing name/recipe: {f.get('name', '?')}")
            continue

        if recipe not in RECIPES:
            print(f"WARNING: unknown recipe '{recipe}' for {name}")
            continue

        # scope validation
        if recipe not in NO_SCOPE_RECIPES:
            scope = params.get("scope")
            if scope not in ("user", "item"):
                print(f"WARNING: bad scope '{scope}' for {name} (must be user|item)")
                continue

        # value_col for slope/drift/trend_acceleration - only rating
        if recipe in ("slope", "drift", "trend_acceleration"):
            if params.get("value_col") != "rating":
                print(f"WARNING: {recipe} requires value_col='rating': {name}")
                continue

        if recipe == "cross_ratio":
            if not params.get("num_col") or not params.get("den_col"):
                print(f"WARNING: cross_ratio missing num_col/den_col: {name}")
                continue

        # extra params
        schema = RECIPES[recipe]["params_schema"]
        if set(params) - set(schema):
            print(f"WARNING: extra params for {recipe} in {name}: "
                  f"{set(params) - set(schema)}")
            continue

        # deduplication by signature (recipe + sorted params)
        signature = (recipe, tuple(sorted((k, str(v)) for k, v in params.items())))
        if signature in seen_signatures:
            print(f"WARNING: duplicate signature: {name}")
            continue
        seen_signatures.add(signature)

        # limit per recipe
        if recipe_counts.get(recipe, 0) >= max_per_recipe:
            print(f"WARNING: too many '{recipe}' (>{max_per_recipe}): {name}")
            continue
        recipe_counts[recipe] = recipe_counts.get(recipe, 0) + 1

        # deduplication by name
        if name in seen_names:
            continue
        seen_names.add(name)

        valid.append(f)

    return valid


# Batching
def generate_features_batched(
    n_features: int = 20,
    batch_size: int = 10,
    model: str = DEFAULT_MODEL,
) -> List[Dict]:
    all_features = []
    n_batches = (n_features + batch_size - 1) // batch_size

    for i in range(n_batches):
        remaining = n_features - len(all_features)
        if remaining <= 0:
            break
        current = min(batch_size, remaining)
        print(f"  Batch {i+1}/{n_batches} (requested {current}, have {len(all_features)})...")
        try:
            batch = generate_features(n_features=current, model=model)
            all_features.extend(batch)
        except Exception as e:
            print(f"  WARNING: Batch failed: {e}")
            continue

    return all_features[:n_features]


if __name__ == "__main__":
    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Model: {DEFAULT_MODEL}")
    print("Generating features (recipe-based, streaming + batched)...")

    target_n = 50
    features = generate_features_batched(n_features=target_n, batch_size=10)
    features = validate_features(features)

    out_path = out_dir / "llm_hypotheses_v2.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(features, f, ensure_ascii=False, indent=2)

    print(f"\n{len(features)} features -> {out_path}")
    for feat in features:
        scope = feat["params"].get("scope", "?")
        print(f"- [{scope:4}] {feat['recipe']:14} | {feat['name']}")