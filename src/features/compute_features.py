"""
Feature computation for RecSys with anti-leakage.

Logic:
- history is the source of aggregates (only events with timestamp < target timestamp).
- target is the dataframe for which we compute features.
- For each target row, we use history events with strictly less timestamp.
- Windows: 7d, 30d, 90d.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from recipes import RECIPES, FeatureContext


INTERIM = Path("data/interim")
PROCESSED = Path("data/processed")
WINDOWS = [7, 30, 90]


# ============================================================
# Core: per-user / per-item index with cumsum
# ============================================================
class TimelineIndex:
    """
    Stores events by key (user_id or item_id).
    Allows O(log M) calculation of aggregates in [t - w, t).
    """
    def __init__(self, df: pd.DataFrame, key: str, value_col: Optional[str] = None):
        self.key = key
        self.value_col = value_col

        df = df[[key, "timestamp"] + ([value_col] if value_col else [])].copy()
        df = df.sort_values([key, "timestamp"]).reset_index(drop=True)

        self.data = {}
        for k, g in df.groupby(key, sort=False):
            ts = g["timestamp"].values.astype("datetime64[ns]")
            if value_col is not None:
                vals = g[value_col].values.astype(np.float64)
                cumsum_vals = np.cumsum(vals)
            else:
                cumsum_vals = None
            cumsum_cnt = np.cumsum(np.ones(len(g), dtype=np.int64))
            self.data[k] = (ts, cumsum_cnt, cumsum_vals)

    def _require_values(self):
        if self.value_col is None:
            raise RuntimeError(
                f"TimelineIndex(key='{self.key}') created without value_col, "
                f".mean()/.std() methods are unavailable"
            )

    def _bounds(self, k, t: np.datetime64, w: Optional[int]):
        """Returns (lower_idx, upper_idx) - half-interval [t - w, t)."""
        if k not in self.data:
            return None
        ts, _, _ = self.data[k]
        upper = np.searchsorted(ts, t, side="left")
        if w is None:
            return 0, upper
        lower_ts = t - np.timedelta64(w, "D")
        lower = np.searchsorted(ts, lower_ts, side="left")
        return lower, upper

    def count(self, k, t, w: Optional[int] = None) -> int:
        b = self._bounds(k, t, w)
        if b is None:
            return 0
        lower, upper = b
        if upper <= lower:
            return 0
        _, cumsum_cnt, _ = self.data[k]
        return int(cumsum_cnt[upper - 1] - (cumsum_cnt[lower - 1] if lower > 0 else 0))

    def mean(self, k, t, w: Optional[int] = None) -> float:
        self._require_values()
        b = self._bounds(k, t, w)
        if b is None:
            return np.nan
        lower, upper = b
        if upper <= lower:
            return np.nan
        _, cumsum_cnt, cumsum_vals = self.data[k]
        cnt = int(cumsum_cnt[upper - 1] - (cumsum_cnt[lower - 1] if lower > 0 else 0))
        if cnt == 0:
            return np.nan
        s = float(cumsum_vals[upper - 1] - (cumsum_vals[lower - 1] if lower > 0 else 0))
        return s / cnt

    def time_since_last(self, k, t) -> float:
        """Days since the last event up to t (NaN if no events)."""
        b = self._bounds(k, t, None)
        if b is None:
            return np.nan
        _, upper = b
        if upper == 0:
            return np.nan
        ts, _, _ = self.data[k]
        last_ts = ts[upper - 1]
        delta = (t - last_ts) / np.timedelta64(1, "D")
        return float(delta)


def compute_from_spec(spec, indices, target_df):
    recipe = spec["recipe"]
    fn = RECIPES[recipe]["fn"]
    return fn(indices, target_df, **spec["params"])


def compute_manual_features(train: pd.DataFrame, target: pd.DataFrame) -> pd.DataFrame:
    """
    Manual baseline: ~14 features from classic RecSys feature engineering.
    """
    print("Computing manual features...")

    # Indices
    user_idx = TimelineIndex(train, key="user_id", value_col="rating")
    item_idx = TimelineIndex(train, key="item_id", value_col="rating")

    train_ui = train.copy()
    train_ui["ui_key"] = train_ui["user_id"].astype(str) + "_" + train_ui["item_id"].astype(str)
    user_item_idx = TimelineIndex(train_ui, key="ui_key")

    train_ug = train.copy()
    train_ug["ug_key"] = train_ug["user_id"].astype(str) + "_" + train_ug["genres"].astype(str)
    user_genre_idx = TimelineIndex(train_ug, key="ug_key", value_col="rating")

    result = target.copy()
    n = len(result)

    uids = result["user_id"].values
    iids = result["item_id"].values
    ts = result["timestamp"].values.astype("datetime64[ns]")
    ui_keys = [f"{u}_{i}" for u, i in zip(uids, iids)]
    ug_keys = (
        [f"{u}_{g}" for u, g in zip(uids, result["genres"].values)]
        if "genres" in result.columns
        else [None] * n
    )

    print("  user...")
    for w in WINDOWS:
        result[f"user_rating_count_{w}d"] = [
            user_idx.count(uids[i], ts[i], w) for i in tqdm(range(n), desc=f"    user_count_{w}d", leave=False)
        ]
    result["user_rating_count_all"] = [user_idx.count(uids[i], ts[i], None) for i in tqdm(range(n), desc="    user_count_all", leave=False)]
    result["user_avg_rating_all"] = [user_idx.mean(uids[i], ts[i], None) for i in tqdm(range(n), desc="    user_avg_all", leave=False)]
    result["user_avg_rating_30d"] = [user_idx.mean(uids[i], ts[i], 30) for i in tqdm(range(n), desc="    user_avg_30d", leave=False)]

    print("  item...")
    for w in [7, 30, 90]:
        result[f"item_rating_count_{w}d"] = [
            item_idx.count(iids[i], ts[i], w) for i in tqdm(range(n), desc=f"    item_count_{w}d", leave=False)
        ]
    result["item_rating_count_all"] = [item_idx.count(iids[i], ts[i], None) for i in tqdm(range(n), desc="    item_count_all", leave=False)]
    result["item_avg_rating_all"] = [item_idx.mean(iids[i], ts[i], None) for i in tqdm(range(n), desc="    item_avg_all", leave=False)]

    print("  user_item...")
    result["user_item_count_all"] = [
        user_item_idx.count(ui_keys[i], ts[i], None) for i in tqdm(range(n), desc="    ui_count", leave=False)
    ]

    if ug_keys[0] is not None:
        print("  user_genre...")
        result["user_genre_avg_rating_30d"] = [
            user_genre_idx.mean(ug_keys[i], ts[i], 30) for i in tqdm(range(n), desc="    ug_avg_30d", leave=False)
        ]

    return result


# ============================================================
# Derivatives (generated from base features)
# ============================================================
def add_derivatives(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ratios and differences from base features.
    Provide non-linearities for GBDT.
    """
    eps = 1.0

    if {"user_rating_count_7d", "user_rating_count_30d"}.issubset(df.columns):
        df["user_activity_ratio_7_30"] = df["user_rating_count_7d"] / (df["user_rating_count_30d"] + eps)

    if {"item_rating_count_7d", "item_rating_count_30d"}.issubset(df.columns):
        df["item_popularity_ratio_7_30"] = df["item_rating_count_7d"] / (df["item_rating_count_30d"] + eps)

    if {"user_avg_rating_all", "item_avg_rating_all"}.issubset(df.columns):
        df["rating_diff_user_item"] = df["user_avg_rating_all"] - df["item_avg_rating_all"]

    return df


def compute_all_features(
    history: pd.DataFrame,
    target: pd.DataFrame,
    specs: list,
) -> pd.DataFrame:
    if "genres" not in history.columns:
        history["genres"] = "unknown"
    if "genres" not in target.columns:
        target["genres"] = "unknown"

    feat = compute_manual_features(history, target)

    ctx = FeatureContext(history)

    for spec in specs:
        recipe = spec.get("recipe")
        if recipe is None or recipe not in RECIPES:
            print(f"WARNING: unknown recipe '{recipe}', skipping {spec.get('name')}")
            continue

        params = spec.get("params", {})

        if recipe in ("genre_match",) or (
            recipe in ("entropy", "unique_ratio")
            and params.get("key") == "genre"
        ):
            print(f"skip genre-based recipe (Yambda has no genres): {spec['name']}")
            continue

        # scope=user_item not supported by time-series recipes
        if params.get("scope") == "user_item":
            print(f"skip scope='user_item': {spec['name']}")
            continue

        if (params.get("scope") == "item"
                and recipe in ("entropy", "unique_ratio")
                and params.get("key") == "genre"):
            print(f"skip degenerate item-genre: {spec['name']}")
            continue

        if (params.get("scope") == "item"
                and recipe in ("entropy", "unique_ratio", "genre_match")):
            print(f"skip degenerate item-genre: {spec['name']}")
            continue

        fn = RECIPES[recipe]["fn"]
        col = f"llm_{spec['name']}"
        try:
            feat[col] = fn(ctx, feat, **params)
        except Exception as e:
            print(f"recipe '{recipe}' failed for {spec['name']}: {e}")
            feat[col] = np.nan

    return add_derivatives(feat)

def load_feature_specs(path) -> list[dict]:
    """Reads JSON and normalizes group_key: list to tuple."""
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        specs = json.load(f)
    for s in specs:
        gk = s.get("group_key")
        if isinstance(gk, list):
            s["group_key"] = tuple(gk)
    return specs

def main():
    train = pd.read_parquet(INTERIM / "train.parquet")
    valid = pd.read_parquet(INTERIM / "valid.parquet")
    test  = pd.read_parquet(INTERIM / "test.parquet")

    specs = load_feature_specs(PROCESSED / "llm_hypotheses_v2.json")
    print(f"LLM specs: {len(specs)}")

    # Train: history=train, target=train
    print("-> train features")
    feat_train = compute_all_features(train, train, specs)
    feat_train.to_parquet(PROCESSED / "features_train.parquet")
    print(f"  saved: {feat_train.shape}")

    # Valid: history=train, target=valid (anti-leakage)
    print("-> valid features")
    feat_valid = compute_all_features(train, valid, specs)
    feat_valid.to_parquet(PROCESSED / "features_valid.parquet")
    print(f"  saved: {feat_valid.shape}")

    # Test: history=train+valid, target=test (anti-leakage)
    print("-> test features")
    train_valid = pd.concat([train, valid], ignore_index=True)
    feat_test = compute_all_features(train_valid, test, specs)
    feat_test.to_parquet(PROCESSED / "features_test.parquet")
    print(f"  saved: {feat_test.shape}")


if __name__ == "__main__":
    main()