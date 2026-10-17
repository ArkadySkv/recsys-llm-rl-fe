"""Recipe catalog. Each recipe = function + JSON-schema of parameters.

Recipe functions receive:
  ctx     - FeatureContext (by-user and by-item indices)
  target  - DataFrame with user_id, item_id, timestamp, genres and already computed
            manual features (for cross_ratio)
  **params - recipe parameters

Return np.ndarray of length len(target).
"""
from __future__ import annotations
from typing import Callable
import numpy as np
import pandas as pd


RECIPES: dict[str, dict] = {}


def register(name: str, params_schema: dict, description: str):
    def deco(fn: Callable):
        RECIPES[name] = {
            "fn": fn,
            "params_schema": params_schema,
            "description": description,
        }
        return fn
    return deco


WINDOW_DAYS = {
    "7d": 7, "30d": 30, "90d": 90, "180d": 180, "365d": 365,
    "730d": 730, "2y": 730,
    "none": None, "all": None, "all_time": None,
}


def _window_days(w):
    if w is None or isinstance(w, (int, np.integer)):
        return w
    w_str = str(w).lower().strip()
    if w_str not in WINDOW_DAYS:
        raise ValueError(
            f"Unknown window '{w}'. Allowed: {sorted(WINDOW_DAYS)}"
        )
    return WINDOW_DAYS[w_str]


# ============================================================
# FeatureContext
# ============================================================
class FeatureContext:
    def __init__(self, history: pd.DataFrame):
        h = history.sort_values(["user_id", "timestamp"]).reset_index(drop=True)
        self.by_user = {uid: self._pack(g)
                        for uid, g in h.groupby("user_id", sort=False)}
        self.by_item = {iid: self._pack(g)
                        for iid, g in h.groupby("item_id", sort=False)}

    @staticmethod
    def _pack(g: pd.DataFrame) -> dict:
        return {
            "ts":      g["timestamp"].values.astype("datetime64[ns]"),
            "rating":  g["rating"].values.astype(np.float64),
            "genre":   (g["genres"].values if "genres" in g.columns
                        else np.array([None] * len(g), dtype=object)),
            "item_id": g["item_id"].values,
            "user_id": g["user_id"].values,
        }

    def slice(self, scope: str, uid, iid, t: np.datetime64, window_days):
        if scope == "user":
            d = self.by_user.get(uid)
        elif scope == "item":
            d = self.by_item.get(iid)
        else:
            return None
        if d is None:
            return None
        ts = d["ts"]
        upper = int(np.searchsorted(ts, t, side="left"))
        if window_days is None:
            lower = 0
        else:
            lower_ts = t - np.timedelta64(int(window_days), "D")
            lower = int(np.searchsorted(ts, lower_ts, side="left"))
        if upper <= lower:
            return None
        return {k: v[lower:upper] for k, v in d.items()}


def _get_series(sl: dict, key: str) -> np.ndarray:
    if key == "genre":
        return sl["genre"]
    return sl.get(key, sl["genre"])


def _drop_na(vals) -> list:
    return [v for v in vals
            if v is not None
            and not (isinstance(v, float) and np.isnan(v))
            and v != ""]


# ============================================================
# Base recipes (5 from 4.25)
# ============================================================

@register("entropy",
          {"scope": "str", "key": "str", "window": "str"},
          "Shannon entropy of key distribution in window. scope in {user, item}")
def _entropy(ctx, target, scope, key, window):
    w = _window_days(window)
    out = np.full(len(target), np.nan)
    for i, row in enumerate(target.itertuples(index=False)):
        t = np.datetime64(row.timestamp, "ns")
        sl = ctx.slice(scope, row.user_id, row.item_id, t, w)
        if sl is None:
            continue
        vals = _drop_na(_get_series(sl, key))
        if len(vals) < 2:
            continue
        _, counts = np.unique(vals, return_counts=True)
        p = counts / counts.sum()
        out[i] = float(-np.sum(p * np.log(p + 1e-12)))
    return out


@register("unique_ratio",
          {"scope": "str", "key": "str", "window": "str"},
          "unique(key) / count(key) in window")
def _unique_ratio(ctx, target, scope, key, window):
    w = _window_days(window)
    out = np.full(len(target), np.nan)
    for i, row in enumerate(target.itertuples(index=False)):
        t = np.datetime64(row.timestamp, "ns")
        sl = ctx.slice(scope, row.user_id, row.item_id, t, w)
        if sl is None:
            continue
        vals = _drop_na(_get_series(sl, key))
        if not vals:
            continue
        out[i] = len(set(vals)) / len(vals)
    return out


@register("slope",
          {"scope": "str", "value_col": "str", "window": "str"},
          "Slope of regression value_col ~ timestamp. value_col='rating'")
def _slope(ctx, target, scope, value_col, window):
    if value_col != "rating":
        raise ValueError(f"slope: value_col must be 'rating', got '{value_col}'")
    w = _window_days(window)
    out = np.full(len(target), np.nan)
    for i, row in enumerate(target.itertuples(index=False)):
        t = np.datetime64(row.timestamp, "ns")
        sl = ctx.slice(scope, row.user_id, row.item_id, t, w)
        if sl is None or len(sl["ts"]) < 2:
            continue
        ts = sl["ts"].astype("int64").astype(np.float64)
        ts = ts - ts.mean()
        if ts.std() < 1e-9:
            continue
        try:
            out[i] = float(np.polyfit(ts, sl["rating"], 1)[0])
        except Exception:
            continue
    return out


@register("drift",
          {"scope": "str", "value_col": "str", "w_short": "str", "w_long": "str"},
          "mean(value, w_short) - mean(value, w_long)")
def _drift(ctx, target, scope, value_col, w_short, w_long):
    if value_col != "rating":
        raise ValueError(f"drift: value_col must be 'rating', got '{value_col}'")
    ws, wl = _window_days(w_short), _window_days(w_long)
    out = np.full(len(target), np.nan)
    for i, row in enumerate(target.itertuples(index=False)):
        t = np.datetime64(row.timestamp, "ns")
        s_s = ctx.slice(scope, row.user_id, row.item_id, t, ws)
        s_l = ctx.slice(scope, row.user_id, row.item_id, t, wl)
        if s_s is None or s_l is None:
            continue
        if len(s_s["rating"]) == 0 or len(s_l["rating"]) == 0:
            continue
        out[i] = float(s_s["rating"].mean() - s_l["rating"].mean())
    return out


@register("session_count",
          {"scope": "str", "gap_hours": "int", "window": "str"},
          "Number of sessions with gap > gap_hours")
def _session_count(ctx, target, scope, gap_hours, window):
    w = _window_days(window)
    gap = np.timedelta64(int(gap_hours), "h")
    out = np.full(len(target), np.nan)
    for i, row in enumerate(target.itertuples(index=False)):
        t = np.datetime64(row.timestamp, "ns")
        sl = ctx.slice(scope, row.user_id, row.item_id, t, w)
        if sl is None or len(sl["ts"]) == 0:
            continue
        ts = sl["ts"]
        if len(ts) == 1:
            out[i] = 1.0
            continue
        out[i] = float(1 + int((np.diff(ts) > gap).sum()))
    return out


# ============================================================
# Extended catalog
# ============================================================

@register("inter_event_time",
          {"scope": "str", "window": "str"},
          "Average interval between adjacent events in hours")
def _inter_event_time(ctx, target, scope, window):
    w = _window_days(window)
    out = np.full(len(target), np.nan)
    for i, row in enumerate(target.itertuples(index=False)):
        t = np.datetime64(row.timestamp, "ns")
        sl = ctx.slice(scope, row.user_id, row.item_id, t, w)
        if sl is None or len(sl["ts"]) < 2:
            continue
        diffs_h = np.diff(sl["ts"]) / np.timedelta64(1, "h")
        out[i] = float(diffs_h.mean())
    return out


@register("genre_match",
          {"scope": "str", "window": "str"},
          "Share of events with the same genre as target item in scope history")
def _genre_match(ctx, target, scope, window):
    w = _window_days(window)
    out = np.full(len(target), np.nan)
    has_genres = "genres" in target.columns
    for i, row in enumerate(target.itertuples(index=False)):
        t = np.datetime64(row.timestamp, "ns")
        target_genre = getattr(row, "genres", None) if has_genres else None
        if target_genre is None or (isinstance(target_genre, float) and np.isnan(target_genre)):
            continue
        sl = ctx.slice(scope, row.user_id, row.item_id, t, w)
        if sl is None:
            continue
        genres = _drop_na(sl["genre"])
        if not genres:
            continue
        out[i] = sum(1 for g in genres if g == target_genre) / len(genres)
    return out


@register("cross_ratio",
          {"num_col": "str", "den_col": "str", "eps": "float"},
          "num_col / (den_col + eps). Operands are already computed columns.")
def _cross_ratio(ctx, target, num_col, den_col, eps=1.0):
    if num_col not in target.columns:
        raise ValueError(f"cross_ratio: '{num_col}' not in columns")
    if den_col not in target.columns:
        raise ValueError(f"cross_ratio: '{den_col}' not in columns")
    num = target[num_col].values.astype(np.float64)
    den = target[den_col].values.astype(np.float64)
    return num / (den + eps)


@register("trend_acceleration",
          {"scope": "str", "value_col": "str",
           "w_recent": "str", "w_older": "str"},
          "Change in mean value between adjacent windows relative to baseline")
def _trend_acceleration(ctx, target, scope, value_col,
                        w_recent, w_older):
    if value_col != "rating":
        raise ValueError(f"trend_acceleration: value_col must be 'rating'")
    wr, wo = _window_days(w_recent), _window_days(w_older)
    out = np.full(len(target), np.nan)
    for i, row in enumerate(target.itertuples(index=False)):
        t = np.datetime64(row.timestamp, "ns")
        s_r = ctx.slice(scope, row.user_id, row.item_id, t, wr)
        s_o = ctx.slice(scope, row.user_id, row.item_id, t, wo)
        if s_r is None or s_o is None:
            continue
        if len(s_r["rating"]) == 0 or len(s_o["rating"]) == 0:
            continue
        mean_r = s_r["rating"].mean()
        mean_o = s_o["rating"].mean()
        out[i] = (mean_r - mean_o) / (abs(mean_o) + 1e-6)
    return out


@register("co_occurrence",
          {"scope": "str", "key": "str", "window": "str"},
          "Share of users who rated target item among users with history_key=key")
def _co_occurrence(ctx, target, scope, key, window):
    """For scope='item': for each target item - share of users who rated item,
    among those who interacted with key in the window."""
    if scope != "item":
        raise ValueError("co_occurrence supports only scope='item'")
    w = _window_days(window)
    has_genres = "genres" in target.columns
    out = np.full(len(target), np.nan)
    for i, row in enumerate(target.itertuples(index=False)):
        t = np.datetime64(row.timestamp, "ns")
        target_genre = getattr(row, "genres", None) if has_genres else None
        if target_genre is None:
            continue
        # users with target_genre in window
        matching_users = 0
        total_users = 0
        for uid, d in ctx.by_user.items():
            sl = ctx.slice("user", uid, None, t, w)
            if sl is None:
                continue
            total_users += 1
            if target_genre in _drop_na(sl["genre"]):
                matching_users += 1
        if total_users == 0:
            continue
        out[i] = matching_users / total_users
    return out


@register("item_age_at_event",
          {"scope": "str", "window": "str"},
          "Days since the first event of item until target timestamp (cap by window)")
def _item_age_at_event(ctx, target, scope, window):
    if scope != "item":
        raise ValueError("item_age_at_event supports only scope='item'")
    w = _window_days(window)
    out = np.full(len(target), np.nan)
    for i, row in enumerate(target.itertuples(index=False)):
        t = np.datetime64(row.timestamp, "ns")
        d = ctx.by_item.get(row.item_id)
        if d is None or len(d["ts"]) == 0:
            continue
        first_ts = d["ts"][0]
        if first_ts >= t:
            continue
        age_days = (t - first_ts) / np.timedelta64(1, "D")
        if w is not None:
            age_days = min(age_days, float(w))
        out[i] = float(age_days)
    return out


@register("repeat_ratio",
          {"scope": "str", "window": "str"},
          "Share of repeated interactions (item encountered more than once in scope)")
def _repeat_ratio(ctx, target, scope, window):
    w = _window_days(window)
    out = np.full(len(target), np.nan)
    for i, row in enumerate(target.itertuples(index=False)):
        t = np.datetime64(row.timestamp, "ns")
        sl = ctx.slice(scope, row.user_id, row.item_id, t, w)
        if sl is None or len(sl["item_id"]) == 0:
            continue
        items = sl["item_id"]
        _, counts = np.unique(items, return_counts=True)
        out[i] = float((counts > 1).sum() / len(counts))
    return out