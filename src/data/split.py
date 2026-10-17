"""
Time-based train/valid/test split for Yambda.

Strategy:
  1. unique timestamps, sorted.
  2. Cumulative number of events.
  3. Cutoff at 70% (train/valid) and 85% (valid/test).
  4. Anti-leak assertions.
"""
from pathlib import Path
import pandas as pd
import numpy as np

from src.data.load_yambda import load_listens, download_yambda

INTERIM = Path("data/interim")
RAW = Path("data/raw/yambda")

TRAIN_RATIO = 0.70
VALID_RATIO = 0.15
# test = 1 - TRAIN_RATIO - VALID_RATIO = 0.15


def split_by_time(df: pd.DataFrame):
    """Splits by unique timestamps - no leakage between splits."""
    ts_counts = df.groupby("timestamp").size().sort_index()
    cum = ts_counts.cumsum()
    total = cum.iloc[-1]

    cutoff_70 = (cum >= total * TRAIN_RATIO).idxmax()
    cutoff_85 = (cum >= total * (TRAIN_RATIO + VALID_RATIO)).idxmax()

    train = df[df["timestamp"] <= cutoff_70].copy()
    valid = df[(df["timestamp"] > cutoff_70) & (df["timestamp"] <= cutoff_85)].copy()
    test  = df[df["timestamp"] > cutoff_85].copy()

    # Anti-leak assertions
    assert train["timestamp"].max() < valid["timestamp"].min(), "Leak train/valid"
    assert valid["timestamp"].max() < test["timestamp"].min(), "Leak valid/test"

    print(f"Cutoff 70%: {cutoff_70}")
    print(f"Cutoff 85%: {cutoff_85}")
    print(f"Train: {len(train):>12,} rows  ({train.timestamp.min()} - {train.timestamp.max()})")
    print(f"Valid: {len(valid):>12,} rows  ({valid.timestamp.min()} - {valid.timestamp.max()})")
    print(f"Test:  {len(test):>12,} rows  ({test.timestamp.min()} - {test.timestamp.max()})")

    print(f"\nUsers per split:")
    print(f"  train: {train['user_id'].nunique():>8,}")
    print(f"  valid: {valid['user_id'].nunique():>8,}")
    print(f"  test:  {test['user_id'].nunique():>8,}")

    return train, valid, test


def main():
    download_yambda(size="50m")
    df = load_listens()
    print(f"\nLoaded Yambda-50M: {len(df):,} events, "
          f"{df['user_id'].nunique():,} users, "
          f"{df['item_id'].nunique():,} items")

    N_USERS = 3000
    rng = np.random.default_rng(seed=42)
    all_users = df["user_id"].unique()
    if len(all_users) > N_USERS:
        sampled = rng.choice(all_users, size=N_USERS, replace=False)
        df = df[df["user_id"].isin(sampled)].copy()
        print(f"Subsampled to {N_USERS:,} users: {len(df):,} events")

    train, valid, test = split_by_time(df)

    INTERIM.mkdir(parents=True, exist_ok=True)
    train.to_parquet(INTERIM / "train.parquet")
    valid.to_parquet(INTERIM / "valid.parquet")
    test.to_parquet(INTERIM / "test.parquet")
    print("\n  INFO: data/interim/{train,valid,test}.parquet created")


if __name__ == "__main__":
    main()