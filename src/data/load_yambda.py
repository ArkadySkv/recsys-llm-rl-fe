"""
Loader for Yambda-50M from HuggingFace.

Structure of listens.parquet:
  uid, item_id, timestamp, is_organic, played_ratio_pct, track_length_seconds

Schema is mapped to pipeline-compatible format:
  user_id, item_id, timestamp (datetime64), label, played_ratio_pct, is_organic

label = 1 if played_ratio_pct >= 50.
"""
from pathlib import Path
import pandas as pd
from datasets import load_dataset


RAW = Path("data/raw/yambda")


def download_yambda(size: str = "50m", force: bool = False) -> None:
    """Downloads listens.parquet and likes.parquet for the given size.

    Args:
        size: "50m", "500m" or "5b"
        force: overwrite existing files
    """
    RAW.mkdir(parents=True, exist_ok=True)

    for fname in ["listens.parquet", "likes.parquet"]:
        out_path = RAW / fname
        if out_path.exists() and not force:
            print(f"  INFO: {fname} already downloaded: {out_path}")
            continue

        print(f"  Downloading {fname} (size={size})...")
        ds = load_dataset(
            "yandex/yambda",
            data_dir=f"flat/{size}",
            data_files=fname,
            split="train",
        )
        df = ds.to_pandas()
        df.to_parquet(out_path, index=False)
        print(f"  INFO: {fname}: {len(df):,} rows -> {out_path}")


def load_listens() -> pd.DataFrame:
    """Loads listens.parquet and normalizes schema."""
    path = RAW / "listens.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run download_yambda() first."
        )

    df = pd.read_parquet(path)

    df = df.rename(columns={"uid": "user_id"})

    # timestamp: delta times in 5s units -> datetime64 for TimelineIndex.
    # Multiply by 5 to get real seconds.
    df["timestamp"] = pd.to_datetime(df["timestamp"].astype("int64") * 5, unit="s")

    df["label"] = (df["played_ratio_pct"] >= 50).astype(int)

    df["event_type"] = "listen"
    
    df["rating"] = (df["played_ratio_pct"] / 20.0).clip(1, 5)

    return df[
        ["user_id", "item_id", "timestamp",
         "rating", "label",
         "played_ratio_pct", "is_organic"]
    ].copy()


if __name__ == "__main__":
    download_yambda(size="50m")
    df = load_listens()
    print(f"\nLoaded listens: {df.shape}")
    print(df.head())
    print(f"\nUsers: {df['user_id'].nunique():,}")
    print(f"Items: {df['item_id'].nunique():,}")
    print(f"Label distribution:\n{df['label'].value_counts(normalize=True)}")