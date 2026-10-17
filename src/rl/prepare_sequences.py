import pandas as pd
import numpy as np
from pathlib import Path
import json

RAW = Path("data/raw/yambda")
INTERIM = Path("data/interim/rl")
INTERIM.mkdir(parents=True, exist_ok=True)

def load_listens():
    df = pd.read_parquet(RAW / "listens.parquet")
    df = df.rename(columns={"uid": "user_id"})
    df["timestamp"] = pd.to_datetime(df["timestamp"].astype("int64") * 5, unit="s")
    return df

def main():
    print("Loading data...")
    df = load_listens()
    
    rng = np.random.default_rng(seed=42)
    users = rng.choice(df["user_id"].unique(), size=3000, replace=False)
    df = df[df["user_id"].isin(users)].copy()
    
    # Создаем компактный словарь
    unique_items = df["item_id"].unique()
    item2idx = {int(item): idx+1 for idx, item in enumerate(unique_items)} # 0 для padding
    print(f"Unique items: {len(item2idx)}")
    
    with open(INTERIM / "item2idx.json", "w") as f:
        json.dump(item2idx, f)
    
    # Применяем маппинг
    df["item_idx"] = df["item_id"].map(item2idx)
    
    # Time split
    df = df.sort_values("timestamp")
    cut70 = df["timestamp"].quantile(0.70)
    cut85 = df["timestamp"].quantile(0.85)
    
    train = df[df["timestamp"] <= cut70]
    valid = df[(df["timestamp"] > cut70) & (df["timestamp"] <= cut85)]
    test = df[df["timestamp"] > cut85]
    
    def build_seqs(subset):
        return (subset.sort_values(["user_id", "timestamp"])
                .groupby("user_id")
                .agg(sequence=("item_idx", list), rewards=("played_ratio_pct", list))
                .reset_index())
                
    train_seqs = build_seqs(train)
    valid_seqs = build_seqs(valid)
    test_seqs = build_seqs(test)
    
    train_seqs = train_seqs[train_seqs["sequence"].apply(len) >= 2]
    valid_seqs = valid_seqs[valid_seqs["sequence"].apply(len) >= 2]
    test_seqs = test_seqs[test_seqs["sequence"].apply(len) >= 2]
    
    train_seqs.to_parquet(INTERIM / "seq_train.parquet")
    valid_seqs.to_parquet(INTERIM / "seq_valid.parquet")
    test_seqs.to_parquet(INTERIM / "seq_test.parquet")
    print(f"Saved: Train {len(train_seqs)}, Valid {len(valid_seqs)}, Test {len(test_seqs)} sequences.")

if __name__ == "__main__":
    main()