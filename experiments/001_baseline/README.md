# Experiment 001: Manual Baseline

## Overview
This experiment establishes the performance baseline using only hand-crafted, classic RecSys features. It serves as the control group to evaluate the effectiveness of LLM-generated features.

## Dataset & Protocol
- **Data:** Yambda-50M subsample (3,000 users with full history).
- **Split:** Strict time-based split (70% Train, 15% Valid, 15% Test).
- **Target:** `label = 1` if `played_ratio_pct >= 50`, else `0`.
- **Evaluation:** NDCG@10 and Recall@10 on users with >= 10 test interactions.

## Features (16 total)
Classic sliding window aggregations calculated with `TimelineIndex` (strict anti-leakage):
- **User:** count of events (7d, 30d, 90d, all), avg rating (all, 30d).
- **Item:** count of events (7d, 30d, 90d, all), avg rating (all).
- **User-Item / Derivatives:** `user_item_count_all`, `rating_diff_user_item`, `user_activity_ratio_7_30`, etc.
- *Note:* Target-proxy columns (`played_ratio_pct`, `is_organic`) are explicitly excluded via `LEAK_COLS` to prevent data leakage.

## Model
- **Algorithm:** LightGBM `LGBMRanker` (`objective="lambdarank"`, `num_leaves=63`).
- **Grouping:** By `user_id` (groups with >10,000 events are filtered out due to LightGBM limits).

## Results (Test Set)
| Metric | Score |
| :--- | :--- |
| **NDCG@10** | 0.7269 |
| **Recall@10** | 0.0731 |

## Reproducibility
Run `python3 src/models/gbdt.py`.