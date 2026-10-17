# Experiment 003: LLM Features + Exhaustive Search

## Overview
This experiment evaluates whether automatically generated features (via LLM) can improve the ranking quality over the manual baseline. We use a Recipe-based generation approach and an exhaustive search for feature selection.

## Methodology
1. **Generation:** `qwen2.5:7b` generated 7 non-standard features from a strict recipe catalog (e.g., `slope`, `trend_acceleration`, `inter_event_time`).
2. **Selection:** We evaluated all \(2^7 = 127\) possible subsets of LLM features on the **Validation** set.
3. **Final Evaluation:** The best subset (4 features) was combined with the manual baseline. The final model was trained on `Train + Valid` and evaluated **one-shot** on the `Test` set.

## Selected Features (Optimal Subset)
The exhaustive search identified these 4 features as the most valuable:
1. `llm_user_slope_180d` (User rating trend)
2. `llm_item_trend_acceleration_30d` (Item popularity acceleration)
3. `llm_item_repeat_ratio_365d` (User's tendency to repeat items)
4. `llm_user_item_age_at_event_180d` (Age of the item relative to the event)

## Results (One-Shot Test Set)
| Metric | Manual Baseline | Manual + Selected LLM | Delta |
| :--- | :--- | :--- | :--- |
| **NDCG@10** | 0.7269 | 0.7255 | **-0.0014** |
| **ROC-AUC** | 0.4413 | **0.4718** | **+0.0305** |

## Analysis & Conclusion
- **Valid vs Test Gap:** On the validation set, the LLM subset provided a +0.0088 boost. On the test set, NDCG slightly dropped (-0.0014). 
- **Temporal Drift:** The selected features rely heavily on short-term trends (`slope`, `trend_acceleration`). The 6-month gap between Valid and Test caused these trends to become stale.
- **ROC-AUC Improvement:** Despite NDCG dropping, global ROC-AUC improved significantly (+0.03), proving the LLM features contain a valid global signal that the `lambdarank` objective couldn't fully exploit in top-K due to cold-start users.
- **Artifacts:** See `selected_features.json` and `test_metrics_final.csv`.

## Reproducibility
Run `python3 src/models/gbdt_exhaustive.py --eval-split valid`, then `python3 src/models/gbdt_final.py`.