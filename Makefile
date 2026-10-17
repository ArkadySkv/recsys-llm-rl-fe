# RecSys LLM+RL Feature Engineering (Yambda-50M) — Pipeline Makefile

.PHONY: all help split features exhaustive select final figures experiments report clean

help:
	@echo "Доступные цели:"
	@echo "  make all         — полный пайплайн (split -> features -> exhaustive -> select -> final)"
	@echo "  make split       — time-based train/valid/test split (анти-лик)"
	@echo "  make features    — ручные + LLM-фичи"
	@echo "  make exhaustive  — exhaustive search на валидации"
	@echo "  make select      — выбор лучшего subset LLM-фичей"
	@echo "  make final       — one-shot оценка на test"
	@echo "  make figures     — графики для отчета"
	@echo "  make experiments — копирование артефактов в experiments/"
	@echo "  make clean       — удалить промежуточные файлы"

all: split features exhaustive select final figures experiments report

rl-all: rl-prepare rl-train-gpt rl-train-reward rl-train-ppo rl-evaluate rl-figures

# Данные и фичи 
split:
	@echo "[split] time-based train/valid/test split..."
	python3 -m src.data.split

features: split
	@echo "[features] расчёт фич для train/valid/test..."
	python3 -m src.features.compute_features

# Selection на Valid 
exhaustive: features
	@echo "[exhaustive] поиск по 2^K подмножествам на valid (~50 мин)..."
	rm -f data/processed/exhaustive_checkpoint.json
	python3 -m src.models.gbdt_exhaustive --eval-split valid

select: exhaustive
	@echo "[select] копирую top-1 subset → selected_features.json"
	python3 -c "import pandas as pd, json, pathlib; \
	  df = pd.read_csv('data/processed/exhaustive_results.csv'); \
	  best = df.sort_values('ndcg', ascending=False).iloc[0]['subset'].split('|'); \
	  pathlib.Path('experiments/002_llm_rl').mkdir(parents=True, exist_ok=True); \
	  json.dump(best, open('experiments/002_llm_rl/selected_features.json','w'), indent=2); \
	  print('selected:', best)"

# Финальная оценка на Test 
final: select
	@echo "[final] обучение на train+valid, оценка на test..."
	python3 -m src.models.gbdt_final

# Визуализация и артефакты 
figures:
	@echo "[figures] генерация графиков..."
	python3 -m src.visualization.make_figures
	@echo "  ✓ reports/figures/*.png"

experiments:
	@echo "[experiments] копирую финальные артефакты..."
	@mkdir -p experiments/001_baseline experiments/002_llm_rl
	# Копируем важность фичей от baseline (если есть)
	@cp data/processed/gbdt_feature_importance.csv experiments/001_baseline/ 2>/dev/null || true
	# Копируем финальные метрики и выбранный subset
	@cp experiments/002_llm_rl/test_metrics_final.csv experiments/002_llm_rl/ 2>/dev/null || true
	@cp data/processed/exhaustive_results.csv experiments/002_llm_rl/ 2>/dev/null || true
	@echo "  ✓ experiments/ заполнены"

report:
	@echo ""
	@echo "════════════════════════════════════════════════════════"
	@echo "  Артефакты:"
	@echo "    reports/figures/     — графики для отчета"
	@echo "    experiments/         — метрики по этапам"
	@echo "    data/processed/      — фичи и CSV"
	@echo "  Отчет: reports/development_log.md"
	@echo "════════════════════════════════════════════════════════"

clean:
	@echo "[clean] удаляю промежуточные артефакты..."
	rm -f data/processed/*.parquet data/processed/*.csv
	rm -f reports/figures/*.png
	@echo " очищено (raw-данные и код сохранены)"

rl-prepare:
	@echo "[rl] Preparing user sequences..."
	python3 -m src.rl.prepare_sequences

rl-train-gpt: rl-prepare
	@echo "[rl] Training GPT-2 next-item predictor..."
	python3 -m src.rl.train_gpt

rl-train-reward: rl-prepare
	@echo "[rl] Training reward model..."
	python3 -m src.rl.train_reward_model

rl-train-ppo: rl-train-gpt rl-train-reward
	@echo "[rl] Fine-tuning with REINFORCE..."
	python3 -m src.rl.train_rl

rl-evaluate: rl-train-ppo
	@echo "[rl] Evaluating base vs RL-tuned models..."
	python3 -m src.rl.evaluate

rl-figures:
	@echo "[rl] Generating plots for Task 2..."
	python3 -m src.visualization.make_figures_rl