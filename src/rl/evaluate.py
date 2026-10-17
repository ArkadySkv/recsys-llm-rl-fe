import torch
import pandas as pd
import numpy as np
from pathlib import Path
from transformers import GPT2LMHeadModel
from src.rl.train_reward_model import RewardModel

INTERIM = Path("data/interim/rl")
MAX_LEN = 30
K = 10

def evaluate_model(model, reward_model, test_seqs, device):
    model.eval()
    reward_model.eval()
    
    ndcgs = []
    rewards = []
    generated_items = []
    
    for s in test_seqs:
        if len(s) < 2: continue
        
        # Оставляем место для 1 нового токена (максимум MAX_LEN - 1)
        hist = s[:-1][-(MAX_LEN - 1):]
        input_seq = torch.tensor(hist).unsqueeze(0).to(device)
        target = s[-1]
        
        with torch.no_grad():
            # Получаем логиты для следующего токена
            logits = model(input_seq).logits[0, -1, :]
            # Берем Top-K предсказаний
            top_k_items = torch.topk(logits, K).indices.tolist()
            
            # 1. NDCG@10 (бинарная релевантность)
            rel = [1 if x == target else 0 for x in top_k_items]
            if sum(rel) == 0:
                ndcgs.append(0.0)
            else:
                rank = rel.index(1) + 1
                ndcgs.append(1.0 / np.log2(rank + 1))
                
            # 2. Reward для лучшего сгенерированного айтема (Top-1)
            best_item = top_k_items[0]
            generated_items.append(best_item)
            
            # Формируем последовательность: история + лучший айтем (длина <= MAX_LEN)
            cand_seq = torch.cat([input_seq, torch.tensor([[best_item]]).to(device)], dim=1)
            # Обрезаем на всякий случай
            cand_seq = cand_seq[:, -MAX_LEN:]
            
            r = reward_model(cand_seq).item()
            rewards.append(r)
            
    # 3. Diversity (количество уникальных айтемов)
    unique_items_count = len(set(generated_items))
    
    return {
        "NDCG@10": np.mean(ndcgs) if ndcgs else 0.0,
        "Avg Reward": np.mean(rewards) if rewards else 0.0,
        "Unique Items": unique_items_count,
        "Total Users": len(generated_items)
    }

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on {device}...")

    # Загружаем тестовые последовательности (берем 1000 для скорости)
    test_df = pd.read_parquet(INTERIM / "seq_test.parquet")
    test_seqs = test_df["sequence"].tolist()[:1000]
    
    # Загружаем базовую модель, чтобы узнать vocab_size
    base_model = GPT2LMHeadModel.from_pretrained("models/gpt2_rec").to(device)
    vocab_size = base_model.config.vocab_size
    
    # Загружаем Reward Model
    rm = RewardModel(vocab_size).to(device)
    rm.load_state_dict(torch.load("models/reward_model/rm.pt", map_location=device))
    
    print("\n--- Evaluating Base Model ---")
    base_metrics = evaluate_model(base_model, rm, test_seqs, device)
    print(base_metrics)
    
    print("\n--- Evaluating RL-Tuned Model ---")
    try:
        rl_model = GPT2LMHeadModel.from_pretrained("models/gpt2_rec_rl").to(device)
        rl_metrics = evaluate_model(rl_model, rm, test_seqs, device)
        print(rl_metrics)
        
        print("\n=== TRADE-OFF ANALYSIS ===")
        print(f"NDCG Change:   {rl_metrics['NDCG@10'] - base_metrics['NDCG@10']:+.4f}")
        print(f"Reward Change: {rl_metrics['Avg Reward'] - base_metrics['Avg Reward']:+.4f}")
        print(f"Diversity:     Base = {base_metrics['Unique Items']}, RL = {rl_metrics['Unique Items']}")
    except OSError:
        print("RL model not found. Please run src/rl/train_rl.py first.")

if __name__ == "__main__":
    main()