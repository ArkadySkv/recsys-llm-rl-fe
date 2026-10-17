import torch
import torch.nn.functional as F
import pandas as pd
from pathlib import Path
from transformers import GPT2LMHeadModel
from src.rl.train_reward_model import RewardModel

INTERIM = Path("data/interim/rl")
MODEL_DIR = Path("models/gpt2_rec")
RL_DIR = Path("models/gpt2_rec_rl")
RL_DIR.mkdir(parents=True, exist_ok=True)

MAX_LEN = 30
BATCH_SIZE = 16
EPOCHS = 2
BETA = 0.1

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training RL on {device}...")
    
    policy = GPT2LMHeadModel.from_pretrained(MODEL_DIR).to(device)
    ref_policy = GPT2LMHeadModel.from_pretrained(MODEL_DIR).to(device)
    ref_policy.eval()
    for param in ref_policy.parameters():
        param.requires_grad = False 
        
    rm = RewardModel(policy.config.vocab_size).to(device)
    rm.load_state_dict(torch.load("models/reward_model/rm.pt", map_location=device))
    rm.eval()

    opt = torch.optim.Adam(policy.parameters(), lr=1e-5)
    ds = pd.read_parquet(INTERIM / "seq_train.parquet")["sequence"].tolist()

    policy.train()
    for epoch in range(EPOCHS):
        total_reward = 0
        total_loss = 0
        steps = 0
        
        for i in range(0, min(len(ds), 3000), BATCH_SIZE):
            batch_seqs = ds[i:i+BATCH_SIZE]
            
            # 1. Manual Left-Padding
            seqs = [torch.tensor(s[-MAX_LEN:]) for s in batch_seqs]
            max_len = max(len(s) for s in seqs)
            inputs_list = []
            for s in seqs:
                pad_len = max_len - len(s)
                # Дополняем нулями СЛЕВА (left-padding)
                padded = torch.cat([torch.zeros(pad_len, dtype=torch.long), s])
                inputs_list.append(padded)
            inputs = torch.stack(inputs_list).to(device)
            
            # 2. Attention Mask (нули = паддинг, их нужно игнорировать)
            attention_mask = (inputs != 0).long().to(device)
            
            # 3. Generate next item
            outputs = policy.generate(
                inputs, 
                attention_mask=attention_mask,
                max_new_tokens=1, 
                do_sample=True, 
                top_k=50,
                pad_token_id=0
            )
            
            # Обрезаем до MAX_LEN, чтобы не переполнить позиционные эмбеддинги
            outputs = outputs[:, -MAX_LEN:]
            
            # Защита от выхода за границы словаря
            vocab_size = policy.config.vocab_size
            outputs = outputs.clamp(0, vocab_size - 1)
            
            generated_items = outputs[:, -1]
            
            with torch.no_grad():
                rewards = rm(outputs).detach()
            
            logits_policy = policy(outputs[:, :-1]).logits[:, -1, :]
            log_probs = F.log_softmax(logits_policy, dim=-1)
            action_log_probs = log_probs.gather(1, generated_items.unsqueeze(1)).squeeze(1)
            
            with torch.no_grad():
                logits_ref = ref_policy(outputs[:, :-1]).logits[:, -1, :]
                probs_ref = F.softmax(logits_ref, dim=-1)
                
            kl_div = F.kl_div(log_probs, probs_ref, reduction='batchmean')
            loss = -(action_log_probs * rewards).mean() + BETA * kl_div
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            total_reward += rewards.mean().item()
            total_loss += loss.item()
            steps += 1
            
        print(f"Epoch {epoch+1}/{EPOCHS} | Mean Reward: {total_reward/steps:.4f} | Loss: {total_loss/steps:.4f}")

    policy.save_pretrained(RL_DIR)
    print(f"RL model saved to {RL_DIR}")

if __name__ == "__main__":
    main()