import torch
import torch.nn as nn
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Model, GPT2Config

INTERIM = Path("data/interim/rl")
MODEL_DIR = Path("models/reward_model")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MAX_LEN = 30
BATCH_SIZE = 64
EPOCHS = 1 # 1 эпоха для избежания переобучения

class RewardDataset(Dataset):
    def __init__(self, path):
        self.data = pd.read_parquet(path)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, i):
        row = self.data.iloc[i]
        seq = row["sequence"][:-1][-MAX_LEN:]
        if len(seq) == 0:
            seq = [0]
        reward = row["rewards"][-1] / 100.0
        return torch.tensor(seq), torch.tensor(reward, dtype=torch.float32)

def collate_fn(batch):
    seqs, rewards = zip(*batch)
    seqs = torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=0)
    return seqs, torch.stack(rewards)

class RewardModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        config = GPT2Config(
            vocab_size=vocab_size, 
            n_positions=MAX_LEN, 
            n_embd=128, 
            n_layer=2, 
            n_head=4,
            resid_pdrop=0.2,
            embd_pdrop=0.2,
            attn_pdrop=0.2,
        )
        self.gpt = GPT2Model(config)
        self.head = nn.Linear(128, 1)
    def forward(self, input_ids):
        out = self.gpt(input_ids).last_hidden_state[:, -1, :]
        return torch.sigmoid(self.head(out)).squeeze(-1)

def main():
    print("Loading data for Reward Model...")
    train_ds = RewardDataset(INTERIM / "seq_train.parquet")
    valid_ds = RewardDataset(INTERIM / "seq_valid.parquet")
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    
    # Берем vocab_size НАПРЯМУЮ из обученной GPT-2, чтобы они точно совпали
    gpt_config = GPT2Config.from_pretrained("models/gpt2_rec")
    vocab_size = gpt_config.vocab_size
    print(f"Using vocab_size from GPT-2: {vocab_size}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RewardModel(vocab_size).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.MSELoss()

    print(f"Training Reward Model on {device}...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for seqs, rewards in train_loader:
            seqs, rewards = seqs.to(device), rewards.to(device)
            opt.zero_grad()
            preds = model(seqs)
            loss = loss_fn(preds, rewards)
            loss.backward()
            opt.step()
            total_loss += loss.item()
            
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for seqs, rewards in valid_loader:
                seqs, rewards = seqs.to(device), rewards.to(device)
                val_loss += loss_fn(model(seqs), rewards).item()
        val_loss /= len(valid_loader)
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Train MSE: {total_loss/len(train_loader):.4f} | Valid MSE: {val_loss:.4f}")

    torch.save(model.state_dict(), MODEL_DIR / "rm.pt")
    print(f"Reward model saved to {MODEL_DIR}")

if __name__ == "__main__":
    main()