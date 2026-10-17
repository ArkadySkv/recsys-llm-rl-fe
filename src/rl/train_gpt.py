import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Config, GPT2LMHeadModel
import pandas as pd
from pathlib import Path

INTERIM = Path("data/interim/rl")
MODEL_DIR = Path("models/gpt2_rec")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MAX_LEN = 30
BATCH_SIZE = 16 # Уменьшили, чтобы влезло в память
EPOCHS = 5

class SeqDataset(Dataset):
    def __init__(self, path):
        self.seqs = pd.read_parquet(path)["sequence"].tolist()
    def __len__(self):
        return len(self.seqs)
    def __getitem__(self, i):
        seq = self.seqs[i][-MAX_LEN:]
        return torch.tensor(seq[:-1]), torch.tensor(seq[1:])

def collate_fn(batch):
    inputs, targets = zip(*batch)
    inputs = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=0)
    targets = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=-100)
    return inputs, targets

def main():
    train_ds = SeqDataset(INTERIM / "seq_train.parquet")
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    
    max_item = max([max(s) for s in train_ds.seqs])
    vocab_size = int(max_item + 1)
    print(f"Vocab size: {vocab_size}")

    config = GPT2Config(vocab_size=vocab_size, n_positions=MAX_LEN, n_embd=128, n_layer=4, n_head=4)
    model = GPT2LMHeadModel(config)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs, labels=targets)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/len(train_loader):.4f}")

    model.save_pretrained(MODEL_DIR)
    print(f"Model saved to {MODEL_DIR}")

if __name__ == "__main__":
    main()