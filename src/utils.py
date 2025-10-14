
import json, random, os
import torch
from torch.utils.data import Dataset, DataLoader

def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class PairDataset(Dataset):
    """
    Consumes multi-statement samples with adjacency labels.
    JSONL format:
    {"statements": [...], "labels": [[...],[...],...]}  # labels[i][j] ∈ {0,1}
    """
    def __init__(self, jsonl_path, tokenizer, max_length=128):
        self.samples = []
        with open(jsonl_path, "r") as f:
            for line in f:
                obj = json.loads(line)
                sents = obj.get("statements", [])
                labels = obj.get("labels", None)
                if labels is None:  # unlabeled -> skip for training
                    continue
                n = len(sents)
                for i in range(n):
                    for j in range(n):
                        if i == j: continue
                        self.samples.append((sents[i], sents[j], labels[i][j]))
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        s1, s2, y = self.samples[idx]
        return s1, s2, int(y)

def collate_pairs(batch, tokenizer, device="cuda", max_length=128):
    s1_list, s2_list, y = zip(*batch)
    enc = tokenizer(list(s1_list), list(s2_list),
                    padding=True, truncation=True, max_length=max_length,
                    return_tensors="pt")
    return {k: v.to(device) for k, v in enc.items()}, torch.tensor(y, dtype=torch.float32).to(device)