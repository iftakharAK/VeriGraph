

import os, math, argparse
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from encoder import TextEncoder
from contradiction_scorer import PairwiseContradictionScorer
from utils import set_seed, PairDataset, collate_pairs

def train(args):
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Tokenizer only (we’ll forward through encoder inside the loop to get pooled states)
    tokenizer = AutoTokenizer.from_pretrained(args.encoder)
    # Dummy encoder to get hidden dim
    enc_model = TextEncoder(args.encoder)
    hidden_size = enc_model.model.config.hidden_size

    # Data
    train_ds = PairDataset(args.train_jsonl, tokenizer, max_length=args.max_length)
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                        collate_fn=lambda b: collate_pairs(b, tokenizer, device, args.max_length))

    # Scorer
    scorer = PairwiseContradictionScorer(hidden_size, args.mlp_hidden).to(device)
    opt = torch.optim.AdamW(list(scorer.parameters()) + list(enc_model.parameters()), lr=args.lr)
    criterion = nn.BCELoss()

    os.makedirs(args.out_dir, exist_ok=True)
    global_step = 0

    for epoch in range(1, args.epochs+1):
        scorer.train(); enc_model.train()
        total = 0.0
        for batch_inputs, labels in loader:
            pooled = enc_model.model(**{k: v for k, v in batch_inputs.items() if k in ["input_ids","attention_mask"]}).last_hidden_state[:,0,:]
            preds = scorer(pooled)
            loss = criterion(preds, labels)
            opt.zero_grad(); loss.backward(); opt.step()
            total += float(loss.item()); global_step += 1

            if args.save_steps and global_step % args.save_steps == 0:
                p = os.path.join(args.out_dir, f"step_{global_step}.pt")
                torch.save({"scorer": scorer.state_dict(), "encoder": enc_model.model.state_dict()}, p)

        avg = total / max(1, len(loader))
        print(f"Epoch {epoch} | loss={avg:.4f}")
        p = os.path.join(args.out_dir, f"epoch_{epoch}.pt")
        torch.save({"scorer": scorer.state_dict(), "encoder": enc_model.model.state_dict()}, p)

    # final
    final_p = os.path.join(args.out_dir, "final.pt")
    torch.save({"scorer": scorer.state_dict(), "encoder": enc_model.model.state_dict()}, final_p)
    print(f"Saved final to {final_p}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_jsonl", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="checkpoints")
    ap.add_argument("--encoder", type=str, default="microsoft/deberta-v3-small")
    ap.add_argument("--mlp_hidden", type=int, default=256)
    ap.add_argument("--max_length", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--save_steps", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    train(args)