

"""
Builds contradiction graphs from multi-statement datasets.
Each graph connects contradictory pairs (edges) between statements (nodes).
"""

import json
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import itertools


class MiniVeriGraph(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = AutoModel.from_pretrained("microsoft/deberta-v3-small")
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(self.encoder.config.hidden_size, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 1)
        )

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = out.last_hidden_state[:, 0, :]
        return torch.sigmoid(self.classifier(pooled)).squeeze(-1)


def build_graph(model_path, dataset_path, output_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MiniVeriGraph().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-small")
    graphs = []

    with open(dataset_path, "r") as f:
        for line in tqdm(f, desc="Building graphs"):
            sample = json.loads(line)
            sents = sample["statements"]
            edges = []

            for (i, j) in itertools.permutations(range(len(sents)), 2):
                enc = tokenizer(sents[i], sents[j], return_tensors="pt",
                                truncation=True, padding=True).to(device)
                with torch.no_grad():
                    score = model(**enc).item()
                label = "contradiction" if score > 0.5 else "non-contradiction"
                edges.append({"source": i, "target": j, "score": round(score, 3), "type": label})

            graphs.append({
                "id": sample["id"],
                "statements": sents,
                "edges": edges
            })

    with open(output_path, "w") as f:
        for g in graphs:
            f.write(json.dumps(g) + "\n")

    print(f"Built graphs saved to {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to contradiction model .pt file")
    parser.add_argument("--dataset", required=True, help="Input dataset JSONL")
    parser.add_argument("--output", required=True, help="Output JSONL for graph dataset")
    args = parser.parse_args()

    build_graph(args.model, args.dataset, args.output)