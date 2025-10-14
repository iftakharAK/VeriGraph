

from typing import List, Dict, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class KnowledgeConsistencyScorer:
    """
    Lightweight entailment-based filter using an MNLI model.
    Returns passages whose entailment prob wrt query exceeds tau.
    """
    def __init__(self, model_name: str = "microsoft/deberta-v3-base"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    @torch.no_grad()
    def entailment_score(self, premise: str, hypothesis: str) -> float:
        batch = self.tokenizer(premise, hypothesis, return_tensors="pt",
                               truncation=True, padding=True).to(self.device)
        logits = self.model(**batch).logits[0]  # (3) MNLI: contradiction/neutral/entailment
        # assume label mapping [contradiction, neutral, entailment]
        prob = torch.softmax(logits, dim=-1)
        return float(prob[-1].item())

    def filter_passages(self, query: str, passages: List[str], tau: float = 0.7, k: int = 5):
        scored = [(p, self.entailment_score(p, query)) for p in passages]
        kept = [p for p, s in sorted(scored, key=lambda x: x[1], reverse=True) if s >= tau]
        return kept[:k]

class FiDCombiner:
    """
    Simple Fusion-in-Decoder style: concatenate top-k passages with query for generator.
    """
    def build_fid_input(self, query: str, passages: List[str]) -> str:
        if not passages:
            return query
        ctx = "\n".join([f"[KNOWLEDGE] {p}" for p in passages])
        return f"{ctx}\n[QUERY] {query}"