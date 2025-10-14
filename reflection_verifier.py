

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class ReflectionVerifier:
    """
    Checks whether the generated explanation is entailed by (s1 + s2 + knowledge).
    Uses MNLI-style NLI model; returns boolean and entailment probability.
    """
    def __init__(self, model_name: str = "microsoft/deberta-v3-base", tau: float = 0.7):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to("cuda" if torch.cuda.is_available() else "cpu")
        self.model.eval()
        self.tau = tau

    @torch.no_grad()
    def verify(self, s1: str, s2: str, explanation: str, knowledge: str = ""):
        premise = f"{s1}\n{s2}\n{knowledge}".strip()
        hyp = explanation
        batch = self.tokenizer(premise, hyp, return_tensors="pt", truncation=True, padding=True).to(self.model.device)
        logits = self.model(**batch).logits[0]
        prob = torch.softmax(logits, dim=-1)
        entail = float(prob[-1].item())
        return entail >= self.tau, entail