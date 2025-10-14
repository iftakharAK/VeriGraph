

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class ExplanationGenerator:
    def __init__(self, model_path: str = "google/flan-t5-base", max_new_tokens: int = 80):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        self.model.to("cuda" if torch.cuda.is_available() else "cpu")
        self.model.eval()
        self.max_new_tokens = max_new_tokens

    @torch.no_grad()
    def generate(self, s1: str, s2: str, knowledge: str = None):
        prompt = f"Describe how these two statements conflict: '{s1}' vs '{s2}'."
        if knowledge:
            prompt = f"[KNOWLEDGE]\n{knowledge}\n[QUERY]\n{prompt}"
        enc = self.tokenizer(prompt, return_tensors="pt", truncation=True, padding=True).to(self.model.device)
        out = self.model.generate(**enc, max_new_tokens=self.max_new_tokens)
        return self.tokenizer.decode(out[0], skip_special_tokens=True)