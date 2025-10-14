

import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel

class TextEncoder(nn.Module):
    """
    DeBERTa-based encoder that returns [CLS]-equivalent pooled token 0 state.
    """
    def __init__(self, model_name: str = "microsoft/deberta-v3-small"):
        super().__init__()
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    @torch.no_grad()
    def encode_pairs(self, s1_list, s2_list, device="cuda", max_length=128):
        """
        Returns (B, H) pooled representations for paired inputs.
        """
        batch = self.tokenizer(
            s1_list, s2_list,
            padding=True, truncation=True, max_length=max_length, return_tensors="pt"
        ).to(device)
        out = self.model(**{k: v for k, v in batch.items() if k in ["input_ids", "attention_mask"]})
        pooled = out.last_hidden_state[:, 0, :]  # token 0 as pooled
        return pooled  # (B, H)

    def forward(self, **kwargs):
        return self.model(**kwargs)