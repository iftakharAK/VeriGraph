# 📘 VeriGraph Datasets

This repository provides sample and synthetic datasets used for evaluating the **VeriGraph** framework — a scalable system for **multi-statement contradiction detection with verified explanations and knowledge-guided reasoning**.

---

## 🧩 Dataset Overview

VeriGraph experiments use **three key datasets**:

| File | Description | Purpose |
|------|--------------|----------|
| `dataset.jsonl` | Multi-statement contradiction dataset derived from NLI + fact-checking sources (MultiNLI, eSNLI, FEVER, SciFact, etc.) | Training the DeBERTa encoder + MLP contradiction scorer |
| `dataset_explanations.jsonl` | Explanation samples aligned with contradiction pairs (T5-formatted) | Fine-tuning the FLAN-T5 / DeepSeek explanation generator |
| `verigraph_multistatement_input.jsonl` | Unlabeled, multi-statement dataset for inference | End-to-end VeriGraph pipeline evaluation (contradiction graph + explanation verification) |

---

## 🧠 Data Schema

### 1️⃣ Multi-Statement Contradiction Dataset (`dataset.jsonl`)
Each record contains multiple related statements (typically 3–4) and optionally a pairwise label matrix for training the contradiction scorer.

```json
{
  "id": 42,
  "statements": [
    "Water boils at 100°C at sea level.",
    "Water does not boil at 100°C.",
    "At higher altitudes, water boils at lower temperatures.",
    "Boiling point of water depends on atmospheric pressure."
  ],
  "labels": [
    [0, 1, 0, 0],
    [1, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0]
  ]
}
