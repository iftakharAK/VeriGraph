# 🧠 VeriGraph: Multi-Statement Contradiction Detection with Verified Explanations

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.2%2B-red.svg)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/Transformers-4.39%2B-yellow.svg)](https://huggingface.co/transformers/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1dM8rE5qXXXXXXX?usp=sharing)

---

### 🧩 Overview

**VeriGraph** is a scalable framework for **multi-statement contradiction detection** and **explanation generation**.  
It integrates DeBERTa-based encoding, graph-based reasoning, and T5/DeepSeek explanation generation with self-reflection verification.

---

## ⚙️ Installation

### 🧩 Option 1 — Using Conda
```bash
conda env create -f environment.yml
conda activate verigraph_env
```

### 🧩 Option 2 — Using venv + pip
```bash
python -m venv verigraph_env
source verigraph_env/bin/activate     # (Linux/Mac)
verigraph_env\Scripts\activate        # (Windows)
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🚀 Running Demos

**1️⃣ Contradiction Detection**
```bash
python notebooks/VeriGraph_demo.ipynb
```
**Example Output:**
```
"Water boils at 100°C." ↔ "Water does not boil at 100°C." → CONTRADICTION (0.94)
```

**2️⃣ Explanation Generation**
```bash
python notebooks/Explanation_examples.ipynb
```
**Example Output:**
```
Input: "Water boils at 100°C." vs "Water does not boil at 100°C."
Output: "These statements contradict because one denies the other."
```

---

## 🧠 Model Components

| Module | Function |
|---------|-----------|
| `encoder.py` | DeBERTa text encoder |
| `contradiction_scorer.py` | Binary classifier for contradiction |
| `graph_builder.py` | Multi-statement graph formation |
| `gnn_reasoner.py` | GAT-based graph reasoning |
| `explanation_generator.py` | FLAN-T5 / DeepSeek explanation model |
| `reflection_verifier.py` | Self-verification of generated explanations |

---

## 📊 Citation

```
@article{Khandokar2025VeriGraph,
  author    = {Iftakhar Ali Khandokar and Priya Deshpande},
  title     = {VeriGraph: Scalable Multi-Statement Contradiction Detection with Verified Explanations and Knowledge-Guided Reasoning},
  journal   = {Under review, IEEE TNNLS / Knowledge-Based Systems},
  year      = {2025}
}
```

---

## 📜 License
Released under the [MIT License](./LICENSE).

---

**Maintainer:** Iftakhar Ali Khandokar  
Department of Electrical and Computer Engineering, Marquette University  
📧 ifty.khandokar@marquette.edu  
🌐 [GitHub @iftakharAK](https://github.com/iftakharAK)
