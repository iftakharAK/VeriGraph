#!/bin/bash
# ============================================================
# VeriGraph – Evaluate Model on Multi-Statement Data
# ============================================================

echo " Running VeriGraph evaluation..."

CONFIG=configs/config_eval.yaml
python src/evaluate.py --config $CONFIG

echo " Evaluation complete. Results saved under data/test_set.jsonl"
