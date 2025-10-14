#!/bin/bash
# ============================================================
#  VeriGraph – Run Fine-Tuning Training
# ============================================================

echo " Starting VeriGraph training..."

CONFIG=configs/config_train.yaml
python src/train.py --config $CONFIG

echo " Training complete. Checkpoints saved under /checkpoints"
