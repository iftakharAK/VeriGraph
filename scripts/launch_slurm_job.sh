#!/bin/bash
# ============================================================
# VeriGraph – SLURM Batch Job Launcher (HPC Compatible)
# ============================================================

#SBATCH --job-name=verigraph_train
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=24G

echo "SLURM job started on node: $(hostname)"
echo "Environment:"
nvidia-smi

module load python/3.10 cuda/12.1
source ~/envs/verigraph/bin/activate

cd ~/VeriGraph
bash scripts/run_training.sh

echo "SLURM job completed successfully."
