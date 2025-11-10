#!/bin/bash
#SBATCH --job-name=gatshift_train
#SBATCH --output=logs/gatshift_%j.out
#SBATCH --error=logs/gatshift_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1           # Request one GPU (if available)
#SBATCH --mem=32G
#SBATCH --time=24:00:00        # 24 hours
#SBATCH --partition=gpu        # Or whatever your cluster calls it

# Load environment
module load anaconda
source activate gatshift

# Go to working directory
cd $SLURM_SUBMIT_DIR

# Run training
python gatshift_train.py \
    --data ../data/final_dataset/graph/graphs_sequential.pt \
    --epochs 100 \
    --batch_size 2 \
    --lr 2e-3 \
    --save_every 10
