#!/bin/bash
#SBATCH --job-name=batch_create
# takes around 4 hours for 26k files
#SBATCH --time=7:00:00
#SBATCH --ntasks=1

# CPU option
#SBATCH --partition=rome
#SBATCH --cpus-per-task=16


source venv/bin/activate

export PYTHONUNBUFFERED=1
python src/dataset_creation/final_stats.py --batches-dir=/projects/prjs1134/data/projects/biodt/storage/batches_2025_03_05
