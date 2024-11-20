#!/bin/bash
#SBATCH --job-name=batch_create
# rome and genoa cause OOM, let's go GPU that has more RAM
#SBATCH --partition=gpu_a100
#SBATCH --time=01:00:00
# SBATCH --nodes=1
# SBATCH --ntasks-per-node=1
# SBATCH --cpus-per-task=18
#SBATCH --gpus-per-node=1


# use:
# sbatch -a 0-735 parallel_batch_create.sh
# first run the following:
# python src/dataset_creation/parallel_batch.py create-list-file

echo "SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"

source venv/bin/activate

export PYTHONUNBUFFERED=1
python src/dataset_creation/parallel_batch.py run-single $SLURM_ARRAY_TASK_ID
