#!/bin/bash
#SBATCH --job-name=batch_create
# rome and genoa cause OOM, let's go GPU that has more RAM
#SBATCH --partition=gpu_a100
# takes around 22 minutes per batch, but let's put 1 hour to be safe
#SBATCH --time=01:00:00
#SBATCH --gpus-per-node=1


# use:
# sbatch -a 0-735 parallel_batch_create.sh
# first run the following to generate the ERA5_days_pairs.json file:
# python src/dataset_creation/parallel_batch.py create-list-file

echo "SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"

source venv/bin/activate

export PYTHONUNBUFFERED=1
python src/dataset_creation/parallel_batch.py run-single $SLURM_ARRAY_TASK_ID
