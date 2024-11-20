#!/bin/bash

# use:
# sbatch -a 0-735 generate_batches.sh
# first run the following:
# python src/dataset_creation/parallel_batch.py create-list-file

echo "SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"


python src/dataset_creation/parallel_batch.py run-single $SLURM_ARRAY_TASK_ID
