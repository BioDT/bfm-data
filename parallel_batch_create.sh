#!/bin/bash
#SBATCH --job-name=batch_create
# takes around 5 minutes for each day of input data. So with CHUNK_SIZE=5 (5 days) it should take around 25 minutes
#SBATCH --time=1:00:00
#SBATCH --ntasks=1

# CPU option (works with reduced batch size, the cheapest)
#SBATCH --partition=rome
#SBATCH --cpus-per-task=16


# GPU option
# SBATCH --partition=gpu_h100
# SBATCH --gpus-per-node=1
# #SBATCH --mem=180GB # better to use mem-per-cpu: # SLURM_MEM_PER_CPU, SLURM_MEM_PER_GPU, and SLURM_MEM_PER_NODE are mutually exclusive.
# SBATCH --mem-per-cpu=11520MB
# SBATCH --cpus-per-task=16

# Himem option
# SBATCH --partition=himem_4tb
# SBATCH --mem=480G
# SBATCH --mem-per-cpu=30GB
# SBATCH --cpus-per-task=16

# Himem option
# SBATCH --partition=himem_8tb
# #SBATCH --mem=960G # better to use mem-per-cpu: # SLURM_MEM_PER_CPU, SLURM_MEM_PER_GPU, and SLURM_MEM_PER_NODE are mutually exclusive.
# SBATCH --mem-per-cpu=60GB
# SBATCH --cpus-per-task=16


CHUNK_SIZE=5 # this means 5 days of data in 1 job (estimated ~5 minutes for each day)

# use:
# sbatch -a 0-1632 parallel_batch_create.sh
# first run the following to get the maximum index
# python src/dataset_creation/parallel_batch.py get-max-index --chunk-size=$CHUNK_SIZE

echo "SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"

source venv/bin/activate

export PYTHONUNBUFFERED=1
# srun python src/dataset_creation/parallel_batch.py run-single $SLURM_ARRAY_TASK_ID --chunk-size=$CHUNK_SIZE
python src/dataset_creation/parallel_batch.py run-single $SLURM_ARRAY_TASK_ID --chunk-size=$CHUNK_SIZE
# memory profiling
# srun python -m memray run src/dataset_creation/parallel_batch.py run-single $SLURM_ARRAY_TASK_ID --chunk-size=$CHUNK_SIZE
# process profiling
# srun python -m cProfile -o stats-$SLURM_JOB_ID.prof src/dataset_creation/parallel_batch.py run-single $SLURM_ARRAY_TASK_ID --chunk-size=$CHUNK_SIZE

# then to visualize profiling
# memray -vvv flamegraph --temporal src/dataset_creation/memray-parallel_batch.py.1362661.bin
# snakeviz stats-$SLURM_JOB_ID.prof
