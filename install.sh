#!/bin/bash
set -e
module purge
module load 2023 CUDA/12.4.0

# module load poetry/1.5.1-GCCcore-12.3.0
module load Python/3.11.3-GCCcore-12.3.0
# module load Python-bundle-PyPI/2023.06-GCCcore-12.3.0
# module load SciPy-bundle/2023.07-gfbf-2023a
# module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1

# default venv name is venv
venv_path="${1:-venv}"

python3 -m venv $venv_path

source $venv_path/bin/activate

## via pip
# pip install -U pip setuptools wheel
# pip install -e .

# via poetry (more precise versions)
pip install poetry
# global poetry may be broken!!!
$venv_path/bin/poetry install
