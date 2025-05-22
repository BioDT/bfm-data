#!/bin/bash
set -e

if [[ $HOSTNAME =~ "snellius" ]]; then
    module purge
    module load 2024 Python/3.12.3-GCCcore-13.3.0
fi

# default venv name is venv
VENV_PATH="${1:-venv}"

if test -d $VENV_PATH; then
    echo "venv: $VENV_PATH already exists, using it"
else
    # create venv
    echo "creating venv at: $VENV_PATH"
    python3 -m venv $VENV_PATH
fi

source $VENV_PATH/bin/activate

pip install poetry
poetry config virtualenvs.in-project true

poetry install