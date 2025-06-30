#!/bin/bash
set -e

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
source $SCRIPT_DIR/env.sh

BIOCUBE_PATH=$BIOCUBE_PARENT_PATH/BioCube
CATALOG_REPORT=$BIOCUBE_PARENT_PATH/catalog_report.parquet
BATCH_DIR=$BIOCUBE_PARENT_PATH/new_monthly_batches

GIT_TOPLEVEL=$(git rev-parse --show-toplevel)
source $GIT_TOPLEVEL/venv/bin/activate

# scan (build catalog_report.parquet)
python $SCRIPT_DIR/scan_biocube.py \
    --root $BIOCUBE_PATH \
    --out $CATALOG_REPORT

# build batches
python $SCRIPT_DIR/build_batches_monthly.py \
    --report $CATALOG_REPORT \
    --batch_dir $BATCH_DIR