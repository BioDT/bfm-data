if [[ $HOSTNAME =~ "snellius" ]]; then
    export BIOCUBE_PARENT_PATH=/projects/prjs1134/data/projects/biodt/storage # snellius
else
    export BIOCUBE_PARENT_PATH=data # local folder
fi
