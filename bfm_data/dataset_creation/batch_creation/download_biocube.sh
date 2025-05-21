#!/bin/bash
set -e

mkdir -p data

# git lfs install # THIS IS FOR snellius or non-sudo
# https://gist.github.com/pourmand1376/bc48a407f781d6decae316a5cfa7d8ab
# test if installed
if ! [ -x "$(command -v git-lfs)" ]; then
    echo 'INFO: git-lfs is not installed. Installing to local folder'
    mkdir -p git-lfs
    pushd git-lfs
    LFS_VERSION=3.6.1
    wget https://github.com/git-lfs/git-lfs/releases/download/v$LFS_VERSION/git-lfs-linux-amd64-v$LFS_VERSION.tar.gz
    tar xvf git-lfs-linux-amd64-v$LFS_VERSION.tar.gz
    cd git-lfs-$LFS_VERSION/
    chmod +x install.sh
    sed -i 's|^prefix="/usr/local"$|prefix="$HOME/.local"|' install.sh
    mkdir -p ~/.local/bin/
    export PATH="$HOME/.local/bin:$PATH"
    ./install.sh
    git-lfs --version
    popd
fi

BIOCUBE_PATH=data/BioCube
if test -d $BIOCUBE_PATH; then
    echo "$BIOCUBE_PATH already exists, using it"
else
    # create venv
    echo "Downloading BioCube at $BIOCUBE_PATH"
    git clone https://huggingface.co/datasets/BioDT/BioCube data/BioCube
fi
