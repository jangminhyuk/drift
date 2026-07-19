#!/bin/bash
# Setup script for UIUC Delta HPC (no sudo required)
# Run this on the login node: bash setup_delta.bash

set -e
RED='\033[0;31m' BLUE='\033[0;34m' GREEN='\033[0;32m' NC='\033[0m'

ENV_PREFIX="$CONDA_ENV"
DATASET_DIR="$STORE/dataset"

echo -e "${BLUE}=== Step 1: Create conda environment ===${NC}"
if [ -d "$ENV_PREFIX" ]; then
    echo "Conda env already exists at $ENV_PREFIX, skipping creation."
else
    conda create --prefix "$ENV_PREFIX" python=3.10 -y
fi

echo -e "${BLUE}=== Step 2: Install Python packages ===${NC}"
conda install --prefix "$ENV_PREFIX" -c conda-forge \
    numpy scipy matplotlib pandas pyyaml scikit-learn p7zip -y

# pyquaternion is pip-only
"$ENV_PREFIX/bin/pip" install pyquaternion

echo -e "${BLUE}=== Step 3: Download and extract UTIL UWB dataset ===${NC}"
if [ -d "$DATASET_DIR/flight-dataset" ]; then
    echo "Dataset already exists at $DATASET_DIR, skipping download."
else
    mkdir -p "$DATASET_DIR"
    cd "$DATASET_DIR"
    echo -e "${BLUE}Downloading dataset (~1.7GB, grab a coffee)...${NC}"
    curl -L https://github.com/utiasDSL/util-uwb-dataset/releases/download/dataset-v1.0/dataset.7z -o dataset.7z
    "$ENV_PREFIX/bin/7z" x ./dataset.7z
    rm dataset.7z
    echo "Dataset extracted to $DATASET_DIR"
fi

echo ""
echo -e "${GREEN}=== Setup complete ===${NC}"
echo ""
echo "To activate the environment:"
echo "  conda activate $ENV_PREFIX"
echo ""
echo "Dataset location: $DATASET_DIR"
echo "You may want to symlink it into the repo:"
echo "  ln -s $DATASET_DIR/flight-dataset $REPO/flight-dataset"
echo "  ln -s $DATASET_DIR/identification-dataset $REPO/identification-dataset"
