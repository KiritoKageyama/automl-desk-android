#!/bin/bash
# install_rapids_wsl.sh
# -------------------------------------------------------------
# This script sets up a Conda environment in WSL2 with RAPIDS (cuML)
# and XGBoost with GPU support.
#
# Prerequisities:
# 1. WSL2 installed (wsl --install)
# 2. NVIDIA Drivers on Windows (no need to install drivers in WSL)
# -------------------------------------------------------------

set -e

echo ">>> Checking for Bash (Required for Miniconda)..."
if ! command -v bash &> /dev/null; then
    echo "Bash not found. Attempting to install..."
    if command -v apk &> /dev/null; then apk add --no-cache bash; 
    elif command -v apt-get &> /dev/null; then apt-get update && apt-get install -y bash;
    else echo "Could not install bash. Please install bash manually."; exit 1; fi
fi

echo ">>> Checking for Conda..."
if ! command -v conda &> /dev/null; then
    echo "Conda not found. Installing Miniconda..."
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    bash miniconda.sh -b -p $HOME/miniconda
    source $HOME/miniconda/bin/activate
    conda init
    echo "Miniconda installed. Please restart your terminal or source bashrc if this script exits."
else
    echo "Conda detected."
fi

# Create Environment
ENV_NAME="automl_gpu_env"
echo ">>> Creating Conda Environment: $ENV_NAME"
echo "This may take a while as it downloads large GPU libraries..."

# Install RAPIDS (cuml), XGBoost (gpu), and Scikit-learn
# Using rapidsai channel as per official docs
conda create -n $ENV_NAME -c rapidsai -c conda-forge -c nvidia \
    cuml=23.10 \
    python=3.10 \
    cuda-version=11.8 \
    xgboost \
    scikit-learn \
    -y

echo ">>> Environment Created Successfully!"
echo "To use this environment manually:"
echo "    conda activate $ENV_NAME"
echo ""
echo "The AutoML App will autodect this env if configured to use 'wsl python'."
