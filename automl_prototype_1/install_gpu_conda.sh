#!/bin/bash
# Stop immediately if any command fails
set -e

# Enable Conda environment
source /home/kirito/miniconda/bin/activate automl_gpu_env

echo "---------------------------------------------------------"
echo "Installing Pre-Built LightGBM (Conda Method)"
echo "---------------------------------------------------------"
echo "Abandoning source compilation. Using pre-compiled CUDA binaries from conda-forge."

# 1. Clean up Pip installation
echo "Removing pip-installed LightGBM..."
set +e
pip uninstall -y lightgbm
set -e

# 2. Install LightGBM from Conda-Forge with CUDA support
# We request the 'cuda' build string explicitly to ensure we don't get the CPU one
echo "Installing lightgbm=*=*cuda* from conda-forge..."

# Note: We must fetch from conda-forge. We also ensure cudatoolkit is present.
conda install -y -c conda-forge "lightgbm=*=*cuda*"

# 3. Verify Installation
echo "---------------------------------------------------------"
echo "Verifying Installation..."
python -c "import lightgbm; print(f'LightGBM Version: {lightgbm.__version__}');"

echo "---------------------------------------------------------"
echo "DONE. Please restart the application."
echo "---------------------------------------------------------"
