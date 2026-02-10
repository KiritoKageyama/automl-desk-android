#!/bin/bash
# Maintenance Script to Standardize Environment
# This aligns all libraries to the 'conda-forge' channel (Gold Standard)
# and updates System libraries.

# Stop on error
set -e

source /home/kirito/miniconda/bin/activate automl_gpu_env

echo "========================================================"
echo "      ENVIRONMENT STANDARDIZATION & CLEANUP"
echo "========================================================"

# 1. SYSTEM LIBRARIES (WSL via APT)
# This fixes the glibc/binutils mismatch at the OS level
echo ""
echo "[1/3] Updating WSL System Libraries..."
echo "      (You may be asked for your sudo password)"
sudo apt-get update && sudo apt-get dist-upgrade -y
sudo apt-get install -y build-essential

# 2. CONDA CONFIGURATION
# Set strict priority to avoid mixing 'defaults' (incompatible) with 'conda-forge'
echo ""
echo "[2/3] Configuring Conda Priority..."
conda config --env --add channels conda-forge
conda config --env --set channel_priority strict
echo "      Channel priority set to 'Strict' (conda-forge)."

# 3. ENVIRONMENT ALIGNMENT
# This effectively 're-solves' the environment to ensure ABI compatibility
echo ""
echo "[3/3] Aligning Project Libraries..."
echo "      Upgrading all packages to latest stable versions from conda-forge..."
conda update --all -y

# 4. PYTHON CACHE CLEANUP
echo ""
echo "[4/4] Cleaning Caches..."
conda clean -a -y
pip cache purge

echo "========================================================"
echo "      MAINTENANCE COMPLETE"
echo "========================================================"
echo "All libraries are now aligned to 'conda-forge' and updated."
