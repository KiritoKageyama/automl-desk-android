#!/bin/bash
set -e

echo "=== AutoML GPU Environment Setup ==="
echo "Checking for Conda..."

# Try to find conda
if command -v conda &> /dev/null; then
    echo "Conda found at $(command -v conda)"
    CONDA_EXE=$(command -v conda)
elif [ -f "$HOME/miniconda3/bin/conda" ]; then
    echo "Conda found at $HOME/miniconda3/bin/conda"
    eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
    CONDA_EXE="$HOME/miniconda3/bin/conda"
elif [ -f "$HOME/anaconda3/bin/conda" ]; then
    echo "Conda found at $HOME/anaconda3/bin/conda"
    eval "$($HOME/anaconda3/bin/conda shell.bash hook)"
    CONDA_EXE="$HOME/anaconda3/bin/conda"
else
    echo "Conda not found. Installing Miniconda automatically..."
    
    # Check for wget
    if ! command -v wget &> /dev/null; then
        echo "Installing wget..."
        sudo apt-get update && sudo apt-get install -y wget
    fi

    # Download and Install Miniconda
    echo "Downloading Miniconda3..."
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
    
    echo "Installing Miniconda3 to $HOME/miniconda3..."
    bash ~/miniconda.sh -b -p $HOME/miniconda3
    rm ~/miniconda.sh
    
    echo "Initializing Conda..."
    eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
    $HOME/miniconda3/bin/conda init bash
    
    CONDA_EXE=$HOME/miniconda3/bin/conda
fi

echo "Checking for 'automl_gpu_env'..."
if conda info --envs | grep -q "automl_gpu_env"; then
    echo "Environment 'automl_gpu_env' already exists. Skipping creation."
    echo "Trying to update/verify..."
    # Optionally update. For now, just assume it's good or let user force clean if needed.
else
    echo "Creating 'automl_gpu_env'..."
    # Create environment
    conda create -n automl_gpu_env -c rapidsai -c conda-forge -c nvidia \
        cuml=24.02 python=3.10 \
        scikit-learn pandas xgboost \
        -y
fi

echo ""
echo "=== Setup Complete ==="
echo "To activate manually: conda activate automl_gpu_env"
