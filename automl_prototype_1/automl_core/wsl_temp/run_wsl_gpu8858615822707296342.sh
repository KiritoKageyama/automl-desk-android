#!/bin/bash
PYTHON_EXEC="$HOME/miniconda/envs/automl_gpu_env/bin/python"
if [ ! -f "$PYTHON_EXEC" ]; then
    PYTHON_EXEC="$HOME/miniconda3/envs/automl_gpu_env/bin/python"
fi

if [ -f "$PYTHON_EXEC" ]; then
    "$PYTHON_EXEC" "$@"
else
    source "$HOME/miniconda/etc/profile.d/conda.sh" 2>/dev/null || source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null
    conda activate automl_gpu_env
    python "$@"
fi
