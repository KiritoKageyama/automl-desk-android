#!/bin/bash
# Wrapper script to execute python in the automl_gpu_env
# Used by PythonExecutionService.java

# Direct path to the python executable in the conda env
# This bypasses the slow 'conda activate' and shell init
PYTHON_EXEC="$HOME/miniconda/envs/automl_gpu_env/bin/python"

# Fallback check
if [ ! -f "$PYTHON_EXEC" ]; then
    # Try alternate location (sometimes miniconda3)
    PYTHON_EXEC="$HOME/miniconda3/envs/automl_gpu_env/bin/python"
fi

if [ -f "$PYTHON_EXEC" ]; then
    "$PYTHON_EXEC" "$@"
else
    echo "Error: Python executable not found at $PYTHON_EXEC" >&2
    # Last ditch fallback to slow method if direct path fails
    source "$HOME/miniconda/etc/profile.d/conda.sh" 2>/dev/null || source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null
    conda activate automl_gpu_env
    python "$@"
fi
