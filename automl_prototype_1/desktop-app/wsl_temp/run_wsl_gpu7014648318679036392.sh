#!/bin/bash
# --- Phase 6: Resource Cap ---
ulimit -v 8388608
export OMP_NUM_THREADS=4
echo "[Wrapper] Enforced Virtual Memory Limit: 8192MB, Threads: 4"
# -----------------------------
PYTHON_EXEC="$HOME/miniconda/envs/automl_gpu_env/bin/python"
if [ ! -f "$PYTHON_EXEC" ]; then
    PYTHON_EXEC="$HOME/miniconda3/envs/automl_gpu_env/bin/python"
fi

if [ -f "$PYTHON_EXEC" ]; then
    echo "[Wrapper] Direct python found at $PYTHON_EXEC"
    "$PYTHON_EXEC" -u "$@"
else
    echo "[Wrapper] Direct python NOT found. Using Conda fallback..."
    source "$HOME/miniconda/etc/profile.d/conda.sh" 2>/dev/null || source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null
    echo "[Wrapper] Conda sourced. Activating env..."
    conda activate automl_gpu_env
    echo "[Wrapper] Env activated. Running python..."
    python -u "$@"
fi
exit $?
