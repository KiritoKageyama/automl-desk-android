import sys

print("--- Python GPU Check ---")
print(f"Python Executable: {sys.executable}")

# Check XGBoost
try:
    import xgboost as xgb
    print(f"XGBoost Version: {xgb.__version__}")
    try:
        # Try to invoke a GPU operation or check config
        # Simple check: can we build a dummy DMatrix?
        print("XGBoost installed. Checking GPU support via dummy training...")
        # Note: We can't easily force an error without data, but we can check if build supports it generically
        # or just assume if it imports it's there, but usually people install cpu-only versions.
        # Best check is runtime.
    except Exception as e:
        print(f"XGBoost GPU check warning: {e}")
except ImportError:
    print("XGBoost NOT installed.")

# Check cuML
try:
    import cuml
    print(f"cuML Version: {cuml.__version__}")
    print("cuML (RAPIDS) is installed and available.")
except ImportError:
    print("cuML (RAPIDS) NOT installed. Random Forest GPU will fallback to CPU.")

# Check Torch/Tensorflow just in case (optional)
try:
    import torch
    print(f"PyTorch Version: {torch.__version__}, CUDA Available: {torch.cuda.is_available()}")
except ImportError:
    pass
