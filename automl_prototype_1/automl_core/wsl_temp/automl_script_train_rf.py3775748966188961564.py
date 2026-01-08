
import argparse
import pandas as pd
import numpy as np
import time
import pickle
import os
import sys

# Configure args
parser = argparse.ArgumentParser(description='Train Random Forest')
parser.add_argument('--data', required=True, help='Path to training data csv (last col is target)')
parser.add_argument('--output', required=True, help='Path to save model pickle')
parser.add_argument('--n_estimators', type=int, default=100)
parser.add_argument('--max_depth', type=int, default=10)
args = parser.parse_args()

def train():
    try:
        print(f"Loading data from {args.data}")
        df = pd.read_csv(args.data)
        X = df.iloc[:, :-1].values.astype(np.float32)
        y = df.iloc[:, -1].values.astype(np.float32)

        # Try cuML for GPU
        try:
            import cudf
            import cuml
            from cuml.ensemble import RandomForestClassifier
            print("GPU Detected. Using cuML RandomForestClassifier.")
            
            # Convert to GPU memory
            X_gpu = cudf.DataFrame(X)
            y_gpu = cudf.Series(y)
            
            model = RandomForestClassifier(n_estimators=args.n_estimators, max_depth=args.max_depth)
            model.fit(X_gpu, y_gpu)
            
            # Save (cuML models are picklable)
            with open(args.output, 'wb') as f:
                pickle.dump(model, f)
            print("Model saved successfully via cuML.")
            
        except ImportError:
            # Fallback to Scikit-Learn (CPU but still often faster than pure Java)
            print("cuml/cudf not found. Falling back to Scikit-Learn (CPU).")
            from sklearn.ensemble import RandomForestClassifier
            
            model = RandomForestClassifier(n_estimators=args.n_estimators, max_depth=args.max_depth, n_jobs=-1)
            model.fit(X, y)
            
            with open(args.output, 'wb') as f:
                pickle.dump(model, f)
            print("Model saved successfully via Scikit-Learn.")

    except Exception as e:
        print(f"Error during training: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    train()
