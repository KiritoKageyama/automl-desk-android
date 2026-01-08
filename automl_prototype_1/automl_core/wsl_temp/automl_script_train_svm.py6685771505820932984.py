
import argparse
import pandas as pd
import numpy as np
import pickle
import sys

parser = argparse.ArgumentParser(description='Train SVM')
parser.add_argument('--data', required=True, help='Path to training data csv')
parser.add_argument('--output', required=True, help='Path to save model pickle')
parser.add_argument('--C', type=float, default=1.0, help="Regularization parameter")
parser.add_argument('--kernel', type=str, default='rbf', help="Kernel type (rbf, linear, poly, sigmoid)")
args = parser.parse_args()

def train():
    try:
        print(f"Loading data from {args.data}")
        df = pd.read_csv(args.data)
        X = df.iloc[:, :-1].values.astype(np.float32)
        y = df.iloc[:, -1].values.astype(np.float32)

        # Try cuML
        try:
            import cudf
            import cuml
            from cuml.svm import SVC
            print(f"GPU Detected. Using cuML SVC (Kernel: {args.kernel}).")

            if args.kernel == 'linear': # cuML linear SVC is specialized
                 # For general SVC
                 pass
            
            X_gpu = cudf.DataFrame(X)
            y_gpu = cudf.Series(y)
            
            model = SVC(C=args.C, kernel=args.kernel, probability=True) # Prob=True for consistency if needed, though slower
            model.fit(X_gpu, y_gpu)
            
            with open(args.output, 'wb') as f:
                pickle.dump(model, f)
            print("Model saved via cuML.")
            
        except ImportError:
            print("cuML not found. Falling back to Scikit-Learn SVM (CPU). WARNING: Slow for large datasets.")
            from sklearn.svm import SVC
            model = SVC(C=args.C, kernel=args.kernel, probability=True)
            model.fit(X, y)
            
            with open(args.output, 'wb') as f:
                pickle.dump(model, f)
            print("Model saved via Scikit-Learn.")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    train()
