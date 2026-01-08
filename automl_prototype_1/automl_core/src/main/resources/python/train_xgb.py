
import argparse
import pandas as pd
import numpy as np
import pickle
import sys
import xgboost as xgb

parser = argparse.ArgumentParser(description='Train XGBoost')
parser.add_argument('--data', required=True, help='Path to training data csv')
parser.add_argument('--output', required=True, help='Path to save model pickle')
parser.add_argument('--n_estimators', type=int, default=100)
parser.add_argument('--max_depth', type=int, default=6)
parser.add_argument('--learning_rate', type=float, default=0.3)
args = parser.parse_args()

def train():
    try:
        print(f"Loading data from {args.data}")
        df = pd.read_csv(args.data)
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values

        # Encode labels to 0..N-1 to satisfy XGBoost use_label_encoder=False requirement
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y = le.fit_transform(y)

        # Check for GPU
        # In modern XGBoost, we set device="cuda"
        # We try to create a GPU classifier
        try:
            print("Attempting to use XGBoost GPU training...")
            model = xgb.XGBClassifier(
                n_estimators=args.n_estimators,
                max_depth=args.max_depth,
                learning_rate=args.learning_rate,
                tree_method="gpu_hist", # Valid for slightly older versions, widely compatible
                # device="cuda" # For bleeding edge, usually gpu_hist is safe alias
                use_label_encoder=False,
                eval_metric="logloss"
            )
            model.fit(X, y)
            print("XGBoost GPU training complete.")
        except Exception as e:
            print(f"GPU Training failed ({e}). Falling back to CPU.", file=sys.stderr)
            model = xgb.XGBClassifier(
                n_estimators=args.n_estimators,
                max_depth=args.max_depth,
                learning_rate=args.learning_rate,
                tree_method="hist",
                use_label_encoder=False,
                eval_metric="logloss"
            )
            model.fit(X, y)
            print("XGBoost CPU training complete.")

        with open(args.output, 'wb') as f:
            pickle.dump(model, f)
        print("Model saved.")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    train()
