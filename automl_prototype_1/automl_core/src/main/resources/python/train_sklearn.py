
import argparse
import pandas as pd
import numpy as np
import pickle
import sys
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier

parser = argparse.ArgumentParser(description='Train Scikit-Learn Models')
parser.add_argument('--data', required=True, help='Path to training data csv')
parser.add_argument('--output', required=True, help='Path to save model pickle')
parser.add_argument('--model_type', required=True, choices=['adaboost', 'gbm', 'dt', 'xgboost'], help='Type of model to train')
parser.add_argument('--n_estimators', type=int, default=100)
parser.add_argument('--max_depth', type=int, default=3)
parser.add_argument('--learning_rate', type=float, default=0.1)
args = parser.parse_args()

def train():
    try:
        print(f"Loading data from {args.data}")
        df = pd.read_csv(args.data)
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values

        print(f"Training {args.model_type}...")
        
        if args.model_type == 'adaboost':
            model = AdaBoostClassifier(
                n_estimators=args.n_estimators,
                learning_rate=args.learning_rate,
                algorithm='SAMME.R'
            )
        
        elif args.model_type == 'gbm':
            model = GradientBoostingClassifier(
                n_estimators=args.n_estimators,
                max_depth=args.max_depth,
                learning_rate=args.learning_rate
            )
            
        elif args.model_type == 'dt':
            model = DecisionTreeClassifier(
                max_depth=None if args.max_depth <= 0 else args.max_depth
            )

        elif args.model_type == 'xgboost':
            try:
                import xgboost as xgb
                # Check for GPU
                try:
                    # Attempt to use GPU parameters
                    model = xgb.XGBClassifier(
                        n_estimators=args.n_estimators,
                        max_depth=args.max_depth,
                        learning_rate=args.learning_rate,
                        tree_method='gpu_hist', 
                        predictor='gpu_predictor',
                        use_label_encoder=False,
                        eval_metric='logloss'
                    )
                    # Simple fit to check if GPU works (lazy check) or just trust config
                    model.fit(X, y)
                    print("Trained XGBoost with GPU parameters.")
                except Exception as gpu_e:
                    print(f"XGBoost GPU failed ({gpu_e}), falling back to CPU.")
                    model = xgb.XGBClassifier(
                        n_estimators=args.n_estimators,
                        max_depth=args.max_depth,
                        learning_rate=args.learning_rate,
                        n_jobs=-1,
                        use_label_encoder=False,
                        eval_metric='logloss'
                    )
                    model.fit(X, y)
            except ImportError:
                print("XGBoost library not found. Using sklearn GradientBoosting as fallback.")
                model = GradientBoostingClassifier(
                    n_estimators=args.n_estimators,
                    max_depth=args.max_depth,
                    learning_rate=args.learning_rate
                )
                model.fit(X, y)

        if args.model_type != 'xgboost': # XGBoost already fitted in block above
            model.fit(X, y)
            
        print(f"{args.model_type} training complete.")

        with open(args.output, 'wb') as f:
            pickle.dump(model, f)
        print("Model saved.")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    train()
