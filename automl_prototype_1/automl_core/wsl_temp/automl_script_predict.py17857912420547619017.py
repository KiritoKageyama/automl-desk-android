
import argparse
import pandas as pd
import numpy as np
import pickle
import sys
import os

parser = argparse.ArgumentParser(description='Predict using saved model')
parser.add_argument('--model', required=True, help='Path to model pickle')
parser.add_argument('--input', required=True, help='Path to input features csv')
parser.add_argument('--output', required=True, help='Path to save predictions csv')
args = parser.parse_args()

def predict():
    try:
        # Load Data
        df = pd.read_csv(args.input)
        X = df.values.astype(np.float32)

        # Load Model
        with open(args.model, 'rb') as f:
            model = pickle.load(f)

        # Predict
        # Check if it is a cuML model (requires data on GPU)
        model_type = str(type(model))
        if 'cuml' in model_type:
            try:
                import cudf
                X_gpu = cudf.DataFrame(X)
                preds = model.predict(X_gpu).to_numpy()
            except ImportError:
                 # This assumes if we loaded a cuml model, we must have cuml installed.
                 # But if environment changed, this might fail.
                 print("Error: Loaded cuML model but cannot import cudf.", file=sys.stderr)
                 sys.exit(1)
        else:
            # Scikit-learn
            preds = model.predict(X)

        # Save preds
        # We output usually just the class, or appended? 
        # Interface expects int[] usually. Let's save as single column CSV, no header
        np.savetxt(args.output, preds, fmt='%d', delimiter=',')
        print(f"Predictions saved to {args.output}")

    except Exception as e:
        print(f"Error during prediction: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    predict()
