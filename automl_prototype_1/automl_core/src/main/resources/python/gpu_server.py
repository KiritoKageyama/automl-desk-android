
import http.server
import socketserver
import json
import os
import sys
import threading
import time
import uuid
import traceback
import signal
import numpy as np

# --- Global State ---
MODELS = {} # Cache loaded models: {model_id: model_obj}
DATASETS = {} # Cache datasets: {id: {X:..., y:..., location:...}}
CUMULATIVE_BATCHES = 0
PORT = 5006
print(f"[Server] PATCHED VERSION LOADED. Starting on {PORT}...")

# Ensure we can import libraries
try:
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.neural_network import MLPClassifier
    # --- Granular GPU Checks ---
    try:
        import cuml
        from cuml.ensemble import RandomForestClassifier as CuRF
        from cuml.svm import SVC as CuSVC
        CUML_AVAILABLE = True
        print("[Server] cuML found. GPU acceleration ENABLED for RF/SVM.")
    except ImportError:
        CUML_AVAILABLE = False
        print("[Server] cuML NOT found. RF/SVM will use CPU.")

    # Check LightGBM GPU
    LGBM_GPU_AVAILABLE = False
    try:
        import lightgbm as lgb
        # We can't easily check for GPU binary without running, but we assume
        # if the user installed it they hope for GPU. We will try-catch later.
        LGBM_AVAILABLE = True
        print("[Server] LightGBM found.")
    except ImportError:
        LGBM_AVAILABLE = False
        print("[Server] LightGBM NOT found.")

    # Check XGBoost GPU
    XGB_GPU_AVAILABLE = False
    try:
        import xgboost as xgb
        XGB_AVAILABLE = True
        print("[Server] XGBoost found.")
    except ImportError:
        XGB_AVAILABLE = False
        print("[Server] XGBoost NOT found.")
        
    # Generic Global Flag (Legacy support, though we should prefer granular)
    GPU_AVAILABLE = CUML_AVAILABLE # For backward compat with valid checks
        
    # Phase 7: Arrow Support
    try:
        import pyarrow as pa
        ARROW_AVAILABLE = True
        print("[Server] PyArrow imported successfully. Zero-Copy Transfer ENABLED.")
    except ImportError:
        ARROW_AVAILABLE = False
        print("[Server] PyArrow NOT found. Using JSON Fallback.")

    # Phase 8: ONNX Support
    try:
        import onnx
        from skl2onnx import to_onnx
        ONNX_AVAILABLE = True
        print("[Server] ONNX tools imported. Model Export ENABLED.")
    except ImportError:
        ONNX_AVAILABLE = False
        print("[Server] ONNX tools NOT found. Model Export DISABLED.")

    # Phase 7: PyTorch Support
    try:
        import torch
        # We assume mlp_algorithm is in the same directory or path
        # In a real deployment, we might need sys.path.append
        sys.path.append(os.getcwd()) 
        import mlp_algorithm
        TORCH_AVAILABLE = True
        print("[Server] PyTorch found. GPU MLP ENABLED.")
    except ImportError as e:
        TORCH_AVAILABLE = False
        print(f"[Server] PyTorch/MLP Script NOT found ({e}). MLP will use CPU (Sklearn).")

except ImportError as e:
    print(f"[Server] CRITICAL: Failed to import basic libraries: {e}")
    sys.exit(1)

def export_to_onnx(model, model_uid, X_sample, algo_type):
    if not ONNX_AVAILABLE: return None
    try:
        import tempfile
        temp_dir = os.path.join(os.getcwd(), "wsl_temp")
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir, exist_ok=True)
        onnx_filename = f"model_{model_uid}.onnx"
        onnx_path = os.path.join(temp_dir, onnx_filename)

        # PyTorch / Transformer Models
        if algo_type in ['MLP', 'CNN']:
             if TORCH_AVAILABLE:
                  # Input shape depends on model. MLP: (1, n_features), CNN: (1, 1, H, W) or similar
                  # We need X_sample to infer shape.
                  dummy_input = torch.from_numpy(X_sample[:1].astype(np.float32))
                  torch.onnx.export(model, dummy_input, onnx_path, verbose=False)
                  return onnx_path
        
        if algo_type == 'BERT':
             if TORCH_AVAILABLE:
                  # BERT is complex, usually requires tokenizer input (input_ids, attention_mask)
                  # For prototype, we might skip true ONNX or use a dummy input if we have the tokenizer.
                  # bert_finetune.py might return a wrapper that handles this?
                  # For now, we'll try standard export if it's a torch module
                  if isinstance(model, torch.nn.Module):
                       # We need dummy input_ids and attention_mask
                       # This is hard without the tokenizer which is inside bert_finetune
                       # Let's try to load them or fail gracefully
                       print("[Server] ONNX Export for BERT is experimental.")
                       # dummy_input = (torch.zeros((1, 128), dtype=torch.long), torch.zeros((1, 128), dtype=torch.long))
                       # torch.onnx.export(model, dummy_input, onnx_path, input_names=['input_ids', 'attention_mask'])
                       # return onnx_path
                       return None 

        if algo_type == 'XGB':
            model.save_model(onnx_path)
            return onnx_path
        if algo_type == 'CATBOOST':
            model.save_model(onnx_path, format="onnx")
            return onnx_path
        if algo_type == 'LGBM':
            try:
                from onnxmltools import convert_lightgbm
                from onnxmltools.convert.common.data_types import FloatTensorType
                initial_types = [('input', FloatTensorType([None, X_sample.shape[1]]))]
                onnx_model = convert_lightgbm(model, initial_types=initial_types)
                onnx.save(onnx_model, onnx_path)
                return onnx_path
            except: 
                return None
        is_cuml = hasattr(model, 'handle')
        if not is_cuml:
            onx = to_onnx(model, X_sample[:1].astype(np.float32))
            with open(onnx_path, "wb") as f: f.write(onx.SerializeToString())
            return onnx_path
    except Exception as e:
        print(f"[Server] Export Failed: {e}")
    return None

class GpuRequestHandler(http.server.BaseHTTPRequestHandler):
    
    def _send_response(self, code, data):
        self.send_response(code)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode('utf-8'))

    def do_POST(self):
        # Route based on path
        if self.path == '/train_arrow':
            self.handle_train_arrow()
            return

        if self.path == '/upload_arrow':
            self.handle_upload_arrow()
            return

        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        
        try:
            request = json.loads(post_data.decode('utf-8'))
            action = request.get('action')
            print(f"[Server] DEBUG: Received Action: {action} Keys: {list(request.keys())}")
            
            if action == 'PING':
                status = {
                    'status': 'ok',
                    'gpu_global': GPU_AVAILABLE,
                    'arrow_support': ARROW_AVAILABLE,
                    'backends': {
                        'rf': 'cuml' if GPU_AVAILABLE else 'sklearn',
                        'svm': 'cuml' if GPU_AVAILABLE else 'sklearn',
                        'xgb': 'unknown'
                    }
                }
                # Check XGBoost specifically
                try:
                    from xgboost import XGBClassifier
                    # quick check for gpu support (naive)
                    status['backends']['xgb'] = 'installed'
                except ImportError:
                    status['backends']['xgb'] = 'missing'

                self._send_response(200, status)
                return

            if action == 'TRAIN_BATCH':
                self.handle_train_batch(request)
            elif action == 'PREDICT_BATCH':
                self.handle_predict_batch(request)
            elif action == 'UPLOAD_DATASET':
                self.handle_upload_dataset(request)
            elif action == 'EXPORT_MODEL':
                self.handle_export_model(request)
            elif action == 'CLEANUP':
                self.handle_cleanup(request)
            else:
                self._send_response(400, {'error': f'Unknown action: {action}'})
                
        except Exception as e:
            traceback.print_exc()
            self._send_response(500, {'error': str(e)})

    def handle_train_arrow(self):
        if not ARROW_AVAILABLE:
            self._send_response(500, {'error': 'PyArrow not installed on server'})
            return

        try:
            content_length = int(self.headers['Content-Length'])
            # Read binary body
            body = self.rfile.read(content_length)
            
            # Read Headers for Metadata
            algo_type = self.headers.get('X-Algo-Type', 'RF')
            params_json = self.headers.get('X-Params', '{}')
            params = json.loads(params_json)
            
            # Parse Arrow Stream
            reader = pa.ipc.open_stream(body)
            table = reader.read_all()
            
            # Extract Features and Labels
            # Expecting columns "f0", "f1"... and "target"
            df = table.to_pandas()
            
            print(f"[Server] Arrow DEBUG: Headers={{'X-Algo-Type': '{algo_type}'}} DF Columns: {df.columns.tolist()}")

            if 'target' not in df.columns:
                 self._send_response(400, {'error': 'Arrow IPC missing "target" column'})
                 return

            y = df['target'].to_numpy()
            X = df.drop(columns=['target']).to_numpy()
            
            # --- DEBUG DATA QUALITY ---
            try:
                print(f"[Server] DATA DEBUG: X_shape={X.shape} y_shape={y.shape}")
                print(f"[Server] DATA DEBUG: X_mean={np.nanmean(X):.4f} X_std={np.nanstd(X):.4f} NonZeros={np.count_nonzero(X)}")
                print(f"[Server] DATA DEBUG: y_mean={np.nanmean(y):.4f} y_unique={np.unique(y)}")
            except:
                pass
            # --------------------------

            # Reconstruct "Job" structure for shared logic
            # Arrow request is fundamentally ONE job per request in this impl
            job_id = str(uuid.uuid4())
            
            # Execute Training (Reuse logic logic?)
            # Just inline for now to ensure correctness
            results = []
            
            # --- HELPER: CPU Fallback Creator (Duplicate for scope) ---
            def get_cpu_model(algo, p):
                if algo == 'RF':
                    return RandomForestClassifier(n_estimators=int(p.get('n_estimators', 100)), n_jobs=-1)
                if algo == 'SVM':
                    return SVC(C=float(p.get('C', 1.0)), kernel=p.get('kernel', 'rbf'), probability=True)
                return None

            try:
                model = None
                if algo_type == 'RF':
                    n_estimators = int(params.get('n_estimators', 100))
                    if GPU_AVAILABLE:
                        try:
                            model = CuRF(n_estimators=n_estimators)
                            model.fit(X, y)
                        except Exception as gpu_err:
                            print(f"[Server] Arrow GPU RF Failed ({gpu_err}). Falling back to CPU.")
                            model = get_cpu_model('RF', params)
                            if model: model.fit(X, y)
                    else:
                        model = get_cpu_model('RF', params)
                        if model: model.fit(X, y)
                
                elif algo_type == 'SVM':
                    C = float(params.get('C', 1.0))
                    kernel = params.get('kernel', 'rbf')
                    if GPU_AVAILABLE:
                         try:
                             model = CuSVC(C=C, kernel=kernel, probability=True)
                             model.fit(X, y)
                         except Exception as gpu_err:
                             print(f"[Server] Arrow GPU SVM Failed ({gpu_err}). Falling back to CPU.")
                             model = get_cpu_model('SVM', params)
                             if model: model.fit(X, y)
                    else:
                         model = get_cpu_model('SVM', params)
                         if model: model.fit(X, y)

                elif algo_type == 'XGB':
                    n_estimators = int(params.get('n_estimators', 100))
                    max_depth = int(params.get('max_depth', 6))
                    learning_rate = float(params.get('learning_rate', 0.3))
                    
                    try:
                        from xgboost import XGBClassifier
                        if GPU_AVAILABLE:
                            try:
                                 model = XGBClassifier(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, tree_method='gpu_hist', eval_metric='logloss')
                                 model.fit(X, y)
                            except Exception as xgb_gpu_err:
                                 print(f"[Server] Arrow XGB GPU Failed ({xgb_gpu_err})")
                                 model = XGBClassifier(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, n_jobs=-1, eval_metric='logloss')
                                 model.fit(X, y)
                        else:
                             model = XGBClassifier(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, n_jobs=-1, eval_metric='logloss')
                             model.fit(X, y)
                    except ImportError:
                         raise Exception("XGBoost not installed")
                
                elif algo_type == 'LGBM':
                    n_estimators = int(params.get('n_estimators', 100))
                    learning_rate = float(params.get('learning_rate', 0.1))
                    try:
                        import lightgbm as lgb
                        if GPU_AVAILABLE:
                            try:
                                model = lgb.LGBMClassifier(n_estimators=n_estimators, learning_rate=learning_rate, device_type='gpu')
                                model.fit(X, y)
                            except Exception as lgbm_gpu_err:
                                print(f"[Server] Arrow LGBM GPU Failed ({lgbm_gpu_err}). Trying CPU.")
                                model = lgb.LGBMClassifier(n_estimators=n_estimators, learning_rate=learning_rate, n_jobs=-1)
                                model.fit(X, y)
                        else:
                             model = lgb.LGBMClassifier(n_estimators=n_estimators, learning_rate=learning_rate, n_jobs=-1)
                             model.fit(X, y)
                    except ImportError:
                         raise Exception("LightGBM not installed")
                
                if model:
                     # model is already fitted in blocks above
                     model_uid = str(uuid.uuid4())
                     MODELS[model_uid] = model
                     
                     # Try Export ONNX
                     onnx_path = export_to_onnx(model, model_uid, X, algo_type)
                     
                     results.append({"id": job_id, "model_id": model_uid, "onnx_path": onnx_path, "success": True})
                else:
                     results.append({"id": job_id, "error": "Unknown Algo", "success": False})

            except Exception as e:
                print(f"[Server] Arrow Train Failed: {e}")
                results.append({"id": job_id, "error": str(e), "success": False})

            self._send_response(200, {"results": results})

        except Exception as e:
            traceback.print_exc()
            self._send_response(500, {'error': f"Arrow processing failed: {str(e)}"})

    def handle_train_batch(self, request):
        """
        Expects:
        {
            "jobs": [
                {"id": "job1", "type": "RF", "params": {...}, "features": [[...]], "labels": [...]},
                ...
            ]
        }
        Returns:
        {
            "results": [
                {"id": "job1", "model_id": "uuid...", "success": true},
                ...
            ]
        }
        """
        jobs = request.get('jobs', [])
        results = []
        
        print(f"[Server] Processing TRAIN batch of {len(jobs)} jobs...")
        
        for job in jobs:
            job_id = job.get('id')
            algo_type = job.get('type')
            params = job.get('params', {})
            
            # Resolve Data (Direct or Cached)
            X = None
            y = None
            dataset_id = job.get('dataset_id')
            
            if dataset_id:
                if dataset_id not in DATASETS:
                    results.append({"id": job_id, "error": f"Dataset {dataset_id} not found", "success": False})
                    continue
                ds = DATASETS[dataset_id]
                X_obj = ds['X']
                y_obj = ds['y']
                is_gpu = (ds['location'] == 'gpu')
                
                rows = job.get('row_indices')
                cols = job.get('feature_indices')
                
                try:
                    # Slicing
                    if is_gpu:
                         if rows: X_obj = X_obj.iloc[rows]; y_obj = y_obj.iloc[rows]
                         if cols: X_obj = X_obj.iloc[:, cols]
                    else:
                         if rows: X_obj = X_obj[rows]; y_obj = y_obj[rows]
                         if cols: X_obj = X_obj[:, cols]
                    X = X_obj
                    y = y_obj
                except Exception as slice_err:
                     results.append({"id": job_id, "error": f"Slicing Error: {slice_err}", "success": False})
                     continue
            else:
                # Legacy Direct
                raw_feats = job.get('features')
                if raw_feats is None:
                    results.append({"id": job_id, "error": "No features/dataset_id provided", "success": False})
                    continue
                X = np.array(raw_feats)
                y = np.array(job.get('labels'))

            # Helper to convert to CPU (numpy) if needed
            def to_cpu(d):
                if hasattr(d, 'to_numpy'): return d.to_numpy()
                return d
            
            try:
                model = None
                
                # --- HELPER: CPU Fallback Creator ---
                def get_cpu_model(algo, p):
                    if algo == 'RF':
                        return RandomForestClassifier(n_estimators=int(p.get('n_estimators', 100)), n_jobs=-1)
                    if algo == 'SVM':
                        return SVC(C=float(p.get('C', 1.0)), kernel=p.get('kernel', 'rbf'), probability=True)
                    return None

                # --- RF ---
                if algo_type == 'RF':
                    n_estimators = int(params.get('n_estimators', 100))
                    if CUML_AVAILABLE:
                        try:
                            # Attempt GPU Fit
                            model = CuRF(n_estimators=n_estimators)
                            model.fit(X, y) # Critical Point
                        except Exception as gpu_err:
                            print(f"[Server] GPU RF Failed ({gpu_err}). Falling back to CPU.")
                            model = get_cpu_model('RF', params)
                            if model: model.fit(to_cpu(X), to_cpu(y))
                    else:
                        model = get_cpu_model('RF', params)
                        if model: model.fit(to_cpu(X), to_cpu(y))
                
                # --- SVM ---
                elif algo_type == 'SVM':
                    C = float(params.get('C', 1.0))
                    kernel = params.get('kernel', 'rbf')
                    if CUML_AVAILABLE:
                        try:
                            model = CuSVC(C=C, kernel=kernel, probability=True)
                            model.fit(X, y)
                        except Exception as gpu_err:
                            print(f"[Server] GPU SVM Failed ({gpu_err}). Falling back to CPU.")
                            model = get_cpu_model('SVM', params)
                            if model: model.fit(to_cpu(X), to_cpu(y))
                    else:
                        model = get_cpu_model('SVM', params)
                        if model: model.fit(to_cpu(X), to_cpu(y))

                # --- XGBoost ---
                elif algo_type == 'XGB':
                    # XGBoost handles GPU internally via 'tree_method' param
                    n_estimators = int(params.get('n_estimators', 100))
                    max_depth = int(params.get('max_depth', 6))
                    learning_rate = float(params.get('learning_rate', 0.3))
                    
                    if XGB_AVAILABLE:
                        from xgboost import XGBClassifier
                        # Try GPU first
                        try:
                            model = XGBClassifier(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, tree_method='gpu_hist', eval_metric='logloss')
                            model.fit(X, y)
                            print("[Server] XGBoost trained on GPU.")
                        except Exception as xgb_gpu_err:
                            print(f"[Server] XGB GPU Failed ({xgb_gpu_err}). Trying CPU.")
                            model = XGBClassifier(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, n_jobs=-1, eval_metric='logloss')
                            model.fit(X, y)
                    else:
                        results.append({"id": job_id, "error": "XGBoost not installed", "success": False})
                        continue
                
                # --- LightGBM ---
                # --- LightGBM ---
                elif algo_type == 'LGBM':
                    n_estimators = int(params.get('n_estimators', 100))
                    learning_rate = float(params.get('learning_rate', 0.1))
                    
                    if LGBM_AVAILABLE:
                        import lightgbm as lgb
                        # Try GPU first
                        try:
                            # --- AUTO-CPU FOR SMALL DATASETS (With Override) ---
                            force_gpu = params.get('force_gpu', False)
                            # Also check meta-flag if available? Arrow header handles it separately?
                            # For Arrow, params is passed via header, so we can put force_gpu there.
                            
                            if X.shape[0] < 2000 and not force_gpu:
                                print(f"[Server] Small Dataset ({X.shape[0]} rows). Forcing CPU for LightGBM stability. (Override with force_gpu=True)")
                                raise Exception("Force CPU for small data")
                            if force_gpu and X.shape[0] < 2000:
                                print(f"[Server] Force GPU Enabled for small LightGBM dataset.")
                            # -----------------------------------

                            # LightGBM GPU support requires specific installation, but we try 'gpu' device_type
                            model = lgb.LGBMClassifier(n_estimators=n_estimators, learning_rate=learning_rate, device_type='gpu')
                            model.fit(X, y)
                            print("[Server] LightGBM trained on GPU.")
                        except Exception as lgbm_gpu_err:
                            if "Force CPU" in str(lgbm_gpu_err): # Expected fallback
                                pass 
                            else:
                                print(f"[Server] LGBM GPU Failed ({lgbm_gpu_err}). Falling back to CPU.")
                            model = lgb.LGBMClassifier(n_estimators=n_estimators, learning_rate=learning_rate, n_jobs=-1)
                            model.fit(X, y)
                    else:
                        results.append({"id": job_id, "error": "LightGBM not installed", "success": False})
                        continue

    # Phase 7: PyTorch Support
    try:
        import torch
        import mlp_algorithm
        TORCH_AVAILABLE = True
        print("[Server] PyTorch found. GPU MLP ENABLED.")
    except ImportError:
        TORCH_AVAILABLE = False
        print("[Server] PyTorch NOT found. MLP will use CPU (Sklearn).")

    # ... (existing imports)

# ... inside handle_train_batch ...

                # --- MLP (Neural Network) ---
                elif algo_type == 'MLP':
                    # Parse hidden layers: "100,50" -> (100, 50)
                    layers_str = str(params.get('hidden_layer_sizes', "100")).strip()
                    try:
                        hidden_layers = tuple(map(int, layers_str.split(',')))
                    except:
                        hidden_layers = (100,)
                    
                    activation = params.get('activation', 'relu')
                    solver = params.get('solver', 'adam')
                    alpha = float(params.get('alpha', 0.0001))
                    max_iter = int(params.get('max_iter', 200))

                    if TORCH_AVAILABLE:
                         # Use our new PyTorch backend
                         try:
                             print("[Server] Training MLP with PyTorch (GPU if available)...")
                             model = mlp_algorithm.train(X, y, params)
                         except Exception as torch_err:
                             print(f"[Server] PyTorch MLP Failed ({torch_err}). Falling back to Sklearn.")
                             model = MLPClassifier(hidden_layer_sizes=hidden_layers, activation=activation, solver=solver, alpha=alpha, max_iter=max_iter, random_state=42)
                             model.fit(X, y)
                    else:
                        # Sklearn MLP is CPU-based (standard)
                        model = MLPClassifier(hidden_layer_sizes=hidden_layers, activation=activation, solver=solver, alpha=alpha, max_iter=max_iter, random_state=42)
                        model.fit(X, y)
                    
                    print(f"[Server] MLP trained. Layers={hidden_layers}")

    try:
        sys.path.append(os.getcwd()) 
        import bert_finetune
        BERT_AVAILABLE = True
        print("[Server] BERT Script found. Gen AI Text Training ENABLED.")
    except ImportError as e:
        BERT_AVAILABLE = False
        print(f"[Server] BERT Script NOT found ({e}). Gen AI Disabled.")

# ... inside handle_train_batch ...

                # --- BERT (Gen AI) ---
                elif algo_type == 'BERT':
                    if not BERT_AVAILABLE:
                         raise Exception("BERT script not found or transformers library missing.")
                    
                    # Check for text paths in params
                    text_paths = params.get('text_paths')
                    
                    if text_paths:
                        # Use paths as X
                        print(f"[Server] Training BERT on {len(text_paths)} text files...")
                        model = bert_finetune.train(text_paths, y, params)
                    else:
                        raise Exception("BERT requires 'text_paths' parameter.")
                        
                    print(f"[Server] BERT Model Trained.")

# ... existing code ...

                    script_path = params.get('script_path')
                    if not script_path or not os.path.exists(script_path):
                        raise Exception(f"Custom script not found: {script_path}")
                    
                    try:
                        import importlib.util
                        # Load module dynamically
                        spec = importlib.util.spec_from_file_location("custom_model_" + job_id, script_path)
                        if spec is None or spec.loader is None:
                             raise Exception(f"Failed to load spec for {script_path}")
                        
                        module = importlib.util.module_from_spec(spec)
                        try:
                            sys.modules["custom_model_" + job_id] = module
                            spec.loader.exec_module(module)
                        except Exception as load_err:
                             raise Exception(f"Script Execution Failed: {load_err}")

                        # Expect a 'train' or 'fit' function
                        if hasattr(module, 'train'):
                            model = module.train(X, y, params)
                        elif hasattr(module, 'fit'):
                            model = module.fit(X, y, params)
                        else:
                            raise Exception("Custom script must define a 'train(X, y, params)' or 'fit(X, y, params)' function.")
                            
                        print(f"[Server] Custom Script executed. Model type: {type(model)}")
                        
                    except Exception as e:
                        print(f"[Server] Custom Script Error: {e}")
                        traceback.print_exc()
                        raise e

                if model:
                    # If we fell back, model is fitted. Use it.
                    # Note: We skipped explicit fit() calls above for non-GPU paths to simplify structure, 
                    # but logic needs to be cleaner.
                    # Reworked above to call fit() inside blocks.
                    
                    model_uid = str(uuid.uuid4())
                    MODELS[model_uid] = model
                    
                    # Try Export ONNX
                    onnx_path = export_to_onnx(model, model_uid, X, algo_type)
                    
                    results.append({"id": job_id, "model_id": model_uid, "onnx_path": onnx_path, "success": True})
                else:
                    results.append({"id": job_id, "error": "Unknown Algo or Init Failed", "success": False})

            except Exception as e:
                print(f"[Server] Job {job_id} failed: {e}")
                results.append({"id": job_id, "error": str(e), "success": False})
        
        self._send_response(200, {"results": results})

    def handle_predict_batch(self, request):
        """
        Expects:
        {
            "jobs": [
                {"id": "job1", "model_id": "uuid...", "features": [[...]]},
                ...
            ]
        }
        Returns:
        {
            "results": [
                {"id": "job1", "predictions": [...], "success": true},
                ...
            ]
        }
        """
        jobs = request.get('jobs', [])
        results = []
        
        print(f"[Server] Processing PREDICT batch of {len(jobs)} jobs...")
        
        for job in jobs:
            job_id = job.get('id')
            model_id = job.get('model_id')
            
            # Resolve Data
            X = None
            dataset_id = job.get('dataset_id')
            if dataset_id:
                if dataset_id not in DATASETS:
                    results.append({"id": job_id, "error": "Dataset not found", "success": False})
                    continue
                ds = DATASETS[dataset_id]
                X_obj = ds['X']
                is_gpu = (ds['location'] == 'gpu')
                
                rows = job.get('row_indices')
                cols = job.get('feature_indices')

                # --- DATA DEBUG (PREDICT) ---
                try:
                    # Log first job only to reduce spam
                    if jobs.index(job) == 0:
                        print(f"[Server] PREDICT DEBUG: Job={job_id} Dataset={dataset_id} RowC={len(rows) if rows else 'All'} ColC={len(cols) if cols else 'All'}")
                        if cols: print(f"[Server] PREDICT DEBUG: Requested Cols: {cols[:10]}...")
                except: pass
                # ----------------------------
                
                try:
                    if is_gpu:
                        if rows: X_obj = X_obj.iloc[rows]
                        if cols: X_obj = X_obj.iloc[:, cols]
                    else:
                        if rows: X_obj = X_obj[rows]
                        if cols: X_obj = X_obj[:, cols]
                    X = X_obj
                    
                    # --- DATA DEBUG (POST-SLICE) ---
                    try:
                        if jobs.index(job) == 0:
                            print(f"[Server] PREDICT DEBUG: Sliced X_shape={X.shape}")
                    except: pass
                    # -------------------------------

                except Exception as e:
                     results.append({"id": job_id, "error": f"Slice Error: {e}", "success": False})
                     continue
            else:
                 X = np.array(job.get('features'))
            
            try:
                model = MODELS.get(model_id)
                if model:
                    preds = model.predict(X)
                    # Convert to list for JSON serialization (handle numpy types)
                    preds_list = preds.tolist() if isinstance(preds, np.ndarray) else list(preds)
                    # Ensure int for classification
                    preds_list = [int(p) for p in preds_list]
                    results.append({"id": job_id, "predictions": preds_list, "success": True})
                else:
                    results.append({"id": job_id, "error": "Model not found", "success": False})
            except Exception as e:
                print(f"[Server] Predict Job {job_id} failed: {e}")
                results.append({"id": job_id, "error": str(e), "success": False})

        self._send_response(200, {"results": results})

    def handle_upload_arrow(self):
        if not ARROW_AVAILABLE:
            self._send_response(500, {'error': 'PyArrow not installed on server'})
            return

        try:
            content_length = int(self.headers['Content-Length'])
            body = self.rfile.read(content_length)
            
            # Read Arrow Stream
            reader = pa.ipc.open_stream(body)
            table = reader.read_all()
            df = table.to_pandas()
            
            print(f"[Server] Arrow Upload Cols: {df.columns.tolist()}")
            
            # Extract target
            if 'target' in df.columns:
                y = df['target'].to_numpy()
                X = df.drop(columns=['target']).to_numpy()
            else:
                y = df.iloc[:, -1].to_numpy()
                X = df.iloc[:, :-1].to_numpy()

            dataset_id = str(uuid.uuid4())
            
            # Cache
            if GPU_AVAILABLE:
                import cudf
                DATASETS[dataset_id] = {
                    'X': cudf.DataFrame(X),
                    'y': cudf.DataFrame(y) if len(y.shape) > 1 else cudf.Series(y),
                    'location': 'gpu'
                }
            else:
                DATASETS[dataset_id] = {
                    'X': X,
                    'y': y,
                    'location': 'cpu'
                }
            
            print(f"[Server] Cached Dataset {dataset_id} via Arrow on {DATASETS[dataset_id]['location']}")
            self._send_response(200, {'id': dataset_id, 'success': True})

        except Exception as e:
            traceback.print_exc()
            self._send_response(500, {'error': f"Arrow Upload Failed: {str(e)}"})

    def handle_export_model(self, request):
        """
        Expects: {"model_id": "uuid"}
        Returns: {"path": "/path/to/model.onnx", "success": true}
        """
        model_id = request.get('model_id')
        if not model_id or model_id not in MODELS:
            self._send_response(404, {'error': 'Model not found'})
            return

        # We assume X_sample usually cached or we try to synthesize based on model input checks?
        # Actually export_to_onnx needs X_sample.
        # Ideally we stored X_sample during training or reference it from DATASETS?
        # For this prototype, we'll check if we can get a sample from a recent dataset
        # OR we just rely on what export_to_onnx can do without it (limited)
        
        # Hack: Use a dummy sample if we have a dataset loaded
        # We need a robust way to get shape.
        # Let's try to find *any* dataset that fits? No.
        # We will assume export_to_onnx runs immediately after training where X is available,
        # OR we fail if we can't find data.
        
        # Improvement: We can store input_shape in MODELS metadata.
        # For now, let's create a dummy input based on model attributes if possible.
        
        dataset_id = request.get('dataset_id') # Optional: user can provide dataset context
        X_sample = np.zeros((1, 10)) # Fallback default
        
        if dataset_id and dataset_id in DATASETS:
             ds = DATASETS[dataset_id]
             X_obj = ds['X']
             if hasattr(X_obj, 'iloc'): X_sample = X_obj.iloc[:1].to_numpy()
             else: X_sample = X_obj[:1]
        
        model = MODELS[model_id]
        
        # Try to infer type
        algo_type = 'UNKNOWN'
        if hasattr(model, 'predict'):
             # Heuristics
             name = str(type(model))
             if 'RandomForest' in name: algo_type = 'RF'
             elif 'XGB' in name: algo_type = 'XGB'
             elif 'MLP' in name: algo_type = 'MLP'
             elif 'SVC' in name: algo_type = 'SVM'
             elif 'Torch' in name or 'Net' in name: algo_type = 'MLP' # Broad catch

        onnx_path = export_to_onnx(model, model_id, X_sample, algo_type)
        
        if onnx_path:
             self._send_response(200, {'path': onnx_path, 'success': True})
        else:
             self._send_response(500, {'error': 'Export failed or not supported'})

    def handle_upload_dataset(self, request):
        """
        Expects:
        {
            "upload_features": [[...]],
            "upload_labels": [...]
        }
        Returns:
        {
            "id": "dataset_id",
            "success": true
        }
        """
        try:
            features = request.get('upload_features')
            labels = request.get('upload_labels')
            
            if features is None or labels is None:
                self._send_response(400, {'error': 'Missing upload_features or upload_labels'})
                return

            X = np.array(features)
            y = np.array(labels)
            
            dataset_id = str(uuid.uuid4())
            
            # Initialize Cache
            if GPU_AVAILABLE:
                import cudf
                # Cache on GPU
                DATASETS[dataset_id] = {
                    'X': cudf.DataFrame(X),
                    'y': cudf.Series(y),
                    'location': 'gpu'
                }
            else:
                # Cache on CPU
                DATASETS[dataset_id] = {
                    'X': X,
                    'y': y,
                    'location': 'cpu'
                }
                
            print(f"[Server] Cached Dataset {dataset_id} on {DATASETS[dataset_id]['location']}")
            self._send_response(200, {'id': dataset_id, 'success': True})
            
        except Exception as e:
            traceback.print_exc()
            self._send_response(500, {'error': str(e)})


    def handle_cleanup(self, request):
        try:
            model_id = request.get('model_id')
            if model_id:
                if model_id in MODELS:
                    del MODELS[model_id]
                
                # Try delete onnx
                try:
                    import os
                    temp_dir = os.path.join(os.getcwd(), "wsl_temp")
                    p = os.path.join(temp_dir, f"model_{model_id}.onnx")
                    if os.path.exists(p):
                        os.remove(p)
                        print(f"[Server] Cleaned up model {model_id} and file {p}")
                except Exception as ex:
                    print(f"[Server] Cleanup Warning: {ex}")
            
            self._send_response(200, {'success': True})
        except Exception as e:
            self._send_response(500, {'error': str(e)})


class ReusableTCPServer(socketserver.TCPServer):
    allow_reuse_address = True

def run_server():
    server_address = ('127.0.0.1', PORT)
    httpd = None
    for i in range(10):
        try:
            httpd = ReusableTCPServer(server_address, GpuRequestHandler)
            print(f"[Server] Starting GPU Server on port {PORT}...")
            break
        except OSError as e:
            if i < 9:
                print(f"[Server] Port {PORT} busy ({e}). Retrying in 1s... ({i+1}/10)")
                import time
                time.sleep(1)
            else:
                raise e
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    httpd.server_close()
    print("[Server] Stopped.")

if __name__ == '__main__':
    # Flush stdout immediately for Java ProcessBuilder to read
    sys.stdout.reconfigure(line_buffering=True)
    run_server()
