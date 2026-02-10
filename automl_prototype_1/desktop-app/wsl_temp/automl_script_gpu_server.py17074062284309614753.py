
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
