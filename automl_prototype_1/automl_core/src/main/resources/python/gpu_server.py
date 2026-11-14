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

# --- Agent Core Integration ---
try:
    import agent_core
    AGENT_AVAILABLE = True
    print("[Server] agent_core loaded. LCA Agent Enabled.")
except ImportError:
    AGENT_AVAILABLE = False
    print("[Server] agent_core NOT found (or deps missing). LCA Agent Disabled.")

# --- Global State ---
MODELS = {} # Cache loaded models: {model_id: model_obj}
DATASETS = {} # Cache datasets: {id: {X:..., y:..., location:...}}
CUMULATIVE_BATCHES = 0
PORT = 5006
TRAINING_CANCEL = threading.Event()
# GPU auth: Java injects AUTOML_GPU_TOKEN when launching the server
GPU_AUTH_ENABLED = os.environ.get('AUTOML_GPU_AUTH', 'true').lower() != 'false'
GPU_AUTH_TOKEN = (os.environ.get('AUTOML_GPU_TOKEN') or '').strip()
GPU_AUTH_HEADER = 'X-AutoML-Gpu-Token'
# Ensure optional flags exist even if torch import path failed early
CNN_AVAILABLE = globals().get('CNN_AVAILABLE', False)
BERT_AVAILABLE = globals().get('BERT_AVAILABLE', False)

# Pre-load ViT if available
VIT_MODEL = None
VIT_PROCESSOR = None

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
        import lstm_algorithm
        TORCH_AVAILABLE = True
        print("[Server] PyTorch found. GPU MLP & LSTM ENABLED.")
        try:
            import cnn_algorithm
            CNN_AVAILABLE = True
            print("[Server] cnn_algorithm imported. CNN ENABLED.")
        except ImportError as cnn_err:
            CNN_AVAILABLE = False
            print(f"[Server] cnn_algorithm NOT found ({cnn_err}). CNN DISABLED.")
        try:
            import bert_finetune
            BERT_AVAILABLE = True
            print("[Server] bert_finetune imported. LLM fine-tune ENABLED.")
        except ImportError as bert_err:
            BERT_AVAILABLE = False
            print(f"[Server] bert_finetune NOT found ({bert_err}). LLM fine-tune DISABLED.")
    except ImportError as e:
        TORCH_AVAILABLE = False
        CNN_AVAILABLE = False
        BERT_AVAILABLE = False
        print(f"[Server] PyTorch/Scripts NOT found ({e}). MLP/LSTM will use CPU (Sklearn) or be disabled.")

    # Phase 8: Vision Transformer (ViT) Support
    try:
        from transformers import ViTImageProcessor, ViTModel
        from PIL import Image
        VIT_AVAILABLE = True
        print("[Server] Transformers (ViT) imported. Vision Intelligence ENABLED.")
    except ImportError:
        VIT_AVAILABLE = False
        print("[Server] Transformers NOT found. ViT DISABLED.")

    # Phase 8: Semantic Memory Support
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
        SENTENCE_TRANSFORMERS_AVAILABLE = True
        print("[Server] Sentence Transformers imported. Semantic Memory ENABLED.")
    except ImportError:
        SENTENCE_TRANSFORMERS_AVAILABLE = False
        print("[Server] Sentence Transformers NOT found. Memory features DISABLED.")

    # Phase 8, Step 3: Local SLM Support
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        LLM_AVAILABLE = True
        print("[Server] Transformers LLM classes imported. Reasoning ENABLED.")
    except ImportError:
        LLM_AVAILABLE = False
        print("[Server] Transformers LLM classes NOT found. Reasoning DISABLED.")

except ImportError as e:
    print(f"[Server] CRITICAL: Failed to import basic libraries: {e}")
    sys.exit(1)

# Global Memory State
EMBEDDING_MODEL = None
VECTOR_STORE = [] # List of {'text': str, 'vector': np.array, 'metadata': dict}

def get_embedding_model():
    global EMBEDDING_MODEL
    if EMBEDDING_MODEL is None and SENTENCE_TRANSFORMERS_AVAILABLE:
        print("[Server] Loading Embedding Model (all-MiniLM-L6-v2)...")
        EMBEDDING_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
        if TORCH_AVAILABLE and torch.cuda.is_available():
            EMBEDDING_MODEL = EMBEDDING_MODEL.to('cuda')
    return EMBEDDING_MODEL

# =============================================================================
# /health  — GET endpoint (no auth required)
# Reports which GPU/ML libraries are active so Java can surface status to UI.
# =============================================================================
def build_health_response():
    """Collect all capability flags into a JSON-serializable dict."""
    import platform
    health = {
        "status": "ok",
        "port": PORT,
        "python": platform.python_version(),
        "libraries": {
            "cuml":                       globals().get("CUML_AVAILABLE", False),
            "lightgbm":                   globals().get("LGBM_AVAILABLE", False),
            "lightgbm_gpu":               globals().get("LGBM_GPU_AVAILABLE", False),
            "xgboost":                    globals().get("XGB_AVAILABLE", False),
            "xgboost_gpu":                globals().get("XGB_GPU_AVAILABLE", False),
            "torch":                      globals().get("TORCH_AVAILABLE", False),
            "torch_cuda":                 False,
            "transformers_vit":           globals().get("VIT_AVAILABLE", False),
            "sentence_transformers":      globals().get("SENTENCE_TRANSFORMERS_AVAILABLE", False),
            "transformers_llm":           globals().get("LLM_AVAILABLE", False),
            "onnx":                       globals().get("ONNX_AVAILABLE", False),
            "pyarrow":                    globals().get("ARROW_AVAILABLE", False),
            "agent_core":                 globals().get("AGENT_AVAILABLE", False),
        },
        "models_cached": len(MODELS),
        "memory_vectors": len(VECTOR_STORE),
    }
    # Check CUDA only if torch is available (avoid NameError)
    if globals().get("TORCH_AVAILABLE", False):
        try:
            health["libraries"]["torch_cuda"] = torch.cuda.is_available()
            if health["libraries"]["torch_cuda"]:
                health["cuda_device_name"] = torch.cuda.get_device_name(0)
        except Exception:
            pass
    return health


# Global LLM State
LLM_MODEL = None
LLM_TOKENIZER = None
LLM_MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

def get_llm():
    global LLM_MODEL, LLM_TOKENIZER
    if LLM_MODEL is None and LLM_AVAILABLE:
        print(f"[Server] Loading SLM: {LLM_MODEL_ID}...")
        try:
            LLM_TOKENIZER = AutoTokenizer.from_pretrained(LLM_MODEL_ID)
            
            # Load with FP16 if GPU available for speed/memory
            if TORCH_AVAILABLE and torch.cuda.is_available():
                print("[Server] Loading LLM to GPU (FP16)...")
                LLM_MODEL = AutoModelForCausalLM.from_pretrained(
                    LLM_MODEL_ID, 
                    torch_dtype=torch.float16,
                    device_map='auto'
                )
            else:
                print("[Server] Loading LLM to CPU (FP32)...")
                LLM_MODEL = AutoModelForCausalLM.from_pretrained(LLM_MODEL_ID)
                
            LLM_MODEL.eval()
            print("[Server] SLM Loaded Successfully.")
        except Exception as e:
            print(f"[Server] Failed to load SLM: {e}")
            LLM_MODEL = None
            
    return LLM_MODEL, LLM_TOKENIZER

def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

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
        if algo_type in ['MLP', 'CNN', 'LSTM']:
             if TORCH_AVAILABLE:
                  # Input shape depends on model. MLP: (1, n_features), CNN: (1, 1, H, W) or similar
                  # We need X_sample to infer shape.
                  # LSTM needs reshaping logic too!
                  dummy_input = torch.from_numpy(X_sample[:1].astype(np.float32))
                  
                  if algo_type == 'LSTM' and hasattr(model, '_reshape_input'):
                       # The model object is the WRAPPER, not the nn.Module if we use the same pattern as MLP/LSTM scripts
                       # Wait, stored MODEL is the wrapper?
                       # Let's check: mlp_algorithm.train returns 'wrapper'.
                       # gpu_server says MODELS[uid] = model.
                       # So 'model' in this function is the Wrapper.
                       # We need to access model.model (the torch module).
                       torch_module = model.model
                       # Use wrapper's reshape logic
                       dummy_input = torch.tensor(model._reshape_input(X_sample[:1]), dtype=torch.float32).to(model.device)
                       torch.onnx.export(torch_module, dummy_input, onnx_path, verbose=False)
                       return onnx_path

                  elif hasattr(model, 'model') and isinstance(model.model, torch.nn.Module):
                       # Generic wrapper support (MLP)
                       torch.onnx.export(model.model, dummy_input, onnx_path, verbose=False)
                       return onnx_path
                  
                  # Fallback for direct modules (if any)
                  if isinstance(model, torch.nn.Module):
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

    def _authorize(self):
        """Require shared GPU token unless auth disabled or health GET."""
        if not GPU_AUTH_ENABLED:
            return True
        if not GPU_AUTH_TOKEN:
            # Misconfigured: refuse mutating work rather than open the door
            print("[Server] WARNING: GPU auth enabled but AUTOML_GPU_TOKEN is empty")
            return False
        provided = self.headers.get(GPU_AUTH_HEADER) or ''
        auth = self.headers.get('Authorization') or ''
        if auth.lower().startswith('bearer '):
            provided = auth[7:].strip()
        return provided == GPU_AUTH_TOKEN

    def do_GET(self):
        """GET endpoints: /health and / (root) — health stays open for probes."""
        if self.path == '/health':
            health = build_health_response()
            health['authRequired'] = GPU_AUTH_ENABLED
            self._send_response(200, health)
        elif self.path == '/':
            self._send_response(200, {
                "status": "ok",
                "server": "AutoML GPU Server",
                "port": PORT,
                "authRequired": GPU_AUTH_ENABLED
            })
        else:
            self._send_response(404, {"error": "Not found", "path": self.path})

    def do_POST(self):
        if not self._authorize():
            self._send_response(401, {
                'error': 'Unauthorized',
                'hint': f'Send header {GPU_AUTH_HEADER} matching AUTOML_GPU_TOKEN'
            })
            return

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
            elif action == 'MEMORY_OP':
                self.handle_memory_ops(request)
            elif action == 'CHAT':
                self.handle_chat_ops(request)
            elif action == 'EXPORT_MODEL':
                self.handle_export_model(request)
            elif action == 'EXTRACT_FEATURES':
                self.handle_extract_features(request)
            elif action == 'CLEANUP':
                self.handle_cleanup(request)
            elif action == 'AGENT_STEP':
                self.handle_agent_step(request)
            elif action == 'AGENT_FEEDBACK':
                self.handle_agent_feedback(request)
            elif action == 'GENERATE_IMAGE':
                self.handle_generate_image(request)
            elif action == 'TRAIN_WORKSPACE':
                self.handle_workspace_train(request)
            elif action == 'CANCEL_TRAINING':
                self.handle_cancel_training(request)
            elif action == 'TRAIN_MMAP':
                self.handle_train_mmap(request)
            elif action == 'UPLOAD_MMAP':
                self.handle_upload_mmap(request)
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

            self._send_response(200, {'results': results})

        except Exception as e:
            traceback.print_exc()
            self._send_response(500, {'error': f"Arrow processing failed: {str(e)}"})

    def handle_train_mmap(self, request):
        """
        Phase 9: Zero-Copy MMAP data transfer.
        Java wrote Arrow IPC to a temp file. We read it via pa.memory_map()
        which maps the file directly into our address space — zero data copying.
        The OS Page Cache means the data is already in RAM from Java's write.
        """
        if not ARROW_AVAILABLE:
            self._send_response(500, {'error': 'PyArrow not installed on server'})
            return

        try:
            mmap_path = request.get('mmap_path')
            data_type = request.get('data_type', 'TABULAR')
            algo_type = request.get('algo_type', 'RF')
            params = request.get('params', {})
            job_id = request.get('job_id', str(uuid.uuid4()))

            if not mmap_path or not os.path.exists(mmap_path):
                self._send_response(400, {'error': f'MMAP file not found: {mmap_path}'})
                return

            import time as _time
            t_start = _time.perf_counter()

            # --- ZERO-COPY READ: pa.memory_map maps file into process address space ---
            mmap_file = pa.memory_map(mmap_path, 'r')
            reader = pa.ipc.open_file(mmap_file)
            table = reader.read_all()

            t_mmap = _time.perf_counter()
            mmap_latency_us = (t_mmap - t_start) * 1_000_000

            file_size_mb = os.path.getsize(mmap_path) / (1024 * 1024)
            print(f"[Server] MMAP Zero-Copy: {file_size_mb:.1f} MB mapped in {mmap_latency_us:.0f} \u00b5s")

            # --- Extract data based on type ---
            df = table.to_pandas()

            if data_type == 'TABULAR':
                if 'target' not in df.columns:
                    self._send_response(400, {'error': 'MMAP Arrow missing "target" column'})
                    return
                y = df['target'].to_numpy()
                X = df.drop(columns=['target']).to_numpy()
            elif data_type == 'IMAGE':
                # Image paths stored in 'image_path' column
                image_paths = df['image_path'].tolist()
                y = df['target'].to_numpy() if 'target' in df.columns else None
                # For image training, delegate to CNN/ViT handler
                # X will be loaded by the image pipeline
                X = np.array(image_paths)  # Placeholder for dispatch
            elif data_type == 'TEXT':
                texts = df['text_content'].tolist()
                y = df['target'].to_numpy() if 'target' in df.columns else None
                X = np.array(texts)
            else:
                # Generic fallback
                y = df['target'].to_numpy() if 'target' in df.columns else np.zeros(len(df))
                X = df.drop(columns=['target'], errors='ignore').to_numpy()

            t_extract = _time.perf_counter()
            extract_ms = (t_extract - t_mmap) * 1000
            print(f"[Server] MMAP Data Extract: {X.shape if hasattr(X, 'shape') else len(X)} rows in {extract_ms:.1f} ms")

            # Close mmap handle
            mmap_file.close()

            # --- Train Model (reuse Arrow training logic) ---
            results = []

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
                            print(f"[Server] MMAP GPU RF Failed ({gpu_err}). CPU fallback.")
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
                            print(f"[Server] MMAP GPU SVM Failed ({gpu_err}). CPU fallback.")
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
                            except Exception:
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
                        model = lgb.LGBMClassifier(n_estimators=n_estimators, learning_rate=learning_rate, n_jobs=-1)
                        model.fit(X, y)
                    except ImportError:
                        raise Exception("LightGBM not installed")

                if model:
                    model_uid = str(uuid.uuid4())
                    MODELS[model_uid] = model
                    onnx_path = export_to_onnx(model, model_uid, X, algo_type)
                    t_total = _time.perf_counter()
                    total_ms = (t_total - t_start) * 1000
                    print(f"[Server] MMAP Train Complete: {algo_type} in {total_ms:.0f} ms (mmap={mmap_latency_us:.0f} \u00b5s)")
                    results.append({"id": job_id, "model_id": model_uid, "onnx_path": onnx_path, "success": True})
                else:
                    results.append({"id": job_id, "error": f"Unknown algo: {algo_type}", "success": False})

            except Exception as e:
                print(f"[Server] MMAP Train Failed: {e}")
                traceback.print_exc()
                results.append({"id": job_id, "error": str(e), "success": False})

            self._send_response(200, {'results': results})

        except Exception as e:
            print(f"[Server] MMAP Handler Error: {e}")
            traceback.print_exc()
            self._send_response(500, {'error': str(e)})

    def handle_upload_mmap(self, request):
        """
        Phase 9: Zero-Copy MMAP dataset upload.
        Same zero-copy mechanism as TRAIN_MMAP, but stores the dataset for
        subsequent cached training.
        """
        if not ARROW_AVAILABLE:
            self._send_response(500, {'error': 'PyArrow not installed on server'})
            return

        try:
            mmap_path = request.get('mmap_path')
            if not mmap_path or not os.path.exists(mmap_path):
                self._send_response(400, {'error': f'MMAP file not found: {mmap_path}'})
                return

            import time as _time
            t_start = _time.perf_counter()

            mmap_file = pa.memory_map(mmap_path, 'r')
            reader = pa.ipc.open_file(mmap_file)
            table = reader.read_all()
            df = table.to_pandas()

            t_mmap = _time.perf_counter()
            mmap_us = (t_mmap - t_start) * 1_000_000

            y = df['target'].to_numpy() if 'target' in df.columns else np.zeros(len(df))
            X = df.drop(columns=['target'], errors='ignore').to_numpy()

            mmap_file.close()

            ds_id = str(uuid.uuid4())
            DATASETS[ds_id] = {'X': X, 'y': y, 'location': 'mmap'}

            file_size_mb = os.path.getsize(mmap_path) / (1024 * 1024)
            print(f"[Server] MMAP Upload: {file_size_mb:.1f} MB, {X.shape[0]} rows cached as {ds_id} (mmap={mmap_us:.0f} \u00b5s)")

            self._send_response(200, {'id': ds_id, 'rows': X.shape[0], 'cols': X.shape[1]})

        except Exception as e:
            print(f"[Server] MMAP Upload Error: {e}")
            traceback.print_exc()
            self._send_response(500, {'error': str(e)})

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



    # ... (existing imports)

# ... inside handle_train_batch ...


                # --- LSTM (Recurrent Neural Network) ---
                elif algo_type == 'LSTM':
                    if TORCH_AVAILABLE:
                         try:
                             print("[Server] Training LSTM with PyTorch (GPU if available)...")
                             model = lstm_algorithm.train(X, y, params)
                         except Exception as torch_err:
                             print(f"[Server] PyTorch LSTM Failed: {torch_err}")
                             raise torch_err # No CPU Fallback for LSTM yet
                    else:
                        raise Exception("PyTorch Not Found. LSTM cannot run.")

                    print(f"[Server] LSTM trained.")

                # --- CNN (Vision) ---
                elif algo_type == 'CNN':
                    if not globals().get('CNN_AVAILABLE', False):
                        raise Exception("cnn_algorithm not available")
                    model = cnn_algorithm.train(X, y, params)
                    print("[Server] CNN trained.")

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



# ... inside handle_train_batch ...

                # --- BERT / HF Transformers (text classification fine-tune) ---
                elif algo_type in ('BERT', 'HF_TRANSFORMER'):
                    if not BERT_AVAILABLE:
                         raise Exception("BERT script not found or transformers library missing.")

                    text_paths = params.get('text_paths')
                    if text_paths:
                        print(f"[Server] Training {algo_type} on {len(text_paths)} text files...")
                        model = bert_finetune.train(text_paths, y, params)
                    elif isinstance(X, (list, np.ndarray)) and len(X) > 0 and isinstance(np.asarray(X).ravel()[0], str):
                        texts = [str(t) for t in np.asarray(X).ravel()]
                        print(f"[Server] Training {algo_type} on {len(texts)} raw texts...")
                        model = bert_finetune.train(texts, y, params)
                    else:
                        raise Exception("BERT/HF_TRANSFORMER requires 'text_paths' or string features.")

                    print(f"[Server] {algo_type} Model Trained.")

                # --- CatBoost ---
                elif algo_type == 'CATBOOST':
                    try:
                        from catboost import CatBoostClassifier
                    except ImportError:
                        results.append({"id": job_id, "error": "CatBoost not installed", "success": False})
                        continue
                    iterations = int(params.get('iterations', params.get('n_estimators', 100)))
                    depth = int(params.get('depth', params.get('max_depth', 6)))
                    learning_rate = float(params.get('learning_rate', 0.1))
                    model = CatBoostClassifier(
                        iterations=iterations,
                        depth=depth,
                        learning_rate=learning_rate,
                        verbose=False,
                        allow_writing_files=False
                    )
                    model.fit(to_cpu(X), to_cpu(y))
                    print(f"[Server] CatBoost trained. iterations={iterations}")

                # --- Custom Script (allowlisted paths only) ---
                elif algo_type == 'CUSTOM_SCRIPT':
                    script_path = params.get('script_path')
                    if not script_path or not os.path.exists(script_path):
                        raise Exception(f"Custom script not found: {script_path}")

                    # Fail closed: only allow scripts under known AutoML roots
                    abs_script = os.path.abspath(script_path)
                    allowed_roots = [
                        os.path.abspath(os.path.join(os.getcwd(), 'automl_core', 'src', 'main', 'resources', 'python')),
                        os.path.abspath(os.path.join(os.getcwd(), '..', 'automl_core', 'src', 'main', 'resources', 'python')),
                        os.path.abspath(os.path.join(os.path.expanduser('~'), '.automl', 'scripts')),
                        os.path.abspath(os.path.join(os.getcwd(), 'wsl_temp')),
                    ]
                    tmp = os.environ.get('TMPDIR') or os.environ.get('TEMP') or '/tmp'
                    allowed_roots.append(os.path.abspath(tmp))
                    if not any(abs_script.startswith(root + os.sep) or abs_script == root for root in allowed_roots):
                        raise Exception(
                            f"CUSTOM_SCRIPT blocked (path not allowlisted): {abs_script}. "
                            f"Place scripts under ~/.automl/scripts or project python resources."
                        )
                    if not abs_script.lower().endswith('.py'):
                        raise Exception("CUSTOM_SCRIPT must be a .py file")

                    try:
                        import importlib.util
                        spec = importlib.util.spec_from_file_location("custom_model_" + job_id, script_path)
                        if spec is None or spec.loader is None:
                             raise Exception(f"Failed to load spec for {script_path}")

                        module = importlib.util.module_from_spec(spec)
                        try:
                            sys.modules["custom_model_" + job_id] = module
                            spec.loader.exec_module(module)
                        except Exception as load_err:
                             raise Exception(f"Script Execution Failed: {load_err}")

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
                    # Allow CNN / path-based models to receive image_paths on the job
                    image_paths = job.get('image_paths') or (job.get('params') or {}).get('image_paths')
                    if image_paths is not None and hasattr(model, '_infer_paths'):
                        model._infer_paths = list(image_paths)
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

    def handle_extract_features(self, request):
        """
        Expects:
        {
            "action": "EXTRACT_FEATURES",
            "images": ["/path/to/img1.png", ...], 
            "model_name": "vit" | "resnet" (default resnet),
            "return_attention": boolean (default false)
        }
        Returns:
        {
            "features": [[...], ...],
            "attention_maps": [[[...]]] (optional, for ViT viz)
        }
        """
        image_paths = request.get('images', [])
        model_name = request.get('model_name', 'resnet').lower()
        return_attn = request.get('return_attention', False)
        
        results = []
        attentions = []

        global VIT_MODEL, VIT_PROCESSOR
        
        try:
            # Lazy Load ViT
            if model_name == 'vit' and VIT_AVAILABLE:
                if VIT_MODEL is None:
                    print("[Server] Loading ViT Model (first run)...")
                    VIT_PROCESSOR = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
                    VIT_MODEL = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
                    if TORCH_AVAILABLE and torch.cuda.is_available():
                        VIT_MODEL = VIT_MODEL.to('cuda')
                    VIT_MODEL.eval()
            
            # Process Images
            batch_images = []
            valid_indices = []
            
            for idx, path in enumerate(image_paths):
                if os.path.exists(path):
                    try:
                        img = Image.open(path).convert("RGB")
                        batch_images.append(img)
                        valid_indices.append(idx)
                    except Exception as e:
                        print(f"[Server] Error reading image {path}: {e}")
                else:
                    print(f"[Server] Image not found: {path}")

            if not batch_images:
                 self._send_response(200, {"features": [], "error": "No valid images found"})
                 return

            if model_name == 'vit' and VIT_AVAILABLE:
                # ViT Extraction
                inputs = VIT_PROCESSOR(images=batch_images, return_tensors="pt")
                if TORCH_AVAILABLE and torch.cuda.is_available():
                    inputs = inputs.to('cuda')

                with torch.no_grad():
                    outputs = VIT_MODEL(**inputs, output_attentions=return_attn)
                
                # Pooler Output (CLS token) for features: [Batch, 768]
                features = outputs.pooler_output.cpu().numpy().tolist()
                
                # Attention Maps for Viz: Tuple of [Batch, Heads, Seq, Seq]
                if return_attn and outputs.attentions:
                    # Get last layer attention: [Batch, Heads, Seq, Seq]
                    last_layer_attn = outputs.attentions[-1].cpu().numpy()
                    # We might want to average inputs or just return raw for Java to process
                    # It's huge, so let's just return the last layer
                    attentions = last_layer_attn.tolist()
                
                self._send_response(200, {"features": features, "attention_maps": attentions})

            else:
                # Fallback to ResNet (via encoded logic or placeholder)
                # Since we don't have a ResNet loaded explicitly here in the snippet,
                # we'll use a simple placeholder or try to load torchvision if available.
                # For now, let's claim not implemented if not ViT, or add ResNet logic.
                if TORCH_AVAILABLE:
                     import torchvision.models as models
                     import torchvision.transforms as transforms
                     # ... (ResNet logic would go here, omitting for brevity/focus on ViT)
                     self._send_response(500, {"error": "ResNet not initialized. Use 'vit'."})
                else:
                     self._send_response(500, {"error": "Deep Learning (Torch) not available."})

        except Exception as e:
            traceback.print_exc()
            self._send_response(500, {'error': str(e)})

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


    def handle_memory_ops(self, request):
        sub_action = request.get('op', 'EMBED')
        
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            self._send_response(500, {"error": "Sentence Transformers not available."})
            return

        model = get_embedding_model()
        if model is None:
             self._send_response(500, {"error": "Failed to load Embedding Model."})
             return

        try:
            if sub_action == 'EMBED':
                text = request.get('text', '')
                if not text:
                    self._send_response(400, {"error": "No text provided"})
                    return
                embedding = model.encode(text).tolist()
                self._send_response(200, {"embedding": embedding})

            elif sub_action == 'STORE':
                text = request.get('text', '')
                metadata = request.get('metadata', {})
                if not text:
                    self._send_response(400, {"error": "No text provided"})
                    return
                
                vector = model.encode(text)
                VECTOR_STORE.append({
                    'text': text,
                    'vector': vector, 
                    'metadata': metadata,
                    'timestamp': time.time()
                })
                self._send_response(200, {"success": True, "count": len(VECTOR_STORE)})

            elif sub_action == 'QUERY':
                query = request.get('query', '')
                top_k = request.get('top_k', 5)
                if not query:
                    self._send_response(400, {"error": "No query provided"})
                    return
                
                if not VECTOR_STORE:
                    self._send_response(200, {"matches": []})
                    return

                query_vec = model.encode(query)
                results = []
                for item in VECTOR_STORE:
                    score = cosine_similarity(query_vec, item['vector'])
                    results.append({
                        'text': item['text'],
                        'metadata': item['metadata'],
                        'score': float(score),
                        'timestamp': item.get('timestamp', 0)
                    })
                
                results.sort(key=lambda x: x['score'], reverse=True)
                self._send_response(200, {"matches": results[:top_k]})

            else:
                 self._send_response(400, {"error": f"Unknown memory operation: {sub_action}"})

        except Exception as e:
            traceback.print_exc()
            self._send_response(500, {'error': str(e)})

    def handle_chat_ops(self, request):
        if not LLM_AVAILABLE:
            self._send_response(500, {"error": "LLM libraries not installed."})
            return

        model, tokenizer = get_llm()
        if model is None:
            self._send_response(500, {"error": "Failed to load LLM."})
            return

        try:
            system_prompt = request.get('system', "You are a helpful AI assistant.")
            user_msg = request.get('user', '')
            max_new_tokens = request.get('max_tokens', 256)
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg}
            ]
            
            try:
                input_ids = tokenizer.apply_chat_template(
                    messages, 
                    return_tensors="pt", 
                    add_generation_prompt=True
                )
            except:
                prompt = f"<|system|>\n{system_prompt}</s>\n<|user|>\n{user_msg}</s>\n<|assistant|>\n"
                input_ids = tokenizer.encode(prompt, return_tensors="pt")

            if TORCH_AVAILABLE and torch.cuda.is_available():
                input_ids = input_ids.to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    input_ids, 
                    max_new_tokens=max_new_tokens, 
                    do_sample=True, 
                    temperature=0.7,
                    top_k=50
                )
            
            input_len = input_ids.shape[1]
            new_tokens = outputs[0][input_len:]
            response_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
            
            self._send_response(200, {"response": response_text.strip()})

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

    def handle_cancel_training(self, request):
        TRAINING_CANCEL.set()
        print("[Server] CANCEL_TRAINING received.")
        self._send_response(200, {'success': True, 'cancelled': True})

    def handle_workspace_train(self, request):
        """
        Flat API used by DL/CNN/LLM JavaFX workspaces.
        Loads data from dataset_path then trains with UI hyperparams.
        """
        TRAINING_CANCEL.clear()
        try:
            algo = (request.get('algo') or 'MLP').upper()
            dataset_path = request.get('dataset_path') or ''
            job_id = request.get('job_id') or str(uuid.uuid4())

            def cancelled():
                return TRAINING_CANCEL.is_set()

            params = {
                'hidden_layer_sizes': request.get('layers') or request.get('hidden_layer_sizes') or '128,64',
                'layers': request.get('layers') or '128,64',
                'activation': request.get('activation') or 'relu',
                'optimizer': request.get('optimizer') or 'Adam',
                'solver': request.get('optimizer') or 'adam',
                'learning_rate': float(request.get('learning_rate') or 0.001),
                'epochs': int(request.get('epochs') or 20),
                'max_iter': int(request.get('epochs') or 20),
                'batch_size': int(request.get('batch_size') or 32),
                'use_gpu': bool(request.get('use_gpu', True)),
                'use_wsl': bool(request.get('use_wsl', False)),
                'fp16': bool(request.get('fp16', False)),
                'backbone': request.get('backbone'),
                'strategy': request.get('strategy'),
                'image_size': request.get('image_size') or '224x224',
                'augment_flip': bool(request.get('augment_flip', False)),
                'augment_rotate': bool(request.get('augment_rotate', False)),
                'augment_crop': bool(request.get('augment_crop', False)),
                'tuning_mode': request.get('tuning_mode'),
                'quantize_8bit': bool(request.get('quantize_8bit', False)),
                'quantize_4bit': bool(request.get('quantize_4bit', False)),
                'model': request.get('model'),
                'hf_model_id': request.get('hf_model_id') or request.get('model'),
                'window_size': int(request.get('window_size') or 1),
                'cancel_check': cancelled,
            }

            note = None
            if algo == 'AUTOENCODER':
                allow_preview = bool(request.get('allow_preview', False))
                if not allow_preview:
                    self._send_response(400, {
                        'success': False,
                        'error': 'Autoencoder is not implemented. Pass allow_preview=true to train an MLP classifier instead (preview mode), or choose MLP/LSTM.'
                    })
                    return
                algo = 'MLP'
                note = 'Autoencoder preview mode: mapped to MLP classifier (not a real autoencoder)'

            model = None
            X = y = None

            if algo in ('MLP', 'LSTM'):
                if not dataset_path or not os.path.exists(dataset_path):
                    self._send_response(400, {'success': False, 'error': f'dataset_path not found: {dataset_path}'})
                    return
                df = pd.read_csv(dataset_path)
                target = request.get('target_column')
                if not target or target not in df.columns:
                    target = df.columns[-1]
                y_raw = df[target]
                X_df = df.drop(columns=[target]).select_dtypes(include=[np.number])
                if X_df.shape[1] == 0:
                    self._send_response(400, {'success': False, 'error': 'No numeric feature columns found'})
                    return
                X = X_df.to_numpy(dtype=float)
                # Encode labels to ints
                classes, y = np.unique(y_raw.astype(str), return_inverse=True)

                if cancelled():
                    self._send_response(200, {'success': False, 'cancelled': True, 'job_id': job_id})
                    return

                if algo == 'MLP':
                    if TORCH_AVAILABLE:
                        model = mlp_algorithm.train(X, y, params)
                    else:
                        layers_str = str(params['hidden_layer_sizes'])
                        try:
                            hidden = tuple(map(int, layers_str.split(',')))
                        except Exception:
                            hidden = (100,)
                        act = str(params['activation']).lower()
                        if act == 'sigmoid':
                            act = 'logistic'
                        if act not in ('relu', 'tanh', 'logistic', 'identity'):
                            act = 'relu'
                        # Sklearn fit is not interruptible mid-pass; check cancel before/after.
                        if cancelled():
                            self._send_response(200, {'success': False, 'cancelled': True, 'job_id': job_id})
                            return
                        max_iter = max(1, int(params['epochs']))
                        model = MLPClassifier(
                            hidden_layer_sizes=hidden,
                            activation=act,
                            max_iter=1,
                            warm_start=True
                        )
                        for _ in range(max_iter):
                            if cancelled():
                                self._send_response(200, {'success': False, 'cancelled': True, 'job_id': job_id})
                                return
                            model.fit(X, y)
                else:  # LSTM
                    if not TORCH_AVAILABLE:
                        raise Exception('PyTorch required for LSTM')
                    # Ensure feature count divisible by window_size
                    ws = max(1, int(params['window_size']))
                    if X.shape[1] % ws != 0:
                        pad = ws - (X.shape[1] % ws)
                        X = np.pad(X, ((0, 0), (0, pad)), mode='constant')
                        params['window_size'] = ws
                    model = lstm_algorithm.train(X, y, params)

            elif algo == 'CNN':
                # Fail closed on unimplemented backbones unless allow_preview
                backbone = str(request.get('backbone') or params.get('backbone') or '').strip()
                bb_u = backbone.upper()
                simple = (not backbone) or any(x in bb_u for x in (
                    'CUSTOM', 'SIMPLE', '3 BLOCK', 'SIMPLECNN'
                ))
                allow_preview = bool(request.get('allow_preview', False))
                if not simple and not allow_preview:
                    self._send_response(400, {
                        'success': False,
                        'error': (
                            f"Backbone '{backbone}' is not implemented. "
                            "Only SimpleCNN/Custom CNN (3 Block) is trained today. "
                            "Select Custom CNN, or pass allow_preview=true to train SimpleCNN anyway."
                        )
                    })
                    return
                if not simple and allow_preview:
                    note = (note + '; ' if note else '') + f"Preview: '{backbone}' remapped to SimpleCNN @ 64x64"
                if not globals().get('CNN_AVAILABLE', False):
                    raise Exception('cnn_algorithm not available')
                if not dataset_path or not os.path.isdir(dataset_path):
                    self._send_response(400, {
                        'success': False,
                        'error': f'CNN expects an image folder (class subdirs): {dataset_path}'
                    })
                    return
                paths, labels, class_names = [], [], []
                for cls_name in sorted(os.listdir(dataset_path)):
                    cls_dir = os.path.join(dataset_path, cls_name)
                    if not os.path.isdir(cls_dir):
                        continue
                    class_names.append(cls_name)
                    cls_idx = len(class_names) - 1
                    for fn in os.listdir(cls_dir):
                        if fn.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp', '.gif')):
                            paths.append(os.path.join(cls_dir, fn))
                            labels.append(cls_idx)
                if not paths:
                    self._send_response(400, {'success': False, 'error': 'No images found under class subdirectories'})
                    return
                params['image_paths'] = paths
                y = np.array(labels, dtype=int)
                X = np.zeros((len(paths), 1))  # placeholder features for export helpers
                model = cnn_algorithm.train(X, y, params)

            elif algo == 'LLM_FINETUNE':
                if not globals().get('BERT_AVAILABLE', False):
                    raise Exception('bert_finetune / transformers not available')
                if not dataset_path or not os.path.exists(dataset_path):
                    self._send_response(400, {'success': False, 'error': f'training corpus not found: {dataset_path}'})
                    return

                texts, labels = [], []
                if dataset_path.lower().endswith('.jsonl'):
                    with open(dataset_path, 'r', encoding='utf-8', errors='ignore') as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            row = json.loads(line)
                            texts.append(str(row.get('text') or row.get('prompt') or row.get('content') or ''))
                            labels.append(int(row.get('label', 0)))
                else:
                    # Plain text without labels — fail closed unless allow_preview
                    allow_preview = bool(request.get('allow_preview', False))
                    if not allow_preview:
                        self._send_response(400, {
                            'success': False,
                            'error': (
                                'LLM fine-tune requires JSONL with {"text","label"} per line. '
                                'Plain text would use synthetic i%2 labels (misleading). '
                                'Pass allow_preview=true only for smoke tests.'
                            )
                        })
                        return
                    with open(dataset_path, 'r', encoding='utf-8', errors='ignore') as f:
                        for i, line in enumerate(f):
                            t = line.strip()
                            if t:
                                texts.append(t)
                                labels.append(i % 2)  # preview-only synthetic labels
                    note = (note + '; ' if note else '') + 'Preview: synthetic i%2 labels on plain text'
                if len(texts) < 2:
                    self._send_response(400, {'success': False, 'error': 'Need at least 2 text samples for fine-tune'})
                    return
                if cancelled():
                    self._send_response(200, {'success': False, 'cancelled': True, 'job_id': job_id})
                    return
                y = np.array(labels, dtype=int)
                model = bert_finetune.train(texts, y, params)
                X = np.array(texts, dtype=object)

            else:
                self._send_response(400, {'success': False, 'error': f'Unsupported workspace algo: {algo}'})
                return

            if cancelled():
                self._send_response(200, {'success': False, 'cancelled': True, 'job_id': job_id})
                return

            model_uid = str(uuid.uuid4())
            MODELS[model_uid] = model
            onnx_path = None
            try:
                if algo not in ('LLM_FINETUNE', 'CNN'):
                    onnx_path = export_to_onnx(model, model_uid, X, algo)
            except Exception as ex:
                print(f"[Server] Workspace ONNX export skipped: {ex}")

            self._send_response(200, {
                'success': True,
                'job_id': job_id,
                'algo': algo,
                'model_id': model_uid,
                'onnx_path': onnx_path,
                'note': note,
                'samples': int(len(y)) if y is not None else 0
            })
        except InterruptedError:
            self._send_response(200, {'success': False, 'cancelled': True})
        except Exception as e:
            traceback.print_exc()
            self._send_response(500, {'success': False, 'error': str(e)})

    def handle_generate_image(self, request):
        """
        GenAI workspace worker.
        Expects: prompt, model, task, width, height, steps, cfg, use_gpu, hf_model_id (optional),
                 allow_preview (optional — enables procedural / unsupported-task fallbacks)
        Returns: {success, path, engine, message} or error
        """
        try:
            prompt = (request.get('prompt') or '').strip()
            if not prompt:
                self._send_response(400, {'success': False, 'error': 'prompt is required'})
                return

            task = request.get('task') or 'Text-to-Image'
            model_name = request.get('model') or 'Stable Diffusion v1.5'
            hf_model = (request.get('hf_model_id') or '').strip()
            width = int(request.get('width') or 512)
            height = int(request.get('height') or 512)
            steps = int(request.get('steps') or 20)
            cfg = float(request.get('cfg') or 7.5)
            use_gpu = bool(request.get('use_gpu', True))
            allow_preview = bool(request.get('allow_preview', False))
            warnings = []

            task_l = str(task).lower()
            unsupported_task = ('inpaint' in task_l or 'image-to-image' in task_l or 'img2img' in task_l)
            if unsupported_task:
                if not allow_preview:
                    self._send_response(400, {
                        'success': False,
                        'error': f'{task} is not implemented yet. Use Text-to-Image, or pass allow_preview=true to run Text-to-Image as a stand-in.'
                    })
                    return
                warnings.append(f'{task} is not fully supported; running Text-to-Image (preview mode).')

            out_dir = os.path.join(os.getcwd(), 'genai_outputs')
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"gen_{uuid.uuid4().hex[:12]}.png")

            engine = 'diffusers'
            message = ''
            is_fallback = False

            try:
                import torch
                from PIL import Image as _PILImage  # noqa: F401

                device = 'cuda' if (use_gpu and torch.cuda.is_available()) else 'cpu'
                model_u = model_name.upper()
                if 'STYLEGAN' in model_u and not hf_model:
                    raise RuntimeError('StyleGAN2 requires an explicit HuggingFace model id')

                use_xl = ('XL' in model_u) or ('sdxl' in hf_model.lower())
                if use_xl:
                    from diffusers import StableDiffusionXLPipeline as PipeCls
                    repo = hf_model or 'stabilityai/stable-diffusion-xl-base-1.0'
                else:
                    from diffusers import StableDiffusionPipeline as PipeCls
                    repo = hf_model or 'runwayml/stable-diffusion-v1-5'

                pipe = PipeCls.from_pretrained(
                    repo,
                    torch_dtype=torch.float16 if device == 'cuda' else torch.float32
                    # safety_checker left at Diffusers default (do not disable)
                )
                pipe = pipe.to(device)
                result = pipe(
                    prompt,
                    num_inference_steps=max(1, min(steps, 50)),
                    guidance_scale=cfg,
                    width=width,
                    height=height
                )
                image = result.images[0]
                image.save(out_path)
                engine = f'diffusers:{repo}@{device}'
                message = f'{task} completed via Diffusers ({model_name}).'
                is_fallback = False
            except Exception as diff_err:
                if not allow_preview:
                    self._send_response(400, {
                        'success': False,
                        'error': f'Diffusers generation failed: {diff_err}. Install diffusers+torch, or pass allow_preview=true for a procedural test image.',
                        'fallback': True
                    })
                    return
                # Procedural preview only when explicitly allowed
                try:
                    from PIL import Image, ImageDraw
                    import hashlib

                    digest = hashlib.md5(prompt.encode('utf-8')).hexdigest()
                    seed_rgb = tuple(int(digest[i:i+2], 16) for i in (0, 2, 4))
                    img = Image.new('RGB', (width, height), seed_rgb)
                    draw = ImageDraw.Draw(img)
                    for y in range(0, height, 8):
                        shade = tuple(min(255, c + (y % 64)) for c in seed_rgb)
                        draw.line([(0, y), (width, y)], fill=shade, width=8)
                    text = prompt if len(prompt) <= 80 else prompt[:77] + '...'
                    draw.rectangle([12, 12, width - 12, 110], fill=(20, 20, 24))
                    draw.text((24, 28), f'{model_name}', fill=(120, 200, 255))
                    draw.text((24, 56), text, fill=(230, 230, 230))
                    draw.text((24, height - 36), f'preview: {diff_err}', fill=(180, 180, 180))
                    img.save(out_path)
                    engine = 'pillow_procedural'
                    message = f'Procedural preview only (Diffusers unavailable: {diff_err})'
                    warnings.append(message)
                    is_fallback = True
                except Exception as pil_err:
                    traceback.print_exc()
                    self._send_response(500, {
                        'success': False,
                        'error': f'GenAI failed: diffusers={diff_err}; pillow={pil_err}'
                    })
                    return

            self._send_response(200, {
                'success': True,
                'path': out_path,
                'engine': engine,
                'message': message,
                'task': task,
                'prompt': prompt,
                'fallback': is_fallback,
                'warnings': warnings
            })
        except Exception as e:
            traceback.print_exc()
            self._send_response(500, {'success': False, 'error': str(e)})

    def handle_agent_step(self, request):
        """LCA Fast Loop: Java Perception snapshot -> Python FastActor action."""
        if not AGENT_AVAILABLE:
            self._send_response(503, {
                'success': False,
                'error': 'agent_core not available on GPU server'
            })
            return
        try:
            snapshot = request.get('snapshot') or {}
            action = agent_core.handle_step(snapshot)
            if action is None:
                action = {
                    'action': 'WAIT',
                    'confidence': 0.0,
                    'q_value': 0.0,
                    'layer': 'NONE',
                    'target_node': None
                }
            self._send_response(200, {'success': True, 'action': action})
        except Exception as e:
            traceback.print_exc()
            self._send_response(500, {'success': False, 'error': str(e)})

    def handle_agent_feedback(self, request):
        """LCA Slow Loop: reward feedback into SlowLearner memory."""
        if not AGENT_AVAILABLE:
            self._send_response(503, {
                'success': False,
                'error': 'agent_core not available on GPU server'
            })
            return
        try:
            agent_core.LEARNER.add_experience(
                request.get('context'),
                request.get('action'),
                float(request.get('reward', 0) or 0),
                request.get('next_context')
            )
            self._send_response(200, {'success': True})
        except Exception as e:
            traceback.print_exc()
            self._send_response(500, {'success': False, 'error': str(e)})


class ReusableTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    """Threaded so CANCEL_TRAINING / health can run while TRAIN_WORKSPACE blocks."""
    allow_reuse_address = True
    daemon_threads = True


def run_server():
    # Localhost-only by default (set AUTOML_GPU_BIND=0.0.0.0 only if you accept LAN exposure).
    bind_host = os.environ.get('AUTOML_GPU_BIND', '127.0.0.1').strip() or '127.0.0.1'
    if GPU_AUTH_ENABLED:
        print(f"[Server] GPU token auth ENABLED (header {GPU_AUTH_HEADER})")
    else:
        print("[Server] GPU token auth DISABLED (AUTOML_GPU_AUTH=false)")
    server_address = (bind_host, PORT)
    httpd = None
    for i in range(10):
        try:
            httpd = ReusableTCPServer(server_address, GpuRequestHandler)
            print(f"[Server] Starting threaded GPU Server on {bind_host}:{PORT}...")
            if bind_host not in ('127.0.0.1', 'localhost', '::1'):
                print("[Server] WARNING: GPU server is bound beyond localhost — no auth is enforced.")
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
