import numpy as np
import os
import json
import uuid
import time
import math

# --- Third Party Imports (Graceful Degradation) ---
try:
    import torch
    import torch.nn as nn
    import torchvision.models as models
    import torchvision.transforms as transforms
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("[AgentCore] PyTorch not found. Vision features will be random noise.")

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("[AgentCore] FAISS not found. Using slow Numpy fallback for VectorStore.")

# --- 1. Vector Store (Abstration) ---
class VectorStore:
    def __init__(self, dimension=256):
        self.dimension = dimension
        self.vectors = np.empty((0, dimension), dtype='float32') # FAISS likes float32
        self.metadata = [] # List of dicts corresponding to vectors
        
        if FAISS_AVAILABLE:
            self.index = faiss.IndexFlatL2(dimension)
        else:
            self.index = None

    def add_or_update(self, vector, meta_dict, error_weight=None):
        """
        Adds vector. If error_weight is provided, it implies a learning update.
        For simplicity in this prototype, we just APPEND new experiences.
        A real system would merge/update near-neighbors.
        """
        vector = vector.astype('float32').reshape(1, -1)
        
        # FAISS add
        if self.index:
            self.index.add(vector)
        
        # Numpy backing (always needed for metadata sync)
        self.vectors = np.vstack([self.vectors, vector])
        self.metadata.append(meta_dict)

    def search(self, vector, k=1):
        """Returns (score, metadata)"""
        if self.vectors.shape[0] == 0:
            return 0.0, None

        vector = vector.astype('float32').reshape(1, -1)
        
        if self.index:
            # FAISS returns L2 distance. Lower is better. 
            # We want "Confidence" or "Score" (0..1).
            # Convert L2 to Sim: 1 / (1 + Dist)
            D, I = self.index.search(vector, k)
            idx = I[0][0]
            dist = D[0][0]
            if idx == -1: return 0.0, None
            
            score = 1.0 / (1.0 + dist)
            return score, self.metadata[idx]
        else:
            # Numpy Brute Force
            # Cosine Sim for compatibility with user equations
            norm_v = np.linalg.norm(vector)
            norms = np.linalg.norm(self.vectors, axis=1)
            dot_prods = np.dot(self.vectors, vector.T).flatten()
            sims = dot_prods / (norms * norm_v + 1e-9)
            
            best_idx = np.argmax(sims)
            return sims[best_idx], self.metadata[best_idx]

# --- 2. Hierarchical Memory ---
class FourLayerMemory:
    def __init__(self):
        # Layer 0: Hardcoded O(1) lookups (The "Instincts")
        # Maps perceptual_hash -> (ActionID, Confidence)
        self.os_axioms = {
            "close_btn_hash": ("CLICK_CLOSE", 1.0), 
            "minimize_btn_hash": ("CLICK_MIN", 1.0)
        }
        
        # Layer 1 & 2: Fast Vector Databases
        self.global_db = VectorStore(dimension=256) 
        # Domains: "Browser", "IDE", "Terminal", "General"
        self.domain_dbs = {
            "browser": VectorStore(dimension=256),
            "ide": VectorStore(dimension=256),
            "terminal": VectorStore(dimension=256),
            "general": VectorStore(dimension=256)
        }
        
        # Layer 3: App-specific overrides (Hash Map for O(1) retrieval)
        # Format: {app_id: {feature_hash: (ActionID, Confidence)}}
        self.app_overrides = {} 

    def get_affordance(self, feature_vector, feature_hash, app_id, domain_id):
        """
        Cascading fast-lookup.
        Returns: (ActionID, Confidence, LayerUsed)
        """
        # 1. OS Override (Instant)
        if feature_hash in self.os_axioms:
            act, conf = self.os_axioms[feature_hash]
            return act, conf, "OS"

        # 2. App Override (Instant)
        if app_id in self.app_overrides and feature_hash in self.app_overrides[app_id]:
            act, conf = self.app_overrides[app_id][feature_hash]
            return act, conf, "APP"

        # 3. Domain Knowledge (Vector Search)
        target_domain_db = self.domain_dbs.get(domain_id, self.domain_dbs["general"])
        dom_conf, dom_meta = target_domain_db.search(feature_vector)
        
        # User Equation: w2 * M_Dom(f|c)
        # We assume w2 = 1.0 for now.
        if dom_conf > 0.85: # High confidence domain match
            return dom_meta['action'], dom_conf, "DOMAIN"

        # 4. Global Fallback (Vector Search)
        glob_conf, glob_meta = self.global_db.search(feature_vector)
        if glob_meta:
            return glob_meta['action'], glob_conf, "GLOBAL"
            
        return "WAIT", 0.0, "NONE"

# --- 3. Fast Actor (System 1) ---
class FastActor:
    def __init__(self, memory_module):
        self.memory = memory_module
        self.encoder = None
        
        if TORCH_AVAILABLE:
            # Load lightweight MobileNetV3 Small
            try:
                # We use default verification. In offline/air-gapped, this needs pre-downloaded weights.
                # For prototype, we initialize random if download fails or use simple structure.
                self.encoder = models.mobilenet_v3_small(pretrained=True)
                self.encoder.classifier = nn.Identity() # Remove classification head, keep features (1024 dim? No, 576)
                # Helper to project to 256
                self.projector = nn.Linear(576, 256) 
                
                self.encoder.eval()
                # self.encoder.to('cuda') # Keep on CPU for simplicity unless we have robust GPU checks again
            except:
                print("[AgentCore] MobileNet download failed. Using random encoder.")
                self.encoder = nn.Sequential(
                    nn.AdaptiveAvgPool2d((1,1)),
                    nn.Flatten(),
                    nn.Linear(3, 256) # Expecting 3 channel input
                )
        
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]) if TORCH_AVAILABLE else None

    def encode(self, image_np):
        """image_np: (H, W, 3) uint8 numpy array"""
        if image_np is None or not TORCH_AVAILABLE or self.transform is None or self.encoder is None:
            return np.random.rand(256).astype('float32')

        try:
            with torch.no_grad():
                img_t = self.transform(image_np)
                img_t = img_t.unsqueeze(0)  # Batch dim
                feats = self.encoder(img_t)  # (1, 576) for MobileNet
                if hasattr(self, 'projector'):
                    feats = self.projector(feats)  # (1, 256)
                return feats.cpu().numpy().flatten().astype('float32')
        except Exception as e:
            print(f"[AgentCore] encode failed, using random features: {e}")
            return np.random.rand(256).astype('float32')

    def _decode_screenshot(self, ui_graph_snapshot):
        """Decode Java's screenshot_base64 into HxWx3 uint8 if present."""
        b64 = ui_graph_snapshot.get('screenshot_base64')
        if not b64:
            return None
        try:
            import base64
            from io import BytesIO
            raw = base64.b64decode(b64)
            if TORCH_AVAILABLE:
                from PIL import Image
                img = Image.open(BytesIO(raw)).convert('RGB')
                return np.asarray(img, dtype=np.uint8)
            # Minimal JPEG/PNG fallback without PIL: unavailable
            return None
        except Exception as e:
            print(f"[AgentCore] Screenshot decode failed: {e}")
            return None

    def step(self, ui_graph_snapshot):
        """
        ui_graph_snapshot: Dict containing:
          - 'screenshot_base64': JPG/PNG from Java PerceptionService
          - 'nodes': List of { 'bbox': [x,y,w,h], 'text': str, ... }
          - 'context': { 'id': str, 'domain': str }
        """
        app_ctx = ui_graph_snapshot.get('context', {'id': 'unknown', 'domain': 'general'})
        if not isinstance(app_ctx, dict):
            app_ctx = {'id': 'unknown', 'domain': 'general'}
        app_ctx.setdefault('id', 'unknown')
        app_ctx.setdefault('domain', 'general')

        nodes = ui_graph_snapshot.get('nodes') or []
        screen_np = self._decode_screenshot(ui_graph_snapshot)
        global_vec = self.encode(screen_np) if screen_np is not None else np.random.rand(256).astype('float32')

        action_scores = []

        # Analyze each UI Node as a potential target
        for node in nodes:
            # Prefer crop-encode when we have a full screenshot + bbox
            f_vec = global_vec
            bbox = node.get('bbox')
            if screen_np is not None and isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
                try:
                    x, y, w, h = [int(v) for v in bbox[:4]]
                    H, W = screen_np.shape[:2]
                    x2, y2 = min(W, x + max(w, 1)), min(H, y + max(h, 1))
                    x, y = max(0, x), max(0, y)
                    if x2 > x and y2 > y:
                        crop = screen_np[y:y2, x:x2]
                        f_vec = self.encode(crop)
                except Exception:
                    f_vec = global_vec

            f_hash = str(node.get('hash', hash(str(node.get('text', '')))))

            act, conf, layer = self.memory.get_affordance(
                 f_vec, f_hash, app_ctx['id'], app_ctx['domain']
            )

            curiosity = 1.0 - conf
            q_val = conf + (0.2 * curiosity)

            action_scores.append({
                'action': act,
                'target_node': bbox,
                'confidence': float(conf),
                'q_value': float(q_val),
                'layer': layer
            })

        # Visual-only fallback when OCR produced no nodes
        if not action_scores:
            act, conf, layer = self.memory.get_affordance(
                global_vec, 'full_frame', app_ctx['id'], app_ctx['domain']
            )
            width = int(ui_graph_snapshot.get('width') or 100)
            height = int(ui_graph_snapshot.get('height') or 100)
            # Conservative default: wait unless memory has a strong prior
            if conf < 0.5:
                return {
                    'action': 'WAIT',
                    'target_node': None,
                    'confidence': float(conf),
                    'q_value': float(conf),
                    'layer': layer or 'NONE'
                }
            return {
                'action': act if act != 'WAIT' else 'CLICK',
                'target_node': [width // 4, height // 4, width // 2, height // 2],
                'confidence': float(conf),
                'q_value': float(conf + 0.2 * (1.0 - conf)),
                'layer': layer
            }

        action_scores.sort(key=lambda x: x['q_value'], reverse=True)
        return action_scores[0]

# --- 4. Slow Learner (System 2) ---
class SlowLearner:
    def __init__(self, memory_module):
        self.memory = memory_module
        self.replay_buffer = [] # List of transitions
        
    def add_experience(self, context, action, reward, next_context):
        """
        context: state before
        action: what we did
        reward: -1 (fail) to 1 (success)
        next_context: state after
        """
        self.replay_buffer.append((context, action, reward, next_context))
        
        if len(self.replay_buffer) > 10:
            self.process_buffer()

    def process_buffer(self):
        """
        Updates memory based on simple RL rule:
        If Reward > 0, Reinforce (App/Domain level).
        If Reward < 0, Suppress (low-confidence wait prior).
        """
        print(f"[SlowLearner] Processing {len(self.replay_buffer)} learnings...")
        for ctx, act, r, n_ctx in self.replay_buffer:
            if r == 0 or not isinstance(act, dict):
                continue

            # Prefer an explicit feature vector; otherwise store a deterministic prior vector
            f_vec = None
            if isinstance(ctx, dict) and ctx.get('feature_vector') is not None:
                f_vec = np.asarray(ctx['feature_vector'], dtype='float32')
            else:
                seed = abs(hash(str(act.get('action')))) % (2**32)
                rng = np.random.RandomState(seed)
                f_vec = rng.rand(256).astype('float32')

            meta = {
                'action': act.get('action', 'WAIT'),
                'target_node': act.get('target_node'),
                'reward': float(r)
            }
            domain = 'general'
            if isinstance(ctx, dict):
                c = ctx.get('context') or {}
                if isinstance(c, dict):
                    domain = c.get('domain', 'general') or 'general'

            if r > 0:
                self.memory.global_db.add_or_update(f_vec, meta)
                dom_db = self.memory.domain_dbs.get(domain, self.memory.domain_dbs['general'])
                dom_db.add_or_update(f_vec, meta)
            else:
                # Suppress: record WAIT so similar states become less clicky
                self.memory.global_db.add_or_update(f_vec, {'action': 'WAIT', 'reward': float(r)})

        self.replay_buffer.clear()

# --- Singleton Export ---
MEMORY = FourLayerMemory()
ACTOR = FastActor(MEMORY)
LEARNER = SlowLearner(MEMORY)

def handle_step(snapshot_json):
    """Main entry point for 'Fast Loop'"""
    # snapshot_json comes from Java
    # returns Action JSON
    best_action = ACTOR.step(snapshot_json)
    return best_action
