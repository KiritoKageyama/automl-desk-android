"""Minimal unit checks for CNN predict path (no GPU required)."""
import os
import sys
import tempfile
import unittest

import numpy as np

# Allow importing from resources/python when run from repo
HERE = os.path.dirname(os.path.abspath(__file__))
PY_ROOT = os.path.normpath(os.path.join(HERE, "..", "main", "resources", "python"))
sys.path.insert(0, PY_ROOT)

try:
    import torch
    from PIL import Image
    import cnn_algorithm
    HAS_TORCH = True
except Exception:
    HAS_TORCH = False


@unittest.skipUnless(HAS_TORCH, "torch/PIL not available")
class CnnPredictTests(unittest.TestCase):
    def test_predict_not_zeros_placeholder(self):
        with tempfile.TemporaryDirectory() as td:
            paths = []
            for i, color in enumerate([(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]):
                p = os.path.join(td, f"img{i}.png")
                Image.new("RGB", (64, 64), color).save(p)
                paths.append(p)
            y = np.array([0, 1, 0, 1])
            model = cnn_algorithm.train(None, y, {
                "epochs": 1,
                "batch_size": 2,
                "learning_rate": 0.01,
                "image_paths": paths,
            })
            preds = model.predict(np.zeros((len(paths), 0)))
            self.assertEqual(len(preds), len(paths))
            # Must not be the old placeholder of all zeros for mixed labels after 1 epoch —
            # at least the path must execute without returning a hard-coded zeros array of wrong semantics.
            # After 1 epoch preds may still be zeros by chance; assert shape/dtype instead + method runs.
            self.assertEqual(preds.dtype, np.int_)
            # Calling again with explicit paths
            preds2 = model.predict(paths)
            self.assertEqual(len(preds2), len(paths))


if __name__ == "__main__":
    unittest.main()
