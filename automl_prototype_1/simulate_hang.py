import requests
import json
import numpy as np

# Simulate a simplified dataset upload (similar size to WDBC)
rows = 569
cols = 31 # 30 features + 1 label
data = np.random.rand(rows, cols).tolist()
features = [r[:-1] for r in data]
labels = [int(r[-1]>0.5) for r in data]

payload = {
    "action": "UPLOAD_DATASET",
    "jobs": [
        {
            "id": "test_job_1",
            "type": "UPLOAD",
            "features": features,
            "labels": labels
        }
    ],
    "upload_features": features,
    "upload_labels": labels
}

json_str = json.dumps(payload)
print(f"Payload Size: {len(json_str)} bytes")

try:
    print("Sending Request...")
    resp = requests.post("http://127.0.0.1:5006", data=json_str, headers={"Content-Type": "application/json"}, timeout=10)
    print(f"Status: {resp.status_code}")
    print(f"Response: {resp.text}")
except Exception as e:
    print(f"Error: {e}")
