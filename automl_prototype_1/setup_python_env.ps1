# AutoML — Python Environment Setup Script
# Run this script to install all Python dependencies
# Requires: Python 3.10+ (pip)

Write-Host "=== AutoML Python Environment Setup ===" -ForegroundColor Cyan

# Paths
$projectRoot = "c:\Users\arinb\IdeaProjects\New_Automl\Automl_desk+Andriod\automl_prototype_1"
$pythonDir = "$projectRoot\automl_core\src\main\resources\python"
$reqFile = "$pythonDir\requirements.txt"

Write-Host "`n[1/5] Installing core scientific + ML packages..." -ForegroundColor Yellow
pip install numpy pandas scipy scikit-learn xgboost lightgbm catboost

Write-Host "`n[2/5] Installing Apache Arrow (Java<->Python bridge)..." -ForegroundColor Yellow
pip install pyarrow

Write-Host "`n[3/5] Installing REST server + web utils..." -ForegroundColor Yellow
pip install fastapi uvicorn python-multipart requests httpx

Write-Host "`n[4/5] Installing code analysis (tree-sitter)..." -ForegroundColor Yellow
pip install tree-sitter tree-sitter-java tree-sitter-python

Write-Host "`n[5/5] Installing vision + ONNX..." -ForegroundColor Yellow
pip install pillow opencv-python onnx onnxruntime psutil joblib tqdm

Write-Host "`n[Optional] Installing PyTorch (CUDA 12.1)..." -ForegroundColor Magenta
Write-Host "Run manually if you have an NVIDIA GPU:" -ForegroundColor Magenta
Write-Host "  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121" -ForegroundColor Gray

Write-Host "`n[Optional] Installing HuggingFace Transformers..." -ForegroundColor Magenta
Write-Host "Run manually:" -ForegroundColor Magenta
Write-Host "  pip install transformers datasets accelerate tokenizers" -ForegroundColor Gray

Write-Host "`n=== Setup Complete! ===" -ForegroundColor Green
Write-Host "Tesseract data path: $env:USERPROFILE\AppData\Local\Tesseract-OCR\tessdata" -ForegroundColor Cyan
Write-Host "Ollama models: ollama pull llama3 | ollama pull mistral | ollama pull codellama" -ForegroundColor Cyan
