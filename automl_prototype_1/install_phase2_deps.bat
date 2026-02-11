@echo off
echo Installing Python dependencies for Phase 2 (Code Understanding)...
pip install tree-sitter tree-sitter-java tree-sitter-python
if %errorlevel% neq 0 (
    echo [ERROR] Failed to install dependencies.
    exit /b %errorlevel%
)
echo [SUCCESS] Dependencies installed.
echo Testing analysis_engine.py logic...
python automl_core/src/main/resources/python/analysis_engine.py "automl_core/src/main/java/com/example/automl_prototype_1/service/agentic/DomainGuardService.java"
