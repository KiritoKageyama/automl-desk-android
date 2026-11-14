# Package: GPU

## Feature Context
This package provides the **Hardware Acceleration Bridge**. It manages the communication between the Java JVM and the Python process that hosts GPU-accelerated libraries (PyTorch, RAPIDS/cuML, XGBoost). It ensures that heavy math operations are offloaded to code that can run on CUDA/ROCm.

## Architecture Position
**Infrastructure Layer**
- **Upstream**: Used by `ExecutionService` when High-Performance Compute (HPC) mode is enabled.
- **Side-stream**: Manages the `gpu_server.py` lifecycle.

## File Map
- **`PythonExecutionService.java`**: The JNI/Process-Builder bridge. Starts the Python server, manages serialization of data (Arrow/JSON), and handles heartbeat checks.
- **`HardwareAcceleratorService.java`**: Detects available hardware (NVIDIA GPU, AMD GPU, CPU features) and configures the environment accordingly.
