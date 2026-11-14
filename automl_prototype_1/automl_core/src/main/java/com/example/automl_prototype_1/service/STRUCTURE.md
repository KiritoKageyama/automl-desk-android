# Package: Service

## Feature Context
This package contains the **Business Logic and Pipeline Orchestration** layer of the AutoML system. It manages the lifecycle of machine learning tasks, from data loading to model training, evaluation, and storage. It serves as the bridge between the raw algorithms and the user interface.

## Architecture Position
**Service / Middleware Layer**
- **Upstream**: invoked by Controllers (UI) to start tasks.
- **Downstream**: Invokes `algorithm` classes, `dataprovider` classes, and `gpu` services.

## File Map
- **`pipeline/ExecutionService.java`**: The primary engine. Manages a persistent thread pool to run ML algorithms in parallel. Handles success/failure callbacks and resource cleanup.
- **`pipeline/PipelineValidator.java`**: Ensures datasets and configurations are valid before execution begins (e.g., checking for non-numeric data in numeric-only models).
- **`RecommendationService.java`**: The "Meta-Learner". Analyzes dataset meta-features (rows, columns, skewness) to recommend the best algorithms before training starts.
- **`LogService.java`**: Centralized logging utility that routes messages to both the console and the UI's log panels.
- **`AutoMlService.java`**: A facade/wrapper often used as the entry point for simpler AutoML requests.
