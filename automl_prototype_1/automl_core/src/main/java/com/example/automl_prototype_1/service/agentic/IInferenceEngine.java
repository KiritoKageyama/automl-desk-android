package com.example.automl_prototype_1.service.agentic;

import java.util.concurrent.CompletableFuture;

/**
 * Strategy interface for inference execution (ONNX vs Python).
 * Phase 6 Step 5.
 */
public interface IInferenceEngine {
    /**
     * Executes prediction asynchronously.
     * 
     * @param modelId  Model identifier (path or ID)
     * @param features Feature matrix for prediction
     * @return Future containing prediction results (int[])
     */
    CompletableFuture<int[]> predict(String modelId, double[][] features);

    /**
     * Gets the engine type name.
     */
    String getEngineName();

    /**
     * Checks if this engine is available/supported on the current system.
     */
    boolean isAvailable();
}
