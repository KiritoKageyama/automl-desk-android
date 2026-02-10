package com.example.automl_prototype_1.service.agentic;

import com.example.automl_prototype_1.service.GpuBatchService;
import java.util.concurrent.CompletableFuture;

/**
 * Legacy inference engine using the persistent Python server.
 */
public class PythonInferenceEngine implements IInferenceEngine {

    @Override
    public CompletableFuture<int[]> predict(String modelId, double[][] features) {
        // Bridge to existing GpuBatchService
        // GpuBatchService returns CompletableFuture<Object>; we cast to int[]
        // Note: The Python server returns int[] for predictions.
        return GpuBatchService.getInstance().submitPredict(modelId, features)
                .thenApply(result -> (int[]) result);
    }

    @Override
    public String getEngineName() {
        return "PYTHON_SERVER";
    }

    @Override
    public boolean isAvailable() {
        // We assume it's available if we are running.
        // Real check involves PythonExecutionService.
        return true;
    }
}
