package com.example.automl_prototype_1.service.agentic.inference;

import com.example.automl_prototype_1.gpu.GpuBatchService;
import java.util.concurrent.CompletableFuture;

/**
 * Legacy inference engine using the persistent Python server.
 */
public class PythonInferenceEngine implements IInferenceEngine {

    @Override
    public CompletableFuture<int[]> predict(String modelId, double[][] features) {
        return GpuBatchService.getInstance().submitPredict(modelId, features)
                .thenApply(result -> (int[]) result);
    }

    @Override
    public String getEngineName() {
        return "PYTHON_SERVER";
    }

    @Override
    public boolean isAvailable() {
        return true;
    }
}
