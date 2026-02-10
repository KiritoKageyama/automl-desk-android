package com.example.automl_prototype_1.classifier;

import com.example.automl_prototype_1.service.IClassifier;
import com.example.automl_prototype_1.service.IGpuCachedClassifier; // Added
import com.example.automl_prototype_1.service.LogService;
import com.example.automl_prototype_1.service.PythonExecutionService;

import java.util.HashMap;
import java.util.Map;

/**
 * LightGBM Classifier (via Python Bridge).
 * Supports GPU acceleration automatically.
 */
public class GpuPythonLightGbmClassifier implements IClassifier, IGpuCachedClassifier {

    private final ThreadLocal<String> pythonModelId = new ThreadLocal<>();
    private final ThreadLocal<String> onnxModelPath = new ThreadLocal<>();
    private final int nEstimators;
    private final double learningRate;
    private final com.example.automl_prototype_1.service.agentic.IInferenceEngine pythonEngine = new com.example.automl_prototype_1.service.agentic.PythonInferenceEngine();
    private final com.example.automl_prototype_1.service.agentic.IInferenceEngine onnxEngine = new com.example.automl_prototype_1.service.agentic.OnnxInferenceEngine();

    public GpuPythonLightGbmClassifier() {
        this(100, 0.1);
    }

    public GpuPythonLightGbmClassifier(int nEstimators, double learningRate) {
        this.nEstimators = nEstimators;
        this.learningRate = learningRate;
    }

    @Override
    public String getName() {
        return "LightGBM (Hybrid Python/ONNX/GPU)";
    }

    @Override
    public void train(double[][] features, int[] labels) {
        if (!PythonExecutionService.getInstance().isAvailable()) {
            throw new RuntimeException("Python Bridge not available for LightGBM.");
        }

        try {
            Map<String, Object> params = new HashMap<>();
            params.put("n_estimators", nEstimators);
            params.put("learning_rate", learningRate);

            java.util.concurrent.CompletableFuture<Object> future = com.example.automl_prototype_1.service.GpuBatchService
                    .getInstance()
                    .submitTrain("LGBM", features, labels, params);

            Object result = future.get();
            if (result instanceof String) {
                String rawStr = (String) result;
                if (rawStr.contains("|")) {
                    String[] parts = rawStr.split("\\|");
                    this.pythonModelId.set(parts[0]);
                    // Convert potential WSL path to Windows path for Java access
                    String convertedPath = com.example.automl_prototype_1.service.PythonExecutionService.getInstance()
                            .convertFromWslPath(parts[1]);
                    this.onnxModelPath.set(convertedPath);
                    LogService.info("GpuLGBM", "Training Success. ONNX Enabled: " + convertedPath);
                } else {
                    this.pythonModelId.set(rawStr);
                    this.onnxModelPath.remove();
                }
            }

        } catch (Exception e) {
            LogService.error("GpuLGBM", "Training Failed: " + e.getMessage());
            throw new RuntimeException("GPU LightGBM Training Failed", e);
        }
    }

    @Override
    public int[] predict(double[][] features) {
        String onnxPath = this.onnxModelPath.get();
        if (onnxPath != null) {
            try {
                return onnxEngine.predict(onnxPath, features).get();
            } catch (Exception e) {
                LogService.warn("GpuLGBM", "ONNX Inference failed. Fallback to Python.");
            }
        }

        String modelId = this.pythonModelId.get();
        if (modelId != null) {
            try {
                return pythonEngine.predict(modelId, features).get();
            } catch (Exception e) {
                LogService.error("GpuLGBM", "Prediction Failed: " + e.getMessage());
                throw new RuntimeException("GPU LightGBM Prediction Failed", e);
            }
        }
        return new int[0];
    }

    @Override
    public void trainCached(String datasetId, int[] featureIndices, int[] rowIndices) {
        if (!PythonExecutionService.getInstance().isAvailable()) {
            throw new RuntimeException("Python Bridge not available for Cached LightGBM.");
        }

        try {
            java.util.Map<String, Object> params = new HashMap<>();
            params.put("n_estimators", nEstimators);
            params.put("learning_rate", learningRate);

            java.util.concurrent.CompletableFuture<Object> future = com.example.automl_prototype_1.service.GpuBatchService
                    .getInstance()
                    .submitTrainCached("LGBM", datasetId, featureIndices, rowIndices, params);

            Object result = future.get();
            if (result instanceof String) {
                String rawStr = (String) result;
                if (rawStr.contains("|")) {
                    String[] parts = rawStr.split("\\|");
                    this.pythonModelId.set(parts[0]);
                    this.onnxModelPath.set(parts[1]);
                } else {
                    this.pythonModelId.set(rawStr);
                    this.onnxModelPath.remove();
                }
            }
        } catch (Exception e) {
            throw new RuntimeException("GPU LightGBM Cached Training Failed", e);
        }
    }

    @Override
    public int[] predictCached(String datasetId, int[] featureIndices, int[] rowIndices) {
        String modelId = this.pythonModelId.get();
        if (modelId != null) {
            try {
                int[] preds = submitPredictCachedAsync(modelId, datasetId, featureIndices, rowIndices).get();
                LogService.info("GpuLGBM",
                        "predictCached returned " + (preds == null ? "NULL" : preds.length + " predictions"));
                return preds;
            } catch (Exception e) {
                LogService.error("GpuLGBM", "Cached Prediction Failed: " + e.getMessage());
                throw new RuntimeException("GPU LightGBM Cached Prediction Failed", e);
            }
        }
        LogService.warn("GpuLGBM", "predictCached called but modelId is NULL - returning empty array");
        return new int[0];
    }

    @Override
    public java.util.concurrent.CompletableFuture<String> submitTrainCachedAsync(String datasetId, int[] featureIndices,
            int[] rowIndices) {
        if (!PythonExecutionService.getInstance().isAvailable()) {
            java.util.concurrent.CompletableFuture<String> f = new java.util.concurrent.CompletableFuture<>();
            f.completeExceptionally(new RuntimeException("Python Bridge unavailable"));
            return f;
        }
        java.util.Map<String, Object> params = new HashMap<>();
        params.put("n_estimators", nEstimators);
        params.put("learning_rate", learningRate);

        return com.example.automl_prototype_1.service.GpuBatchService.getInstance()
                .submitTrainCached("LGBM", datasetId, featureIndices, rowIndices, params)
                .thenApply(res -> (String) res);
    }

    @Override
    public java.util.concurrent.CompletableFuture<int[]> submitPredictCachedAsync(String modelId, String datasetId,
            int[] featureIndices, int[] rowIndices) {
        return com.example.automl_prototype_1.service.GpuBatchService.getInstance()
                .submitPredictCached(modelId, datasetId, featureIndices, rowIndices)
                .thenApply(res -> (int[]) res);
    }

    public void cleanup() {
        String modelId = this.pythonModelId.get();
        if (modelId != null) {
            try {
                com.example.automl_prototype_1.service.GpuBatchService.getInstance().submitCleanup(modelId);
                this.pythonModelId.remove();
            } catch (Exception e) {
                LogService.error("GpuLGBM", "Cleanup failed for " + modelId);
            }
        }
    }
}
