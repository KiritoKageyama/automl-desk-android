package com.example.automl_prototype_1.classifier.ml;

import com.example.automl_prototype_1.service.IClassifier;
import com.example.automl_prototype_1.gpu.IGpuCachedClassifier; // Added
import com.example.automl_prototype_1.service.LogService;
import com.example.automl_prototype_1.gpu.PythonExecutionService;
import com.example.automl_prototype_1.service.agentic.inference.IInferenceEngine;
import com.example.automl_prototype_1.service.agentic.inference.PythonInferenceEngine;
import com.example.automl_prototype_1.service.agentic.inference.OnnxInferenceEngine;

import java.util.HashMap;
import java.util.Map;

/**
 * CatBoost Classifier (via Python Bridge).
 * Supports GPU acceleration automatically.
 */
public class GpuPythonCatBoostClassifier implements IClassifier, IGpuCachedClassifier {

    private final ThreadLocal<String> pythonModelId = new ThreadLocal<>();
    private final ThreadLocal<String> onnxModelPath = new ThreadLocal<>();
    private final int iterations;
    private final double learningRate;
    private final IInferenceEngine pythonEngine = new PythonInferenceEngine();
    private final IInferenceEngine onnxEngine = new OnnxInferenceEngine();

    public GpuPythonCatBoostClassifier() {
        this(500, 0.1);
    }

    public GpuPythonCatBoostClassifier(int iterations, double learningRate) {
        this.iterations = iterations;
        this.learningRate = learningRate;
    }

    public String getOnnxModelPath() {
        return onnxModelPath.get();
    }

    @Override
    public String getName() {
        return "CatBoost (Hybrid Python/ONNX/GPU)";
    }

    @Override
    public void train(double[][] features, int[] labels) {
        if (!PythonExecutionService.getInstance().isAvailable()) {
            throw new RuntimeException("Python Bridge not available for CatBoost.");
        }

        try {
            // Detect number of classes
            int numClasses = java.util.Arrays.stream(labels).max().orElse(0) + 1;

            Map<String, Object> params = new HashMap<>();
            params.put("iterations", iterations);
            params.put("learning_rate", learningRate);

            if (numClasses > 2) {
                params.put("loss_function", "MultiClass");
                params.put("eval_metric", "MultiClass");
                params.put("classes_count", numClasses); // CatBoost might infer this, but good to be explicit
                LogService.info("GpuCatBoost",
                        "Multi-class detected (" + numClasses + " classes). Switched to 'MultiClass'.");
            } else {
                params.put("loss_function", "Logloss");
                params.put("eval_metric", "Logloss");
            }

            java.util.concurrent.CompletableFuture<Object> future = com.example.automl_prototype_1.gpu.GpuBatchService
                    .getInstance()
                    .submitTrain("CATBOOST", features, labels, params);

            Object result = future.get();
            if (result instanceof String) {
                String rawStr = (String) result;
                if (rawStr.contains("|")) {
                    String[] parts = rawStr.split("\\|");
                    this.pythonModelId.set(parts[0]);
                    this.onnxModelPath.set(parts[1]);
                    LogService.info("GpuCatBoost", "Training Success. ONNX Enabled: " + parts[1]);
                } else {
                    this.pythonModelId.set(rawStr);
                    this.onnxModelPath.remove();
                }
            }

        } catch (Exception e) {
            LogService.error("GpuCatBoost", "Training Failed: " + e.getMessage());
            throw new RuntimeException("GPU CatBoost Training Failed", e);
        }
    }

    @Override
    public int[] predict(double[][] features) {
        String onnxPath = this.onnxModelPath.get();
        if (onnxPath != null) {
            try {
                return onnxEngine.predict(onnxPath, features).get();
            } catch (Exception e) {
                LogService.warn("GpuCatBoost", "ONNX Inference failed. Fallback to Python.");
            }
        }

        String modelId = this.pythonModelId.get();
        if (modelId != null) {
            try {
                return pythonEngine.predict(modelId, features).get();
            } catch (Exception e) {
                LogService.error("GpuCatBoost", "Prediction Failed: " + e.getMessage());
                throw new RuntimeException("GPU CatBoost Prediction Failed", e);
            }
        }
        return new int[0];
    }

    @Override
    public void trainCached(String datasetId, int[] featureIndices, int[] rowIndices) {
        if (!PythonExecutionService.getInstance().isAvailable()) {
            throw new RuntimeException("Python Bridge not available of Cached CatBoost.");
        }

        try {
            java.util.Map<String, Object> params = new java.util.HashMap<>();
            params.put("iterations", iterations);
            params.put("learning_rate", learningRate);

            java.util.concurrent.CompletableFuture<Object> future = com.example.automl_prototype_1.gpu.GpuBatchService
                    .getInstance()
                    .submitTrainCached("CATBOOST", datasetId, featureIndices, rowIndices, params);

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
            throw new RuntimeException("GPU CatBoost Cached Training Failed", e);
        }
    }

    @Override
    public int[] predictCached(String datasetId, int[] featureIndices, int[] rowIndices) {
        String modelId = this.pythonModelId.get();
        if (modelId != null) {
            try {
                return submitPredictCachedAsync(modelId, datasetId, featureIndices, rowIndices).get();
            } catch (Exception e) {
                throw new RuntimeException("GPU CatBoost Cached Prediction Failed", e);
            }
        }
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
        java.util.Map<String, Object> params = new java.util.HashMap<>();
        params.put("iterations", iterations);
        params.put("learning_rate", learningRate);

        return com.example.automl_prototype_1.gpu.GpuBatchService.getInstance()
                .submitTrainCached("CATBOOST", datasetId, featureIndices, rowIndices, params)
                .thenApply(res -> (String) res);
    }

    @Override
    public java.util.concurrent.CompletableFuture<int[]> submitPredictCachedAsync(String modelId, String datasetId,
            int[] featureIndices, int[] rowIndices) {
        return com.example.automl_prototype_1.gpu.GpuBatchService.getInstance()
                .submitPredictCached(modelId, datasetId, featureIndices, rowIndices)
                .thenApply(res -> (int[]) res);
    }

    public void cleanup() {
        String modelId = this.pythonModelId.get();
        if (modelId != null) {
            try {
                com.example.automl_prototype_1.gpu.GpuBatchService.getInstance().submitCleanup(modelId);
                this.pythonModelId.remove();
            } catch (Exception e) {
                LogService.error("GpuCatBoost", "Cleanup failed for " + modelId);
            }
        }
    }
}
