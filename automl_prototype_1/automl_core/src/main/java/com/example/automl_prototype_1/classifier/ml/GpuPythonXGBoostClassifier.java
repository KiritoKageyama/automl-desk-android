package com.example.automl_prototype_1.classifier.ml;

import com.example.automl_prototype_1.service.IClassifier;
import com.example.automl_prototype_1.gpu.IGpuCachedClassifier;
import com.example.automl_prototype_1.gpu.PythonExecutionService;
import com.example.automl_prototype_1.service.LogService;
import com.example.automl_prototype_1.service.agentic.inference.IInferenceEngine;
import com.example.automl_prototype_1.service.agentic.inference.PythonInferenceEngine;
import com.example.automl_prototype_1.service.agentic.inference.OnnxInferenceEngine;

/**
 * A Python-bridged implementation of XGBoost.
 * This version supports true GPU acceleration via 'gpu_hist' in the Python
 * XGBoost library.
 */
public class GpuPythonXGBoostClassifier implements IClassifier, IGpuCachedClassifier {

    private final int nEstimators;
    private final int maxDepth;
    private final ThreadLocal<String> pythonModelId = new ThreadLocal<>();
    private final ThreadLocal<String> onnxModelPath = new ThreadLocal<>();

    private final IInferenceEngine pythonEngine = new PythonInferenceEngine();
    private final IInferenceEngine onnxEngine = new OnnxInferenceEngine();

    public GpuPythonXGBoostClassifier() {
        this.nEstimators = 50;
        this.maxDepth = 6;
    }

    public GpuPythonXGBoostClassifier(int nEstimators, int maxDepth) {
        this.nEstimators = nEstimators;
        this.maxDepth = maxDepth;
    }

    public String getOnnxModelPath() {
        return onnxModelPath.get();
    }

    @Override
    public String getName() {
        return "XGBoost (Hybrid Python/ONNX/GPU)";
    }

    @Override
    public void train(double[][] features, int[] labels) {
        if (features.length == 0 || features[0].length == 0)
            return;

        if (PythonExecutionService.getInstance().isAvailable()) {
            try {
                if (!PythonExecutionService.getInstance().isServerRunning()) {
                    PythonExecutionService.getInstance().startGpuServer();
                }

                // Detect number of classes
                int numClasses = java.util.Arrays.stream(labels).max().orElse(0) + 1;

                java.util.Map<String, Object> params = new java.util.HashMap<>();
                params.put("n_estimators", nEstimators);
                params.put("max_depth", maxDepth);
                params.put("learning_rate", 0.1);
                params.put("min_child_weight", 1);
                params.put("subsample", 0.8);
                params.put("colsample_bytree", 0.8);

                if (numClasses > 2) {
                    params.put("objective", "multi:softmax");
                    params.put("num_class", numClasses);
                    params.put("eval_metric", "mlogloss");
                    LogService.info("GpuPyXGB",
                            "Multi-class detected (" + numClasses + " classes). Switched to 'multi:softmax'.");
                } else {
                    params.put("objective", "binary:logistic");
                    params.put("eval_metric", "logloss");
                }

                try {
                    trainInternal(features, labels, params, 0);
                } catch (Exception e) {
                    throw new RuntimeException("GPU Training Failed after retries: " + e.getMessage(), e);
                }
            } catch (Exception e) {
                String msg = "CRITICAL: GPU XGBoost Training Failed. " + e.getMessage();
                LogService.error("GpuPyXGB", msg);
                throw new RuntimeException(msg + " Please switch to CPU version.");
            }
        } else {
            String msg = "CRITICAL: GPU/Python Bridge unavailable but GPU classifier selected.";
            LogService.error("GpuPyXGB", msg);
            throw new RuntimeException(msg + " Please enable GPU in settings or switch to standard XGBoost.");
        }
    }

    private void trainInternal(double[][] features, int[] labels, java.util.Map<String, Object> params, int attempt)
            throws Exception {
        try {
            java.util.concurrent.CompletableFuture<Object> future = com.example.automl_prototype_1.gpu.GpuBatchService
                    .getInstance()
                    .submitTrain("XGB", features, labels, params);

            Object result = future.get();
            if (result instanceof String) {
                String rawStr = (String) result;
                if (rawStr.contains("|")) {
                    String[] parts = rawStr.split("\\|");
                    this.pythonModelId.set(parts[0]);
                    this.onnxModelPath.set(parts[1]);
                    LogService.info("GpuPyXGB", "Training Success. ONNX Enabled: " + parts[1]);
                } else {
                    this.pythonModelId.set(rawStr);
                    this.onnxModelPath.remove();
                }
            }
        } catch (java.util.concurrent.ExecutionException ee) {
            if (attempt < 5 && (ee.getCause() instanceof java.net.ConnectException
                    || ee.getCause() instanceof java.io.IOException)) {
                LogService.info("GpuPyXGB", "[WARNING] Connection issue. Retrying (Attempt " + (attempt + 1) + ")...");
                Thread.sleep(2000);
                trainInternal(features, labels, params, attempt + 1);
            } else {
                throw ee;
            }
        }
    }

    @Override
    public int[] predict(double[][] features) {
        String onnxPath = this.onnxModelPath.get();
        if (onnxPath != null) {
            try {
                return onnxEngine.predict(onnxPath, features).get();
            } catch (Exception e) {
                LogService.warn("GpuPyXGB", "ONNX Inference failed (" + e.getMessage() + "). Fallback to Python.");
            }
        }

        String modelId = this.pythonModelId.get();
        if (modelId != null && PythonExecutionService.getInstance().isAvailable()) {
            try {
                return pythonEngine.predict(modelId, features).get();
            } catch (Exception e) {
                String msg = "CRITICAL: GPU XGBoost Prediction Failed. " + e.getMessage();
                LogService.error("GpuPyXGB", msg);
                throw new RuntimeException(msg);
            }
        } else {
            throw new RuntimeException("CRITICAL: XGBoost Model not trained or GPU unavailable.");
        }
    }

    @Override
    public void trainCached(String datasetId, int[] featureIndices, int[] rowIndices) {
        try {
            String rawStr = submitTrainCachedAsync(datasetId, featureIndices, rowIndices).get();
            if (rawStr.contains("|")) {
                String[] parts = rawStr.split("\\|");
                this.pythonModelId.set(parts[0]);
                this.onnxModelPath.set(parts[1]);
            } else {
                this.pythonModelId.set(rawStr);
                this.onnxModelPath.remove();
            }
        } catch (Exception e) {
            throw new RuntimeException("GPU XGB Cached Training Failed", e);
        }
    }

    @Override
    public int[] predictCached(String datasetId, int[] featureIndices, int[] rowIndices) {
        String modelId = this.pythonModelId.get();
        if (modelId != null) {
            try {
                return submitPredictCachedAsync(modelId, datasetId, featureIndices, rowIndices).get();
            } catch (Exception e) {
                throw new RuntimeException("GPU XGB Cached Prediction Failed", e);
            }
        }
        return new int[0];
    }

    @Override
    public java.util.concurrent.CompletableFuture<String> submitTrainCachedAsync(String datasetId, int[] featureIndices,
            int[] rowIndices) {
        if (PythonExecutionService.getInstance().isAvailable()) {
            java.util.Map<String, Object> params = new java.util.HashMap<>();
            params.put("n_estimators", nEstimators);
            params.put("max_depth", maxDepth);
            // Add other optimization params if needed
            params.put("learning_rate", 0.1);
            params.put("min_child_weight", 1);
            params.put("subsample", 0.8);
            params.put("colsample_bytree", 0.8);

            return com.example.automl_prototype_1.gpu.GpuBatchService.getInstance()
                    .submitTrainCached("XGB", datasetId, featureIndices, rowIndices, params)
                    .thenApply(res -> (String) res);
        } else {
            java.util.concurrent.CompletableFuture<String> f = new java.util.concurrent.CompletableFuture<>();
            f.completeExceptionally(new RuntimeException("Python Bridge unavailable"));
            return f;
        }
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
                LogService.error("GpuPyXGB", "Cleanup failed for " + modelId);
            }
        }
    }
}
