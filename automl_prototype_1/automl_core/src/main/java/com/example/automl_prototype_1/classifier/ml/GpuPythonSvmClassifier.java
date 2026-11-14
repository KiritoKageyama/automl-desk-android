package com.example.automl_prototype_1.classifier.ml;

import com.example.automl_prototype_1.service.IClassifier;
import com.example.automl_prototype_1.gpu.IGpuCachedClassifier;
import com.example.automl_prototype_1.gpu.PythonExecutionService;
import com.example.automl_prototype_1.service.LogService;
import com.example.automl_prototype_1.service.agentic.inference.IInferenceEngine;
import com.example.automl_prototype_1.service.agentic.inference.PythonInferenceEngine;
import com.example.automl_prototype_1.service.agentic.inference.OnnxInferenceEngine;

/**
 * A Python-bridged implementation of SVM.
 * This version supports true GPU acceleration for RBF/Poly kernels via cuML.
 */
public class GpuPythonSvmClassifier implements IClassifier, IGpuCachedClassifier {

    private final double C;
    private final String kernel;
    private final ThreadLocal<String> pythonModelId = new ThreadLocal<>();
    private final ThreadLocal<String> onnxModelPath = new ThreadLocal<>();
    private final IInferenceEngine pythonEngine = new PythonInferenceEngine();
    private final IInferenceEngine onnxEngine = new OnnxInferenceEngine();

    public GpuPythonSvmClassifier() {
        this(1.0, "rbf");
    }

    public GpuPythonSvmClassifier(double C, String kernel) {
        this.C = C;
        this.kernel = kernel;
    }

    public String getOnnxModelPath() {
        return onnxModelPath.get();
    }

    @Override
    public String getName() {
        return "SVM (Hybrid Python/ONNX/GPU - " + kernel + ")";
    }

    @Override
    public void train(double[][] features, int[] labels) {
        if (features.length == 0 || features[0].length == 0)
            return;

        // Strict GPU Execution
        if (PythonExecutionService.getInstance().isAvailable()) {
            try {
                if (!PythonExecutionService.getInstance().isServerRunning()) {
                    PythonExecutionService.getInstance().startGpuServer();
                }

                java.util.Map<String, Object> params = new java.util.HashMap<>();
                params.put("C", C);
                params.put("kernel", kernel);

                java.util.concurrent.CompletableFuture<Object> future = com.example.automl_prototype_1.gpu.GpuBatchService
                        .getInstance()
                        .submitTrain("SVM", features, labels, params);

                Object result = future.get();
                if (result instanceof String) {
                    String rawStr = (String) result;
                    if (rawStr.contains("|")) {
                        String[] parts = rawStr.split("\\|");
                        this.pythonModelId.set(parts[0]);
                        this.onnxModelPath.set(parts[1]);
                        LogService.info("GpuPySVM", "Training Success. ONNX Enabled: " + parts[1]);
                    } else {
                        this.pythonModelId.set(rawStr);
                        this.onnxModelPath.remove();
                    }
                    return;
                }
            } catch (Exception e) {
                // Critical Error - STOP and ASK
                String msg = "CRITICAL: GPU SVM Training Failed. " + e.getMessage();
                LogService.error("GpuPySVM", msg);
                throw new RuntimeException(msg + " Please switch to CPU version.");
            }
        }

        // If Python not available at all but user selected GPU Wrapper
        String msg = "CRITICAL: GPU/Python Bridge unavailable but GPU classifier selected.";
        LogService.error("GpuPySVM", msg);
        throw new RuntimeException(msg + " Please enable GPU in settings or switch to standard SVM.");
    }

    @Override
    public int[] predict(double[][] features) {
        String onnxPath = this.onnxModelPath.get();
        if (onnxPath != null) {
            try {
                return onnxEngine.predict(onnxPath, features).get();
            } catch (Exception e) {
                LogService.warn("GpuPySVM", "ONNX Inference failed. Fallback to Python.");
            }
        }

        String modelId = this.pythonModelId.get();
        if (modelId != null && PythonExecutionService.getInstance().isAvailable()) {
            try {
                return pythonEngine.predict(modelId, features).get();
            } catch (Exception e) {
                String msg = "CRITICAL: GPU SVM Prediction Failed. " + e.getMessage();
                LogService.error("GpuPySVM", msg);
                throw new RuntimeException(msg);
            }
        }
        throw new RuntimeException("CRITICAL: SVM Model not trained or GPU unavailable.");
    }

    @Override
    public void trainCached(String datasetId, int[] featureIndices, int[] rowIndices) {
        if (PythonExecutionService.getInstance().isAvailable()) {
            try {
                if (!PythonExecutionService.getInstance().isServerRunning()) {
                    PythonExecutionService.getInstance().startGpuServer();
                }

                java.util.Map<String, Object> params = new java.util.HashMap<>();
                params.put("C", C);
                params.put("kernel", kernel);

                java.util.concurrent.CompletableFuture<Object> future = com.example.automl_prototype_1.gpu.GpuBatchService
                        .getInstance()
                        .submitTrainCached("SVM", datasetId, featureIndices, rowIndices, params);

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
                String msg = "CRITICAL: GPU SVM Cached Training Failed. " + e.getMessage();
                LogService.error("GpuPySVM", msg);
                throw new RuntimeException(msg);
            }
        } else {
            throw new RuntimeException("GPU/Python Bridge unavailable for Cached Training");
        }
    }

    @Override
    public int[] predictCached(String datasetId, int[] featureIndices, int[] rowIndices) {
        String modelId = this.pythonModelId.get();
        if (modelId != null && PythonExecutionService.getInstance().isAvailable()) {
            try {
                return submitPredictCachedAsync(modelId, datasetId, featureIndices, rowIndices).get();
            } catch (Exception e) {
                throw new RuntimeException("GPU SVM Cached Prediction Failed", e);
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
        params.put("C", C);
        params.put("kernel", kernel);

        return com.example.automl_prototype_1.gpu.GpuBatchService.getInstance()
                .submitTrainCached("SVM", datasetId, featureIndices, rowIndices, params)
                .thenApply(res -> (String) res);
    }

    @Override
    public java.util.concurrent.CompletableFuture<int[]> submitPredictCachedAsync(String modelId, String datasetId,
            int[] featureIndices, int[] rowIndices) {
        return com.example.automl_prototype_1.gpu.GpuBatchService.getInstance()
                .submitPredictCached(modelId, datasetId, featureIndices, rowIndices)
                .thenApply(res -> (int[]) res);
    }
}
