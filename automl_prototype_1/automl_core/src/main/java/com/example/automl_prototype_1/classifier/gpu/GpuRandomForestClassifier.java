package com.example.automl_prototype_1.classifier.gpu;

import com.example.automl_prototype_1.service.IClassifier;
import com.example.automl_prototype_1.gpu.IGpuCachedClassifier;
import com.example.automl_prototype_1.gpu.PythonExecutionService;
import com.example.automl_prototype_1.service.LogService;
import com.example.automl_prototype_1.service.agentic.inference.IInferenceEngine;
import com.example.automl_prototype_1.service.agentic.inference.PythonInferenceEngine;
import com.example.automl_prototype_1.service.agentic.inference.OnnxInferenceEngine;
import smile.classification.RandomForest;
import smile.data.DataFrame;
import smile.data.formula.Formula;

import java.util.Properties;

/**
 * A GPU-enabled wrapper for Random Forest Classifier.
 * Uses Python Bridge (cuML/scikit-learn) if available, falls back to CPU Smile
 * implementation.
 */
public class GpuRandomForestClassifier implements IClassifier, IGpuCachedClassifier {
    private final int nTrees;
    private RandomForest javaModel;
    private final ThreadLocal<String> pythonModelId = new ThreadLocal<>();
    private final ThreadLocal<String> onnxModelPath = new ThreadLocal<>();
    private boolean usePython = false;

    public String getOnnxModelPath() {
        return onnxModelPath.get();
    }

    private final IInferenceEngine pythonEngine = new PythonInferenceEngine();
    private final IInferenceEngine onnxEngine = new OnnxInferenceEngine();

    public GpuRandomForestClassifier(int nTrees) {
        this.nTrees = nTrees;
    }

    public GpuRandomForestClassifier() {
        this(100);
    }

    @Override
    public String getName() {
        return usePython ? "Random Forest (Hybrid Python/ONNX/GPU)" : "Random Forest (Java CPU Fallback)";
    }

    @Override
    public void train(double[][] features, int[] labels) {
        if (features.length == 0 || features[0].length == 0) {
            return;
        }

        // 1. Try Python Bridge (Batched)
        if (PythonExecutionService.getInstance().isAvailable()) {
            try {
                // Ensure Server is Running
                PythonExecutionService.getInstance().startGpuServer();

                LogService.info("GpuRF", "Submitting training job to GPU Batch Service...");

                java.util.Map<String, Object> params = new java.util.HashMap<>();
                params.put("n_estimators", nTrees);

                // Submit and Block (Synchronous wait for Async Batch)
                java.util.concurrent.CompletableFuture<Object> future = com.example.automl_prototype_1.gpu.GpuBatchService
                        .getInstance()
                        .submitTrain("RF", features, labels, params);

                Object result = future.get(); // Blocking call

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
                    this.usePython = true;
                    LogService.info("GpuRF", "GPU Training successful. Model ID: " + this.pythonModelId.get());
                    return;
                }
            } catch (Exception e) {
                LogService.error("GpuRF", "GPU Batch Training failed: " + e.getMessage());
                // Fallthrough to Java
            }
        }

        // 2. Fallback to Java (Smile)
        LogService.info("GpuRF", "Falling back to pure Java implementation (CPU).");
        this.usePython = false;

        int[][] labelMatrix = new int[labels.length][1];
        for (int i = 0; i < labels.length; i++) {
            labelMatrix[i][0] = labels[i];
        }

        DataFrame dataFrame = DataFrame.of(features).merge(DataFrame.of(labelMatrix, "Class"));
        Formula formula = Formula.of("Class", ".");
        Properties props = new Properties();
        props.setProperty("smile.random.forest.trees", String.valueOf(this.nTrees));
        this.javaModel = RandomForest.fit(formula, dataFrame, props);
    }

    @Override
    public int[] predict(double[][] features) {
        String onnxPath = this.onnxModelPath.get();
        if (this.usePython && onnxPath != null) {
            try {
                return onnxEngine.predict(onnxPath, features).get();
            } catch (Exception e) {
                LogService.warn("GpuRF", "ONNX Inference failed. Fallback to Python.");
            }
        }

        String modelId = this.pythonModelId.get();
        if (this.usePython && modelId != null) {
            try {
                return pythonEngine.predict(modelId, features).get();
            } catch (Exception e) {
                LogService.error("GpuRF", "GPU Inference failed: " + e.getMessage());
                return new int[features.length];
            }
        }

        // Java Fallback
        if (this.javaModel == null)
            return new int[features.length];

        DataFrame predictionFrame = DataFrame.of(features);
        return this.javaModel.predict(predictionFrame);
    }

    @Override
    public void trainCached(String datasetId, int[] featureIndices, int[] rowIndices) {
        if (PythonExecutionService.getInstance().isAvailable()) {
            try {
                java.util.Map<String, Object> params = new java.util.HashMap<>();
                params.put("n_estimators", nTrees);

                java.util.concurrent.CompletableFuture<Object> future = com.example.automl_prototype_1.gpu.GpuBatchService
                        .getInstance()
                        .submitTrainCached("RF", datasetId, featureIndices, rowIndices, params);

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
                    this.usePython = true;
                }
            } catch (Exception e) {
                LogService.error("GpuRF", "Cached Training Failed: " + e.getMessage());
                throw new RuntimeException("GPU RF Cached Training Failed", e);
            }
        } else {
            throw new RuntimeException("Python Bridge unavailable for Cached Training");
        }
    }

    @Override
    public int[] predictCached(String datasetId, int[] featureIndices, int[] rowIndices) {
        String modelId = this.pythonModelId.get();
        if (this.usePython && modelId != null) {
            try {
                return submitPredictCachedAsync(modelId, datasetId, featureIndices, rowIndices).get();
            } catch (Exception e) {
                throw new RuntimeException("GPU RF Cached Prediction Failed", e);
            }
        }
        return new int[0];
    }

    @Override
    public java.util.concurrent.CompletableFuture<String> submitTrainCachedAsync(String datasetId, int[] featureIndices,
            int[] rowIndices) {
        if (!PythonExecutionService.getInstance().isAvailable()) {
            throw new RuntimeException("Python Bridge unavailable");
        }
        java.util.Map<String, Object> params = new java.util.HashMap<>();
        params.put("n_estimators", nTrees);
        return com.example.automl_prototype_1.gpu.GpuBatchService.getInstance()
                .submitTrainCached("RF", datasetId, featureIndices, rowIndices, params)
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
