package com.example.automl_prototype_1.service.pipeline;

import ai.onnxruntime.OrtException;
import com.example.automl_prototype_1.service.LogService;
import com.example.automl_prototype_1.service.agentic.inference.OnnxInferenceEngine;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Service to execute ONNX models locally using the Microsoft ONNX Runtime.
 * Delegates to OnnxInferenceEngine (single source of truth with CUDA support).
 */
public class OnnxInferenceService {

    private static OnnxInferenceService instance;
    private final OnnxInferenceEngine engine;
    /** modelId (or alias) -> absolute onnx file path used by the engine session cache */
    private final Map<String, String> modelPaths = new ConcurrentHashMap<>();

    private OnnxInferenceService() {
        this.engine = new OnnxInferenceEngine();
        LogService.info("OnnxService", "ONNX Service Initialized (delegating to OnnxInferenceEngine).");
    }

    public static synchronized OnnxInferenceService getInstance() {
        if (instance == null) {
            instance = new OnnxInferenceService();
        }
        return instance;
    }

    /**
     * Registers and pre-loads an ONNX model session.
     *
     * @param modelId  logical id (UUID) or path alias
     * @param onnxPath absolute path to .onnx file
     */
    public void loadModel(String modelId, String onnxPath) throws OrtException {
        if (modelId == null || onnxPath == null) {
            throw new IllegalArgumentException("modelId and onnxPath are required");
        }
        modelPaths.put(modelId, onnxPath);
        modelPaths.put(onnxPath, onnxPath);
        try {
            engine.preload(onnxPath);
            LogService.info("OnnxService", "Model loaded: " + modelId + " from " + onnxPath);
        } catch (Exception e) {
            LogService.error("OnnxService", "Failed to preload ONNX model: " + e.getMessage());
            if (e.getCause() instanceof OrtException) {
                throw (OrtException) e.getCause();
            }
            throw new OrtException("Failed to preload ONNX model: " + e.getMessage());
        }
    }

    /**
     * Runs inference by resolving modelId → path, then delegating to OnnxInferenceEngine.
     */
    public int[] predict(String modelId, double[][] features) throws OrtException {
        String path = modelPaths.getOrDefault(modelId, modelId);
        try {
            return engine.predict(path, features).get();
        } catch (Exception e) {
            LogService.error("OnnxService", "Inference failed: " + e.getMessage());
            if (e.getCause() instanceof OrtException) {
                throw (OrtException) e.getCause();
            }
            throw new RuntimeException("ONNX inference failed for model: " + modelId, e);
        }
    }

    public void unloadModel(String modelId) {
        String path = modelPaths.remove(modelId);
        if (path != null) {
            modelPaths.remove(path);
            engine.unload(path);
        }
        LogService.info("OnnxService", "Unload requested for: " + modelId);
    }
}
