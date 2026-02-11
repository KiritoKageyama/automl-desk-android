package com.example.automl_prototype_1.service.pipeline;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import com.example.automl_prototype_1.service.LogService;

import java.nio.FloatBuffer;
import java.util.Collections;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Service to execute ONNX models locally using the Microsoft ONNX Runtime.
 * This removes the HTTP overhead for predictions.
 */
public class OnnxInferenceService {

    private static OnnxInferenceService instance;
    private final OrtEnvironment env;

    // Cache: Model ID -> OrtSession
    private final Map<String, OrtSession> sessionCache = new ConcurrentHashMap<>();

    private OnnxInferenceService() {
        this.env = OrtEnvironment.getEnvironment();
        LogService.info("OnnxService", "ONNX Environment Initialized.");
    }

    public static synchronized OnnxInferenceService getInstance() {
        if (instance == null) {
            instance = new OnnxInferenceService();
        }
        return instance;
    }

    /**
     * Loads an ONNX model from the filesystem.
     * 
     * @param modelId  Unique ID to associate with this session.
     * @param onnxPath Absolute path to the .onnx file.
     */
    public void loadModel(String modelId, String onnxPath) throws OrtException {
        if (sessionCache.containsKey(modelId)) {
            return; // Already loaded
        }

        OrtSession.SessionOptions opts = new OrtSession.SessionOptions();
        // We could enable GPU here if we had the GPU package, but for now CPU is fast
        // enough for inference
        // opts.addCUDA(0);

        LogService.info("OnnxService", "Loading model " + modelId + " from " + onnxPath);
        OrtSession session = env.createSession(onnxPath, opts);
        sessionCache.put(modelId, session);
    }

    /**
     * Runs inference on the given features.
     * 
     * @param modelId  The ID of the loaded model.
     * @param features 2D array of features [rows][cols]
     * @return Array of predictions (int convention for classification)
     */
    public int[] predict(String modelId, double[][] features) throws OrtException {
        OrtSession session = sessionCache.get(modelId);
        if (session == null) {
            throw new RuntimeException("Model " + modelId + " not loaded in ONNX Service.");
        }

        int rows = features.length;
        int cols = features[0].length;

        // Convert double[][] to flattened float[] (ONNX typically expects Float tensors
        // for ML models)
        // Most sklearn converters export as Float32
        float[] flatFeatures = new float[rows * cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                flatFeatures[i * cols + j] = (float) features[i][j];
            }
        }

        // Create Tensor
        long[] shape = new long[] { rows, cols };

        try (OnnxTensor tensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(flatFeatures), shape)) {
            String inputName = session.getInputNames().iterator().next(); // Assumes single input

            Map<String, OnnxTensor> inputs = Collections.singletonMap(inputName, tensor);

            // Run
            try (OrtSession.Result result = session.run(inputs)) {
                // Assuming output is the first element and is a Long/Int tensor (label) or
                // Float (probability)
                // Scikit-learn ONNX usually returns [Label, Probabilities]
                // We want the Label.

                // String labelOutputName = session.getOutputNames().iterator().next();
                // Alternatively result.get(0)

                Object outputVal = result.get(0).getValue();

                if (outputVal instanceof long[]) {
                    long[] longLabels = (long[]) outputVal;
                    int[] intLabels = new int[longLabels.length];
                    for (int i = 0; i < longLabels.length; i++)
                        intLabels[i] = (int) longLabels[i];
                    return intLabels;
                } else if (outputVal instanceof int[]) {
                    return (int[]) outputVal;
                } else {
                    // Fallback for float outputs (regression or unexpected) calls
                    LogService.warn("OnnxService", "Unexpected output type: " + outputVal.getClass().getSimpleName());
                    return new int[rows]; // Error or dummy
                }
            }
        }
    }

    public void unloadModel(String modelId) {
        OrtSession session = sessionCache.remove(modelId);
        if (session != null) {
            try {
                session.close();
            } catch (OrtException e) {
                LogService.error("OnnxService", "Error closing session: " + e.getMessage());
            }
        }
    }
}
