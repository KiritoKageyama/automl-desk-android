package com.example.automl_prototype_1.service.agentic.inference;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.OrtSession.SessionOptions;
import com.example.automl_prototype_1.service.LogService;

import java.nio.FloatBuffer;
import java.util.Collections;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;

/**
 * High-performance inference engine using ONNX Runtime (GPU/CPU).
 * Phase 6 Step 5.
 */
public class OnnxInferenceEngine implements IInferenceEngine {

    private final OrtEnvironment env;
    private final SessionOptions sessionOptions;
    private final Map<String, OrtSession> sessionCache = new ConcurrentHashMap<>();

    public OnnxInferenceEngine() {
        this.env = OrtEnvironment.getEnvironment();
        this.sessionOptions = new SessionOptions();
        try {
            // Try to enable CUDA
            this.sessionOptions.addCUDA(0);
            LogService.info("OnnxInference", "CUDA Provider enabled for ONNX Runtime.");
        } catch (Exception e) {
            LogService.warn("OnnxInference", "CUDA provider not available, using CPU: " + e.getMessage());
        }
    }

    /**
     * Eagerly create/cache an OrtSession for the given onnx file path.
     */
    public void preload(String onnxPath) {
        sessionCache.computeIfAbsent(onnxPath, path -> {
            try {
                return env.createSession(path, sessionOptions);
            } catch (Exception e) {
                throw new RuntimeException("Failed to load ONNX model: " + path, e);
            }
        });
        LogService.info("OnnxInference", "Preloaded session: " + onnxPath);
    }

    public void unload(String onnxPath) {
        OrtSession session = sessionCache.remove(onnxPath);
        if (session != null) {
            try {
                session.close();
            } catch (Exception e) {
                LogService.warn("OnnxInference", "Failed to close session: " + e.getMessage());
            }
        }
    }

    @Override
    public CompletableFuture<int[]> predict(String modelId, double[][] features) {
        return CompletableFuture.supplyAsync(() -> {
            try {
                OrtSession session = sessionCache.computeIfAbsent(modelId, path -> {
                    try {
                        return env.createSession(path, sessionOptions);
                    } catch (Exception e) {
                        throw new RuntimeException("Failed to load ONNX model: " + path, e);
                    }
                });

                return runInference(session, features);
            } catch (Exception e) {
                throw new RuntimeException("ONNX Inference Failed", e);
            }
        });
    }

    private int[] runInference(OrtSession session, double[][] features) throws Exception {
        int rows = features.length;
        if (rows == 0)
            return new int[0];
        int cols = features[0].length;

        FloatBuffer buffer = FloatBuffer.allocate(rows * cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                buffer.put((float) features[i][j]);
            }
        }
        buffer.flip();

        long[] shape = new long[] { rows, cols };
        try (OnnxTensor tensor = OnnxTensor.createTensor(env, buffer, shape)) {
            String inputName = session.getInputNames().iterator().next();

            try (OrtSession.Result result = session.run(Collections.singletonMap(inputName, tensor))) {
                Object value = result.get(0).getValue();
                if (value instanceof long[]) {
                    long[] longLabels = (long[]) value;
                    int[] intLabels = new int[longLabels.length];
                    for (int i = 0; i < longLabels.length; i++)
                        intLabels[i] = (int) longLabels[i];
                    return intLabels;
                } else if (value instanceof int[]) {
                    return (int[]) value;
                } else if (value instanceof float[]) {
                    LogService.warn("OnnxInference", "Received float output, expected int labels.");
                    throw new RuntimeException("Unexpected output type: float[]");
                }
                throw new RuntimeException("Unexpected output type: " + value.getClass().getName());
            }
        }
    }

    @Override
    public String getEngineName() {
        return "ONNX_RUNTIME";
    }

    @Override
    public boolean isAvailable() {
        return true;
    }
}
