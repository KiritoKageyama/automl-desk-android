package com.example.automl_prototype_1.service.agentic.inference;

import com.example.automl_prototype_1.service.IClassifier;
import com.example.automl_prototype_1.service.LogService;
import java.io.FileOutputStream;
import java.io.ObjectOutputStream;
import java.io.IOException;

/**
 * Service for exporting trained models.
 * Supports:
 * 1. Java Serialization (Native Java models like Random Forest, KNN) - Fallback
 * 2. ONNX Export (Standard Interchange Format) - via Python bridge for
 * supported models
 */
public class OnnxExportService {

    public OnnxExportService() {
    }

    /**
     * Exports the classifier to a file.
     * Attempts ONNX first if supported, otherwise falls back to Java Serialization.
     * 
     * @param classifier The trained classifier.
     * @param filepath   The destination path (e.g., "model.onnx" or "model.ser").
     * @return boolean Success status.
     */
    public boolean exportModel(IClassifier classifier, String filepath) {
        if (classifier == null) {
            LogService.error("Export", "Cannot export null classifier.");
            return false;
        }

        String algoName = classifier.getName();
        LogService.info("Export", "Exporting model: " + algoName + " to " + filepath);

        // Check availability of ONNX export for this model type
        if (isOnnxSupported(algoName)) {
            return exportToOnnx(classifier, filepath);
        } else {
            return exportToJavaSerialization(classifier, filepath);
        }
    }

    private boolean isOnnxSupported(String algoName) {
        // Currently, only Python-bridged models (XGBoost/Deep Learning) easily support
        // ONNX via existing tools.
        return algoName.contains("Python") || algoName.contains("Deep Learning") || algoName.contains("MLP")
                || algoName.contains("Visual") || algoName.contains("BERT") || algoName.contains("XGBoost")
                || algoName.contains("CNN");
    }

    private boolean exportToOnnx(IClassifier classifier, String filepath) {
        // Delegate to Python Bridge
        boolean success = com.example.automl_prototype_1.gpu.PythonExecutionService.getInstance()
                .exportModel(classifier, filepath);

        if (success) {
            LogService.info("Export", "ONNX export successful via Python Bridge.");
            return true;
        } else {
            LogService.warn("Export", "ONNX export failed via bridge. Falling back to serialization.");
            return exportToJavaSerialization(classifier, filepath + ".fallback.ser");
        }
    }

    private boolean exportToJavaSerialization(IClassifier classifier, String filepath) {
        try (FileOutputStream fileOut = new FileOutputStream(filepath);
                ObjectOutputStream out = new ObjectOutputStream(fileOut)) {
            out.writeObject(classifier);
            LogService.info("Export", "Serialized model saved successfully.");
            return true;
        } catch (IOException e) {
            LogService.error("Export", "Serialization failed: " + e.getMessage());
            return false;
        }
    }
}
