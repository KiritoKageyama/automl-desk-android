package com.example.automl_prototype_1.classifier.cnn;

import com.example.automl_prototype_1.service.IClassifier;
import com.example.automl_prototype_1.gpu.PythonExecutionService;
import java.io.Serializable;
import java.util.HashMap;
import java.util.Map;
import java.io.File;

public class CnnClassifier implements IClassifier, Serializable {

    private final Map<String, Object> parameters;
    private final String scriptPath; // Path to standard cnn_algorithm.py resource

    public CnnClassifier() {
        this.parameters = new HashMap<>();
        // Default parameters
        this.parameters.put("epochs", 10);
        this.parameters.put("batch_size", 32);
        this.parameters.put("learning_rate", 0.001);

        // Resolve script path
        // We'll extract it using PythonExecutionService logic if needed,
        // but for CustomScriptClassifier logic, we need an absolute path.
        // We can use a trick: standard resource path extraction.
        try {
            File script = PythonExecutionService.getInstance().extractScriptToTemp("cnn_algorithm.py");
            this.scriptPath = script.getAbsolutePath();
        } catch (java.io.IOException e) {
            throw new RuntimeException("Failed to extract cnn_algorithm.py", e);
        }
    }

    public void setEpochs(int epochs) {
        parameters.put("epochs", epochs);
    }

    public void setBatchSize(int batchSize) {
        parameters.put("batch_size", batchSize);
    }

    public void setImagePaths(java.util.List<String> paths) {
        parameters.put("image_paths", paths);
    }

    @Override
    public void train(double[][] features, int[] labels) {
        // features might be empty for Image data if we passed paths via params?
        // Actually, PythonExecutionService expects features to be double[][].
        // For images, we might need to pass file paths.
        // But IClassifier.train takes double[][].
        // If Dataset.features is empty (custom image dataset), we need a way to pass
        // file paths.

        // Phase 2 Refactor allowed Dataset to have filePaths.
        // But IClassifier.train signature is fixed to double[][].
        // We need to overload PythonExecutionService.trainModel to accept file paths?
        // Or we encode file paths into double[][]? No, that's hacky.

        // Solution: Pass file paths via parameters if features are empty/null.
        PythonExecutionService.getInstance().trainModel(this, features, labels, parameters);
    }

    @Override
    public int[] predict(double[][] features) {
        return PythonExecutionService.getInstance().predict(this, features);
    }

    // helper to get script path for the service
    public String getScriptPath() {
        return scriptPath;
    }

    @Override
    public String getName() {
        return "Convolutional Neural Network (CNN)";
    }
}
