package com.example.automl_prototype_1.classifier.dl;

import com.example.automl_prototype_1.service.IClassifier;
import com.example.automl_prototype_1.gpu.PythonExecutionService;
import java.io.Serializable;
import java.util.HashMap;
import java.util.Map;

/**
 * LSTM Classifier (Long Short-Term Memory).
 * Designed for Time Series or Sequential Data.
 * Wraps the Python/PyTorch implementation via GpuServer.
 */
public class LstmClassifier implements IClassifier, Serializable {

    private final Map<String, Object> parameters;

    public LstmClassifier() {
        this.parameters = new HashMap<>();
        // Default parameters
        this.parameters.put("algo_type", "LSTM"); // Critical for GPU Server routing
        this.parameters.put("epochs", 50);
        this.parameters.put("batch_size", 32);
        this.parameters.put("learning_rate", 0.001);
        this.parameters.put("window_size", 10); // Default window size
        this.parameters.put("hidden_dim", 64);
        this.parameters.put("num_layers", 2);
        this.parameters.put("dropout", 0.2);
    }

    public void setWindowSize(int windowSize) {
        parameters.put("window_size", windowSize);
    }

    public void setEpochs(int epochs) {
        parameters.put("epochs", epochs);
    }

    public void setHiddenDim(int dim) {
        parameters.put("hidden_dim", dim);
    }

    @Override
    public void train(double[][] features, int[] labels) {
        // Delegate to Python GPU Server
        // The server will handle "algo_type": "LSTM"
        PythonExecutionService.getInstance().trainModel(this, features, labels, parameters);
    }

    @Override
    public int[] predict(double[][] features) {
        return PythonExecutionService.getInstance().predict(this, features);
    }

    public Map<String, Object> getParameters() {
        return parameters;
    }

    @Override
    public String getName() {
        return "LSTM (Time Series)";
    }
}
