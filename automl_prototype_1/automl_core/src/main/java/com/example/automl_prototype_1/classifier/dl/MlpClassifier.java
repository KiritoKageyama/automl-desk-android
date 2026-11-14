package com.example.automl_prototype_1.classifier.dl;

import com.example.automl_prototype_1.service.IClassifier;
import com.example.automl_prototype_1.gpu.PythonExecutionService;
import java.util.HashMap;
import java.util.Map;
import java.io.Serializable;

/**
 * Wrapper for Scikit-Learn's MLPClassifier (Multi-Layer Perceptron),
 * executed via the Python backend.
 * Allows custom hidden layer configurations.
 */
public class MlpClassifier implements IClassifier, Serializable {

    // Default: One hidden layer with 100 neurons
    private String hiddenLayerSizes = "100";
    private String activation = "relu";
    private String solver = "adam";
    private double alpha = 0.0001;
    private int maxIter = 200;

    public MlpClassifier() {
        // Default constructor
    }

    public MlpClassifier(String hiddenLayerSizes) {
        this.hiddenLayerSizes = hiddenLayerSizes;
    }

    public MlpClassifier(String hiddenLayerSizes, String activation, String solver, double alpha, int maxIter) {
        this.hiddenLayerSizes = hiddenLayerSizes;
        this.activation = activation;
        this.solver = solver;
        this.alpha = alpha;
        this.maxIter = maxIter;
    }

    public void setHiddenLayerSizes(String hiddenLayerSizes) {
        this.hiddenLayerSizes = hiddenLayerSizes;
    }

    public String getHiddenLayerSizes() {
        return hiddenLayerSizes;
    }

    @Override
    public String getName() {
        return "MLP Neural Network (Custom)";
    }

    @Override
    public void train(double[][] features, int[] labels) {
        Map<String, Object> params = new HashMap<>();
        params.put("hidden_layer_sizes", this.hiddenLayerSizes);
        params.put("activation", this.activation);
        params.put("solver", this.solver);
        params.put("alpha", this.alpha);
        params.put("max_iter", this.maxIter);

        // Delegate to Python Backend
        // We use the "MLP" algo_type which we just added to gpu_server.py
        PythonExecutionService.getInstance().trainModel(this, features, labels, params);
    }

    @Override
    public int[] predict(double[][] features) {
        return PythonExecutionService.getInstance().predict(this, features);
    }
}
