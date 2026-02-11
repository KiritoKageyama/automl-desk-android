package com.example.automl_prototype_1.classifier;

import com.example.automl_prototype_1.service.IClassifier;
import com.example.automl_prototype_1.gpu.PythonExecutionService;
import java.io.Serializable;
import java.util.HashMap;
import java.util.Map;

public class CustomScriptClassifier implements IClassifier, Serializable {

    private final String scriptPath;
    private final Map<String, Object> parameters;

    public CustomScriptClassifier(String scriptPath) {
        this.scriptPath = scriptPath;
        this.parameters = new HashMap<>();
    }

    public CustomScriptClassifier(String scriptPath, Map<String, Object> parameters) {
        this.scriptPath = scriptPath;
        this.parameters = parameters != null ? parameters : new HashMap<>();
    }

    public void setParameter(String key, Object value) {
        parameters.put(key, value);
    }

    public String getScriptPath() {
        return scriptPath;
    }

    @Override
    public void train(double[][] features, int[] labels) {
        PythonExecutionService.getInstance().trainModel(this, features, labels, parameters);
    }

    @Override
    public int[] predict(double[][] features) {
        return PythonExecutionService.getInstance().predict(this, features);
    }

    @Override
    public String getName() {
        return "Custom Script (" + new java.io.File(scriptPath).getName() + ")";
    }
}
