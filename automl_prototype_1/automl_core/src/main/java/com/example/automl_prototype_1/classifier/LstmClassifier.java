package com.example.automl_prototype_1.classifier;

import com.example.automl_prototype_1.service.IClassifier;
import com.example.automl_prototype_1.gpu.PythonExecutionService;
import java.io.Serializable;
import java.util.HashMap;
import java.util.Map;

/**
 * LSTM Classifier for Time Series data.
 * Wraps a Python-based LSTM model (Keras/PyTorch).
 */
public class LstmClassifier implements IClassifier, Serializable {

    // Hyperparameters
    private int units = 50;
    private int epochs = 10;
    private int batchSize = 32;
    private boolean bidirectional = false;

    public LstmClassifier() {
    }

    public void setUnits(int units) {
        this.units = units;
    }

    public void setEpochs(int epochs) {
        this.epochs = epochs;
    }

    public void setBatchSize(int batchSize) {
        this.batchSize = batchSize;
    }

    public void setBidirectional(boolean bidirectional) {
        this.bidirectional = bidirectional;
    }

    @Override
    public String getName() {
        return "LSTM (Time Series) Classifier";
    }

    @Override
    public void train(double[][] features, int[] labels) {
        Map<String, Object> params = new HashMap<>();
        params.put("units", units);
        params.put("epochs", epochs);
        params.put("batch_size", batchSize);
        params.put("bidirectional", bidirectional);

        // Algo Type "LSTM" must be handled in PythonExecutionService -> GpuServer
        PythonExecutionService.getInstance().trainModel(this, features, labels, params);
    }

    @Override
    public int[] predict(double[][] features) {
        return PythonExecutionService.getInstance().predict(this, features);
    }
}
