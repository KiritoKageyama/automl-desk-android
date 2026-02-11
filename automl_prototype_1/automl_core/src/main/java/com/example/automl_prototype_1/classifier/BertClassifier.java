package com.example.automl_prototype_1.classifier;

import com.example.automl_prototype_1.service.IClassifier;
import com.example.automl_prototype_1.gpu.PythonExecutionService;
import java.io.Serializable;
import java.util.HashMap;
import java.util.Map;
import java.util.List;

/**
 * Classifier that uses a Transformer model (DistilBERT) for text
 * classification.
 * Executed via Python backend (bert_finetune.py).
 */
public class BertClassifier implements IClassifier, Serializable {

    private int epochs = 3;
    private int batchSize = 16;
    private List<String> filePaths; // For text files

    // Custom Architecture Params
    private boolean trainFromScratch = false;
    private int numLayers = 6;
    private int hiddenSize = 768;
    private int numHeads = 12;

    public BertClassifier() {
    }

    public void setFilePaths(List<String> paths) {
        this.filePaths = paths;
    }

    public void setTrainFromScratch(boolean scratch) {
        this.trainFromScratch = scratch;
    }

    public void setNumLayers(int n) {
        this.numLayers = n;
    }

    public void setHiddenSize(int n) {
        this.hiddenSize = n;
    }

    public void setNumHeads(int n) {
        this.numHeads = n;
    }

    @Override
    public String getName() {
        return "BERT (Gen AI) Text Classifier";
    }

    @Override
    public void train(double[][] features, int[] labels) {
        Map<String, Object> params = new HashMap<>();
        params.put("epochs", epochs);
        params.put("batch_size", batchSize);

        // Pass Architecture Params
        params.put("scratch", trainFromScratch);
        params.put("num_layers", numLayers);
        params.put("hidden_size", hiddenSize);
        params.put("num_heads", numHeads);

        if (filePaths != null && !filePaths.isEmpty()) {
            params.put("text_paths", filePaths);
            // We pass dummy features, Python will look at 'text_paths' or infer from X if X
            // is string array
            // Since signature is double[][], we can't pass strings in X directly here.
            // But PythonExecutionService handles 'filePaths' injection if we modify it,
            // OR we pass paths in params and Python side handles it.
            // Our bert_finetune.py expects X to be passed.
            // We will rely on PythonExecutionService to handle the 'text_paths' -> 'X'
            // conversion
            // OR we send paths as a special parameter.

            // Actually, PythonExecutionService.trainModel sends 'features'.
            // For images, we used 'image_paths'.
            // For text, we should use 'text_paths'.
        }

        // We need to tell gpu_server to use 'BERT' algo type
        // This requires registering 'BERT' in gpu_server.py

        // Hack: Use CUSTOM_SCRIPT for now to avoid modifying gpu_server.py heavily?
        // No, we modify gpu_server.py for first-class support.

        // We use a custom method in PythonExecutionService or rely on a new Type.
        // Let's add "BERT" to PythonExecutionService's logic or just pass "BERT" as
        // name?
        // PythonExecutionService derives type from class name? No, it uses 'algo_type'
        // in params or infers.
        // It infers 'MLP' from MlpClassifier.
        // We should add 'BERT' inference.

        PythonExecutionService.getInstance().trainModel(this, features, labels, params);
    }

    @Override
    public int[] predict(double[][] features) {
        return PythonExecutionService.getInstance().predict(this, features);
    }
}
