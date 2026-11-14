package com.example.automl_prototype_1.classifier.llm;

import com.example.automl_prototype_1.service.IClassifier;
import com.example.automl_prototype_1.gpu.PythonExecutionService;
import java.io.Serializable;
import java.util.HashMap;
import java.util.Map;
import java.util.List;

/**
 * A generic wrapper for any Hugging Face model.
 */
public class GenericHuggingFaceClassifier implements IClassifier, Serializable {

    private String modelId; // e.g., "facebook/bart-large"
    private String customName; // e.g., "My BART Model"
    private int epochs = 3;
    private int batchSize = 8;
    private List<String> filePaths;

    public GenericHuggingFaceClassifier(String modelId, String customName) {
        this.modelId = modelId;
        this.customName = customName;
    }

    // No-arg constructor for serialization
    public GenericHuggingFaceClassifier() {
    }

    public void setFilePaths(List<String> paths) {
        this.filePaths = paths;
    }

    @Override
    public String getName() {
        return customName != null ? customName : modelId;
    }

    @Override
    public void train(double[][] features, int[] labels) {
        Map<String, Object> params = new HashMap<>();
        params.put("epochs", epochs);
        params.put("batch_size", batchSize);
        params.put("hf_model_id", modelId); // Start passing the ID

        // Use the BERT algo type for now, but bert_finetune.py will handle the ID
        // Or we might need a new algoType "HF_TRANSFORMER"
        params.put("algo_type", "HF_TRANSFORMER");

        if (filePaths != null && !filePaths.isEmpty()) {
            params.put("text_paths", filePaths);
        }

        // We use trainModel. PythonExecutionService needs to handle "HF_TRANSFORMER"
        // mapping logic
        // if we rely on class-based mapping, OR we explicitly pass algo_type.
        PythonExecutionService.getInstance().trainModel(this, features, labels, params);
    }

    @Override
    public int[] predict(double[][] features) {
        return PythonExecutionService.getInstance().predict(this, features);
    }

    public String getModelId() {
        return modelId;
    }
}
