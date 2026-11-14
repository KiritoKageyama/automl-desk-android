package com.example.automl_prototype_1.gpu;

import com.example.automl_prototype_1.service.IClassifier;

import java.util.HashMap;
import java.util.Map;

/**
 * Maps Java classifiers to GPU-server algo types. Extracted for unit testing.
 */
public final class AlgoTypeResolver {

    private AlgoTypeResolver() {
    }

    public static Resolved resolve(IClassifier classifier, Map<String, Object> params) {
        Map<String, Object> out = params != null ? new HashMap<>(params) : new HashMap<>();
        String algoType;

        if (classifier instanceof com.example.automl_prototype_1.classifier.dl.MlpClassifier) {
            algoType = "MLP";
        } else if (classifier != null && classifier.getName() != null && classifier.getName().contains("Random Forest")) {
            algoType = "RF";
        } else if (classifier instanceof com.example.automl_prototype_1.classifier.genai.CustomScriptClassifier cs) {
            algoType = "CUSTOM_SCRIPT";
            String script = cs.getScriptPath();
            if (!ScriptAllowlist.isAllowed(script)) {
                throw new IllegalArgumentException(
                        "CUSTOM_SCRIPT blocked (path not allowlisted): " + script
                                + ". Place scripts under ~/.automl/scripts or project python resources.");
            }
            out.put("script_path", script);
        } else if (classifier instanceof com.example.automl_prototype_1.classifier.cnn.CnnClassifier cnn) {
            algoType = "CNN";
            if (cnn.getScriptPath() != null) {
                out.put("script_path", cnn.getScriptPath());
            }
        } else if (classifier instanceof com.example.automl_prototype_1.classifier.llm.GenericHuggingFaceClassifier hf) {
            algoType = "HF_TRANSFORMER";
            if (hf.getModelId() != null) {
                out.put("hf_model_id", hf.getModelId());
            }
        } else if (classifier instanceof com.example.automl_prototype_1.classifier.llm.BertClassifier) {
            algoType = "BERT";
        } else if (classifier instanceof com.example.automl_prototype_1.classifier.dl.LstmClassifier) {
            algoType = "LSTM";
        } else if (classifier instanceof com.example.automl_prototype_1.classifier.ml.GpuPythonCatBoostClassifier) {
            algoType = "CATBOOST";
        } else {
            algoType = "RF";
        }

        return new Resolved(algoType, out);
    }

    public record Resolved(String algoType, Map<String, Object> params) {
    }
}
