package com.example.automl_prototype_1.registry;

import com.example.automl_prototype_1.service.IClassifier;
import com.example.automl_prototype_1.classifier.ml.RandomForestClassifier;
import com.example.automl_prototype_1.classifier.ml.LogisticRegression;
import com.example.automl_prototype_1.classifier.ml.KnnClassifier;
import com.example.automl_prototype_1.classifier.ml.XGBoostClassifier;
import com.example.automl_prototype_1.classifier.ml.SvmClassifier;
import com.example.automl_prototype_1.classifier.dl.MlpClassifier;
import com.example.automl_prototype_1.classifier.cnn.CnnClassifier;
import com.example.automl_prototype_1.classifier.ml.GpuKnnClassifier;
import com.example.automl_prototype_1.classifier.ml.GpuLogisticRegression;
import com.example.automl_prototype_1.classifier.ml.GpuRandomForestClassifier;
import com.example.automl_prototype_1.classifier.ml.GpuPythonSvmClassifier;
import com.example.automl_prototype_1.classifier.ml.GpuPythonXGBoostClassifier;
import com.example.automl_prototype_1.classifier.ml.GpuPythonCatBoostClassifier;
import com.example.automl_prototype_1.classifier.ml.GpuPythonLightGbmClassifier;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Central registry and factory for managing available classifiers.
 * Acts as a precise factory creating the correct implementation
 * (CPU/GPU/Python) based on name.
 */
public class ClassifierRegistry {
    private static final ClassifierRegistry instance = new ClassifierRegistry();

    public static ClassifierRegistry getInstance() {
        return instance;
    }

    private ClassifierRegistry() {
    }

    /**
     * Retrieves a NEW classifier instance by its name.
     * Handles GPU switching and Python bridge selection logic.
     * 
     * @param name The name of the classifier.
     * @return The classifier instance, or null if not found.
     */
    public IClassifier getClassifier(String name) {
        if (name == null)
            return null;

        // Dynamic Model Hub / Creator Lab registrations
        try {
            java.util.Map<String, String> dynamic = com.example.automl_prototype_1.service.AppStateService
                    .getInstance().getDynamicModels();
            if (dynamic != null && dynamic.containsKey(name)) {
                return createDynamicClassifier(name, dynamic.get(name));
            }
        } catch (Exception ignored) {
        }

        return switch (name) {
            // Standard CPU
            case "Random Forest" -> new RandomForestClassifier();
            case "Logistic Regression" -> new LogisticRegression();
            case "KNN" -> new KnnClassifier();
            case "XGBoost" -> new XGBoostClassifier();
            case "SVM" -> new SvmClassifier();

            // Deep Learning
            case "Multilayer Perceptron (ANN)" -> new MlpClassifier();
            case "CNN (Deep Learning)" -> new CnnClassifier();
            case "BERT (Gen AI Text)" -> new com.example.automl_prototype_1.classifier.llm.BertClassifier();

            // GPU Wrappers (Aparapi / Smile-GPU-Stub / Native)
            case "GPU-KNN" -> new GpuKnnClassifier();

            case "GPU-Logistic Regression" -> new GpuLogisticRegression();
            case "Random Forest (GPU Accelerated)" -> new GpuRandomForestClassifier();
            case "SVM (Linear GPU)" -> new GpuPythonSvmClassifier(1.0, "linear");
            case "XGBoost (GPU Accelerated)" -> new GpuPythonXGBoostClassifier();

            // Python Bridge Wrappers
            case "XGBoost (Python/GPU Bridge)" -> new GpuPythonXGBoostClassifier();
            case "SVM (Python/GPU/RBF Bridge)" -> new GpuPythonSvmClassifier();
            case "CatBoost (Python/GPU Bridge)" -> new GpuPythonCatBoostClassifier();
            case "LightGBM (Python/GPU Bridge)" -> new GpuPythonLightGbmClassifier();
            case "LSTM (Time Series)" -> new com.example.automl_prototype_1.classifier.dl.LstmClassifier();

            // Fallbacks for fuzzy matching (kept for backward compatibility with old
            // registry logic)
            default -> {
                if (name.contains("KNN") && name.contains("GPU"))
                    yield new GpuKnnClassifier();
                if (name.contains("SVM") && name.contains("GPU"))
                    yield new GpuPythonSvmClassifier();
                yield null;
            }
        };
    }

    private IClassifier createDynamicClassifier(String displayName, String modelId) {
        if (modelId == null || modelId.isBlank()) {
            return null;
        }
        if (modelId.startsWith("CLASS:")) {
            String className = modelId.substring("CLASS:".length()).trim();
            try {
                Class<?> clazz = Class.forName(
                        "com.example.automl_prototype_1.rl.extensions." + className);
                Object instance = clazz.getDeclaredConstructor().newInstance();
                if (instance instanceof IClassifier classifier) {
                    return classifier;
                }
            } catch (ClassNotFoundException notLoaded) {
                // Compiled but not on the running classpath — use script bridge fallback.
                return new com.example.automl_prototype_1.classifier.genai.CustomScriptClassifier(
                        modelId, java.util.Map.of("class_name", className, "display_name", displayName));
            } catch (Exception e) {
                return new com.example.automl_prototype_1.classifier.genai.CustomScriptClassifier(
                        modelId, java.util.Map.of("class_name", className, "display_name", displayName,
                                "load_error", e.getMessage()));
            }
            return new com.example.automl_prototype_1.classifier.genai.CustomScriptClassifier(
                    modelId, java.util.Map.of("class_name", className, "display_name", displayName));
        }
        // HuggingFace / Model Hub ids
        return new com.example.automl_prototype_1.classifier.llm.GenericHuggingFaceClassifier(modelId, displayName);
    }

    /**
     * Returns a list of all supported classifier names.
     * 
     * @return List of names.
     */
    public List<String> getClassifierNames() {
        List<String> names = new ArrayList<>();
        // Basic (Always available CPU versions)
        names.add("KNN");
        names.add("Random Forest");
        names.add("SVM");
        names.add("XGBoost");
        names.add("Logistic Regression");

        // Deep Learning
        names.add("Multilayer Perceptron (ANN)");
        names.add("CNN (Deep Learning)");
        names.add("BERT (Gen AI Text)");

        // Advanced (GPU) - Only if GPU is ENABLED
        if (com.example.automl_prototype_1.service.AppStateService.getInstance().isUseGpu()) {
            names.add("GPU-KNN");
            names.add("GPU-Logistic Regression");
            names.add("Random Forest (GPU Accelerated)");
            names.add("XGBoost (GPU Accelerated)");
            names.add("SVM (Linear GPU)");

            // Python Bridge Wrappers (Explicit GPU Backend)
            names.add("XGBoost (Python/GPU Bridge)");
            names.add("SVM (Python/GPU/RBF Bridge)");
            names.add("CatBoost (Python/GPU Bridge)");
            names.add("LightGBM (Python/GPU Bridge)");

            // Deep Learning (Time Series)
            names.add("LSTM (Time Series)");
        }

        // Model Hub / Creator Lab dynamic registrations
        try {
            java.util.Map<String, String> dynamic = com.example.automl_prototype_1.service.AppStateService
                    .getInstance().getDynamicModels();
            if (dynamic != null) {
                for (String key : dynamic.keySet()) {
                    if (!names.contains(key)) {
                        names.add(key);
                    }
                }
            }
        } catch (Exception ignored) {
        }

        return Collections.unmodifiableList(names);
    }
}
