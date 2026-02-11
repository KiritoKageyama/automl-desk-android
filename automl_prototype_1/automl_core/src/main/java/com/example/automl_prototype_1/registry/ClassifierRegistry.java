package com.example.automl_prototype_1.registry;

import com.example.automl_prototype_1.service.IClassifier;
import com.example.automl_prototype_1.classifier.cpu.RandomForestClassifier;
import com.example.automl_prototype_1.classifier.cpu.LogisticRegression;
import com.example.automl_prototype_1.classifier.cpu.KnnClassifier;
import com.example.automl_prototype_1.classifier.cpu.XGBoostClassifier;
import com.example.automl_prototype_1.classifier.cpu.SvmClassifier;
import com.example.automl_prototype_1.classifier.MlpClassifier;
import com.example.automl_prototype_1.classifier.CnnClassifier;
import com.example.automl_prototype_1.classifier.gpu.GpuKnnClassifier;
import com.example.automl_prototype_1.classifier.gpu.GpuLogisticRegression;
import com.example.automl_prototype_1.classifier.gpu.GpuRandomForestClassifier;
import com.example.automl_prototype_1.classifier.gpu.GpuPythonSvmClassifier;
import com.example.automl_prototype_1.classifier.gpu.GpuPythonXGBoostClassifier;
import com.example.automl_prototype_1.classifier.gpu.GpuPythonCatBoostClassifier;
import com.example.automl_prototype_1.classifier.gpu.GpuPythonLightGbmClassifier;

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
            case "BERT (Gen AI Text)" -> new com.example.automl_prototype_1.classifier.BertClassifier();

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
        }

        return Collections.unmodifiableList(names);
    }
}
