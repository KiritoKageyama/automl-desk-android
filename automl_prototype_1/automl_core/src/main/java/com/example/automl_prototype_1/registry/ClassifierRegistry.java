package com.example.automl_prototype_1.registry;

import com.example.automl_prototype_1.service.IClassifier;
import com.example.automl_prototype_1.classifier.*;

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

            // GPU Wrappers (Aparapi / Smile-GPU-Stub / Native)
            case "GPU-KNN" -> new GpuKnnClassifier();
            case "GPU-Logistic Regression" -> new GpuLogisticRegression();
            case "Random Forest (GPU Accelerated)" -> new GpuRandomForestClassifier();
            case "SVM (Linear GPU)" -> new GpuSvmClassifier();
            case "XGBoost (GPU Accelerated)" -> new GpuXGBoostClassifier();

            // Python Bridge Wrappers
            case "XGBoost (Python/GPU Bridge)" -> new GpuXGBoostClassifier(); // Mapped to the refactored class
            case "SVM (Python/GPU/RBF Bridge)" -> new GpuPythonSvmClassifier();
            case "AdaBoost (Python)" -> new PythonAdaBoostClassifier();
            case "Gradient Boosting (Python)" -> new PythonGradientBoostingClassifier();
            case "Decision Tree (Python)" -> new PythonDecisionTreeClassifier();

            // Fallbacks for fuzzy matching (kept for backward compatibility with old
            // registry logic)
            default -> {
                if (name.contains("KNN") && name.contains("GPU"))
                    yield new GpuKnnClassifier();
                if (name.contains("SVM") && name.contains("GPU"))
                    yield new GpuSvmClassifier();
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
        // Basic
        names.add("KNN");
        names.add("Random Forest");
        names.add("SVM");
        names.add("XGBoost");
        names.add("Logistic Regression");

        // Advanced
        names.add("GPU-KNN");
        names.add("GPU-Logistic Regression");
        names.add("Random Forest (GPU Accelerated)");
        names.add("XGBoost (GPU Accelerated)");
        names.add("SVM (Linear GPU)");

        // Python
        names.add("XGBoost (Python/GPU Bridge)");
        names.add("SVM (Python/GPU/RBF Bridge)");
        names.add("AdaBoost (Python)");
        names.add("Decision Tree (Python)");

        return Collections.unmodifiableList(names);
    }
}
