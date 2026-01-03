package com.example.automl_prototype_1.registry;

import com.example.automl_prototype_1.service.IClassifier;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Central registry for managing available classifiers.
 * Allows dynamic registration and discovery of classifiers.
 */
public class ClassifierRegistry {
    private static final ClassifierRegistry instance = new ClassifierRegistry();

    public static ClassifierRegistry getInstance() {
        return instance;
    }

    private final Map<String, IClassifier> classifiers = new ConcurrentHashMap<>();

    private ClassifierRegistry() {
    }

    /**
     * Registers a new classifier.
     * 
     * @param classifier The classifier to register.
     */
    public void register(IClassifier classifier) {
        if (classifier != null) {
            classifiers.put(classifier.getName(), classifier);
            System.out.println("Registered classifier: " + classifier.getName());
        }
    }

    /**
     * Retrieves a classifier by its name.
     * 
     * @param name The name of the classifier.
     * @return The classifier instance, or null if not found.
     */
    public IClassifier getClassifier(String name) {
        boolean useGpu = com.example.automl_prototype_1.service.AppStateService.getInstance().isUseGpu();

        if (useGpu) {
            if (name.contains("KNN"))
                return new com.example.automl_prototype_1.classifier.GpuKnnClassifier();
            if (name.contains("SVM"))
                return new com.example.automl_prototype_1.classifier.GpuSvmClassifier();
            if (name.contains("Random Forest"))
                return new com.example.automl_prototype_1.classifier.GpuRandomForestClassifier();
            if (name.contains("XGBoost"))
                return new com.example.automl_prototype_1.classifier.GpuXGBoostClassifier();
        }

        // Fallback or default CPU behavior
        return classifiers.get(name);
    }

    /**
     * Returns a list of all registered classifiers.
     * 
     * @return Unmodifiable list of classifiers.
     */
    public List<IClassifier> getAllClassifiers() {
        return Collections.unmodifiableList(new ArrayList<>(classifiers.values()));
    }

    /**
     * Returns a list of all registered classifier names.
     * 
     * @return Unmodifiable list of names.
     */
    public List<String> getClassifierNames() {
        return Collections.unmodifiableList(new ArrayList<>(classifiers.keySet()));
    }
}
