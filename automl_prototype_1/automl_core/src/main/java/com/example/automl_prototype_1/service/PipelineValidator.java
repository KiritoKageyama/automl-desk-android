package com.example.automl_prototype_1.service;

import com.example.automl_prototype_1.model.Dataset;
import java.util.ArrayList;
import java.util.List;

public class PipelineValidator {

    public record ValidationResult(boolean isValid, List<String> errors, List<String> warnings) {
    }

    public ValidationResult validate(Dataset dataset, String scaling, String algorithm, String classifier) {
        List<String> errors = new ArrayList<>();
        List<String> warnings = new ArrayList<>();

        if (dataset == null) {
            errors.add("No dataset selected.");
            return new ValidationResult(false, errors, warnings);
        }

        int startErrors = errors.size();

        // 1. Check Data Characteristics
        boolean hasNegatives = checkForNegativeValues(dataset);
        boolean isSmallDataset = dataset.getRecordCount() < 2000;
        int numClasses = dataset.getUniqueLabelCount(dataset.getTargetColumnName());

        // 2. Check Classifier Constraints
        if (classifier != null) {
            classifier = classifier.toLowerCase();

            // Naive Bayes (Conceptual check, though not in list currently, good practice)
            if (classifier.contains("naive bayes") && hasNegatives && !"Normalization".equals(scaling)) {
                errors.add("Naive Bayes requires non-negative data. Use Normalization (MinMax).");
            }

            // SVM / KNN Scaling
            if ((classifier.contains("svm") || classifier.contains("knn") || classifier.contains("nearest"))
                    && "None".equals(scaling)) {
                warnings.add(
                        "SVM and KNN are distance-based and perform poorly without scaling. Recommendation: Standardization.");
            }

            // GPU Compatibility
            if (classifier.contains("gpu")) {
                if (!HardwareAcceleratorService.isGpuAvailable()) {
                    errors.add("GPU Classifier selected but no GPU/CuML detected.");
                }
                if (isSmallDataset && !AppStateService.getInstance().isForceGpu()) {
                    warnings.add("Dataset is small (" + dataset.getRecordCount()
                            + " rows). GPU overhead may make this slower than CPU.");
                }
            }

            // Logistic Regression Multi-class
            if (classifier.contains("logistic") && numClasses > 2) {
                // Check if backend supports multinomial
                // Skl and cuML do, but warn about complexity
                warnings.add("Logistic Regression on Multi-class (" + numClasses
                        + ") can be slow or less accurate than Trees.");
            }
        }

        // 3. Check Algorithm Constraints
        if (algorithm != null) {
            // Heuristic checks
            if (algorithm.contains("Genetic") && dataset.getFeatureCount() > 10000) {
                warnings.add("Genetic Algorithm on extremely high features (>10k) may be slow.");
            }
        }

        return new ValidationResult(errors.size() == startErrors, errors, warnings);
    }

    private boolean checkForNegativeValues(Dataset dataset) {
        // Simple check on a sample
        try {
            double[][] sample = dataset.getFeatures(100);
            for (double[] row : sample) {
                for (double val : row) {
                    if (val < 0)
                        return true;
                }
            }
        } catch (Exception e) {
        }
        return false;
    }
}
