package com.example.automl_prototype_1.algorithm;

import com.example.automl_prototype_1.model.Dataset;
import com.example.automl_prototype_1.model.ExecutionResult;
import com.example.automl_prototype_1.service.AppStateService;
import com.example.automl_prototype_1.service.IClassifier;
import com.example.automl_prototype_1.service.LogService;
import smile.validation.metric.Accuracy;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * An "Algorithm" that simply uses the features manually selected by the user.
 * It bypasses the optimization loop.
 */
public class ManualSelectionAlgorithm implements OptimizationAlgorithm {

    @Override
    public String getAlgorithmName() {
        return "Manual Selection";
    }

    @Override
    public ExecutionResult execute(Dataset dataset, Map<String, Object> configuration) {
        // This should not be called directly if we want trial support, but optimization
        // relies on it.
        // We will just call optimize logic here
        return optimize(dataset, (IClassifier) configuration.get("classifier"), configuration, 1, 1);
    }

    @Override
    public ExecutionResult optimize(Dataset dataset, IClassifier classifier, Map<String, Object> configuration,
            int trialNum, int totalTrials) {
        long startTime = System.currentTimeMillis();
        LogService.info("ManualSelection", "Starting execution with user-selected features...");

        // 1. Get Selected Indices from AppState
        List<Integer> selectedIndices = AppStateService.getInstance().getManualFeatureIndices();
        if (selectedIndices == null || selectedIndices.isEmpty()) {
            LogService.warn("ManualSelection", "No features selected! Defaulting to ALL features.");
            // Default to all
            int numFeatures = dataset.getFeatureCount();
            selectedIndices = new ArrayList<>();
            for (int i = 0; i < numFeatures; i++)
                selectedIndices.add(i);
        }

        double[][] X = dataset.getFeatures();
        int[] y = dataset.getLabelVector();
        int numRows = X.length;
        int numSelected = selectedIndices.size();

        // 2. Filter Dataset
        double[][] X_filtered = new double[numRows][numSelected];
        for (int i = 0; i < numRows; i++) {
            for (int j = 0; j < numSelected; j++) {
                int originalIndex = selectedIndices.get(j);
                if (originalIndex < X[0].length) {
                    X_filtered[i][j] = X[i][originalIndex];
                }
            }
        }

        // 3. Train & Evaluate (Simple Split for now, or use the Algo's robust CV?)
        // To be consistent with other algos, we should ideally use CV.
        // But for simplicity in this phase, let's do a simple 80/20 split or 5-fold CV
        // manually.
        // Let's do a simple 5-fold CV to return a valid metric.

        int k = 5;
        double totalAcc = 0;
        int foldSize = numRows / k;

        for (int i = 0; i < k; i++) {
            int start = i * foldSize;
            int end = (i == k - 1) ? numRows : (i + 1) * foldSize;
            int testSize = end - start;
            int trainSize = numRows - testSize;

            double[][] trainFeatures = new double[trainSize][numSelected];
            int[] trainLabels = new int[trainSize];
            double[][] testFeatures = new double[testSize][numSelected];
            int[] testLabels = new int[testSize];

            int trainIdx = 0;
            int testIdx = 0;

            for (int r = 0; r < numRows; r++) {
                if (r >= start && r < end) {
                    testFeatures[testIdx] = X_filtered[r];
                    testLabels[testIdx] = y[r];
                    testIdx++;
                } else {
                    trainFeatures[trainIdx] = X_filtered[r];
                    trainLabels[trainIdx] = y[r];
                    trainIdx++;
                }
            }

            classifier.train(trainFeatures, trainLabels);
            int[] preds = classifier.predict(testFeatures);
            totalAcc += Accuracy.of(testLabels, preds);
        }

        double finalAcc = totalAcc / k;
        long endTime = System.currentTimeMillis();

        ExecutionResult result = new ExecutionResult(
                getAlgorithmName(),
                finalAcc,
                0.0, 0.0, 0.0, // Precision/Recall/F1 omitted for brevity
                dataset.getFeatureCount(),
                numSelected,
                endTime - startTime,
                new ArrayList<>(), // No fitness history
                new ArrayList<>(),
                new ArrayList<>(),
                new int[0][0] // No confusion matrix
        );

        result.setFeatureIndices(selectedIndices.stream().mapToInt(i -> i).toArray());

        LogService.info("ManualSelection", "Completed. Accuracy: " + finalAcc);
        return result;
    }
}
