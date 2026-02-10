package com.example.automl_prototype_1.service.agentic;

import com.example.automl_prototype_1.model.Dataset;
import java.util.HashMap;
import java.util.Map;

/**
 * Service responsible for analyzing a Dataset and extracting meta-features.
 */
public class MetaFeatureExtractor {

    /**
     * Extracts meta-features from the given dataset.
     * 
     * @param dataset The dataset to analyze.
     * @return A MetaFeatures object containing the analysis results.
     */
    public MetaFeatures extract(Dataset dataset) {
        if (dataset == null || dataset.getFeatures() == null) {
            return new MetaFeatures(0, 0, 0, 0.0, 0.0);
        }

        double[][] features = dataset.getFeatures();
        int numInstances = features.length;
        int numFeatures = (numInstances > 0) ? features[0].length : 0;

        // Calculate Sparsity
        long zeroCount = 0;
        long totalElements = (long) numInstances * numFeatures;
        if (totalElements > 0) {
            for (double[] row : features) {
                for (double val : row) {
                    if (val == 0.0)
                        zeroCount++;
                }
            }
        }
        double sparsity = (totalElements > 0) ? (double) zeroCount / totalElements : 0.0;

        // Calculate Class Imbalance (if labels exist)
        int numClasses = 0;
        double imbalanceRatio = 0.0;
        int[] labels = dataset.getLabels();

        if (labels != null && labels.length > 0) {
            Map<Integer, Integer> classCounts = new HashMap<>();
            for (int label : labels) {
                classCounts.put(label, classCounts.getOrDefault(label, 0) + 1);
            }
            numClasses = classCounts.size();

            if (numClasses > 0) {
                int minCount = Integer.MAX_VALUE;
                int maxCount = Integer.MIN_VALUE;
                for (int count : classCounts.values()) {
                    if (count < minCount)
                        minCount = count;
                    if (count > maxCount)
                        maxCount = count;
                }
                imbalanceRatio = (minCount > 0) ? (double) maxCount / minCount : 0.0;
            }
        }

        // Calculate Skewness and Kurtosis (Max)
        double maxSkewness = 0.0;
        double maxKurtosis = 0.0;

        if (numInstances > 1 && numFeatures > 0) {
            for (int j = 0; j < numFeatures; j++) {
                // Collect column data
                double[] col = new double[numInstances];
                double sum = 0.0;
                for (int i = 0; i < numInstances; i++) {
                    col[i] = features[i][j];
                    sum += col[i];
                }
                double mean = sum / numInstances;

                double sumSqDiff = 0.0;
                double sumCuDiff = 0.0;
                double sumQuDiff = 0.0;

                for (double val : col) {
                    double diff = val - mean;
                    sumSqDiff += diff * diff;
                    sumCuDiff += diff * diff * diff;
                    sumQuDiff += diff * diff * diff * diff;
                }

                double variance = sumSqDiff / (numInstances - 1);
                double stdev = Math.sqrt(variance);

                if (stdev > 1e-9) {
                    double skew = (sumCuDiff / numInstances) / Math.pow(stdev, 3);
                    double kurt = (sumQuDiff / numInstances) / Math.pow(stdev, 4) - 3.0; // Excess Kurtosis

                    if (Math.abs(skew) > Math.abs(maxSkewness))
                        maxSkewness = Math.abs(skew);
                    if (Math.abs(kurt) > Math.abs(maxKurtosis))
                        maxKurtosis = Math.abs(kurt);
                }
            }
        }

        return new MetaFeatures(numInstances, numFeatures, numClasses, imbalanceRatio, sparsity, maxSkewness,
                maxKurtosis);
    }
}
