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

        return new MetaFeatures(numInstances, numFeatures, numClasses, imbalanceRatio, sparsity);
    }
}
