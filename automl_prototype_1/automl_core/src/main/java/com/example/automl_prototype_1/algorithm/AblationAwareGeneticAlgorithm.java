package com.example.automl_prototype_1.algorithm;

import com.example.automl_prototype_1.service.IClassifier;
import smile.validation.metric.Accuracy;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class AblationAwareGeneticAlgorithm extends GeneticAlgorithm {

    private final boolean useProxy;
    private final boolean useBias;

    public AblationAwareGeneticAlgorithm(int populationSize, int maxIterations, boolean useProxy, boolean useBias) {
        super(populationSize, maxIterations);
        this.useProxy = useProxy;
        this.useBias = useBias;
    }

    @Override
    protected IClassifier getSearchClassifier(IClassifier originalClassifier) {
        if (!useProxy) {
            System.out.println("[Ablation] Proxy DISABLED. Using original classifier for search.");
            return originalClassifier;
        }
        return super.getSearchClassifier(originalClassifier);
    }

    @Override
    protected double calculateFitness(double[] position, IClassifier classifier, List<Fold> folds) {
        if (!useBias) {
            // Re-implementation of fitness calculation WITHOUT the low-cardinality bonus
            try {
                // No bonus calculation here.

                double penalty = 0.0; // Penalty removed as per original code

                double totalAccuracy = 0.0;
                int searchFolds = Math.min(3, this.kFolds);

                for (int i = 0; i < searchFolds; i++) {
                    List<double[]> trainFeaturesList = new ArrayList<>();
                    List<Integer> trainLabelsList = new ArrayList<>();

                    double[][] testFeaturesWithSelection = filterFeatures(folds.get(i).features, position);
                    int[] testLabels = folds.get(i).labels;

                    if (testFeaturesWithSelection.length > 0 && testFeaturesWithSelection[0].length == 0) {
                        return 0.0;
                    }

                    for (int j = 0; j < this.kFolds; j++) {
                        if (i == j)
                            continue;
                        trainFeaturesList.addAll(Arrays.asList(filterFeatures(folds.get(j).features, position)));
                        for (int label : folds.get(j).labels) {
                            trainLabelsList.add(label);
                        }
                    }

                    if (trainFeaturesList.isEmpty() || testFeaturesWithSelection.length == 0) {
                        continue;
                    }

                    double[][] trainFeatures = trainFeaturesList.toArray(new double[0][]);
                    int[] trainLabels = trainLabelsList.stream().mapToInt(Integer::intValue).toArray();

                    classifier.train(trainFeatures, trainLabels);
                    int[] predictions = classifier.predict(testFeaturesWithSelection);
                    totalAccuracy += Accuracy.of(testLabels, predictions);
                }

                double avgAccuracy = totalAccuracy / searchFolds;
                // Return accuracy only (minus penalty if it existed), no bias
                return Math.max(0.0, avgAccuracy - penalty);

            } catch (Exception e) {
                return 0.0;
            }
        }
        // If bias is enabled, use the standard logic
        return super.calculateFitness(position, classifier, folds);
    }
}
