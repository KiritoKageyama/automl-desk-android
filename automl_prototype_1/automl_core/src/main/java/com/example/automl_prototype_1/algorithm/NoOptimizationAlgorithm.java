package com.example.automl_prototype_1.algorithm;

import java.util.Arrays;
import java.util.List;
import com.example.automl_prototype_1.model.Dataset;
import com.example.automl_prototype_1.service.IClassifier;
import com.example.automl_prototype_1.service.LogService;

/**
 * A pass-through algorithm that selects ALL features.
 * Used when Feature Selection is disabled by the user ("Skip Feature
 * Selection").
 * Extends AbstractFeatureSelectionAlgorithm to reuse the K-Fold evaluation
 * logic.
 */
public class NoOptimizationAlgorithm extends AbstractFeatureSelectionAlgorithm {

    private double[] allFeaturesSolution;
    private double fitness = 0.0;

    public NoOptimizationAlgorithm() {
        super(5, 5); // Defaults, unused
    }

    @Override
    public String getAlgorithmName() {
        return "No Optimization (Full Feature Set)";
    }

    @Override
    protected int getMaxIterations() {
        return 0; // Skip optimization loop entirely
    }

    @Override
    protected double[] getBestSolution() {
        return allFeaturesSolution;
    }

    @Override
    protected double getBestFitness() {
        return fitness;
    }

    @Override
    protected double getAverageFitness() {
        return fitness;
    }

    @Override
    protected void initialize(int numFeatures, Dataset dataset, IClassifier classifier, List<Fold> folds) {
        // Select ALL Features (1.0)
        this.allFeaturesSolution = new double[numFeatures];
        Arrays.fill(this.allFeaturesSolution, 1.0);

        // Initial evaluation (Optional, just to have a fitness value)
        // We reuse the parent's calculateFitness method
        this.fitness = calculateFitness(this.allFeaturesSolution, classifier, folds);
        LogService.info("NoOpt", String.format("Baseline Fitness (All Features): %.4f", this.fitness));
    }

    @Override
    protected void updatePopulation(int iteration, int numFeatures, Dataset dataset, IClassifier classifier,
            List<Fold> folds) {
        // Do nothing. No evolution.
    }
}
