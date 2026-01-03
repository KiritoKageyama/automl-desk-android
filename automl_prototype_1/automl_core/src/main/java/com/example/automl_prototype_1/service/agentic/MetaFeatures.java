package com.example.automl_prototype_1.service.agentic;

/**
 * Data Transfer Object containing meta-features of a dataset.
 * These features are used by the AgenticAdvisor to suggest pipelines.
 */
public class MetaFeatures {
    private final int numInstances;
    private final int numFeatures;
    private final int numClasses;
    private final double classImbalanceRatio;
    private final double sparsity;

    public MetaFeatures(int numInstances, int numFeatures, int numClasses, double classImbalanceRatio,
            double sparsity) {
        this.numInstances = numInstances;
        this.numFeatures = numFeatures;
        this.numClasses = numClasses;
        this.classImbalanceRatio = classImbalanceRatio;
        this.sparsity = sparsity;
    }

    public int getNumInstances() {
        return numInstances;
    }

    public int getNumFeatures() {
        return numFeatures;
    }

    public int getNumClasses() {
        return numClasses;
    }

    public double getClassImbalanceRatio() {
        return classImbalanceRatio;
    }

    public double getSparsity() {
        return sparsity;
    }

    @Override
    public String toString() {
        return String.format("MetaFeatures{instances=%d, features=%d, classes=%d, imbalance=%.2f, sparsity=%.2f}",
                numInstances, numFeatures, numClasses, classImbalanceRatio, sparsity);
    }
}
