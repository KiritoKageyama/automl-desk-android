package com.example.automl_prototype_1.model;

import java.util.List;

public class ExecutionResult {

    private String algorithmName;
    private double accuracy;
    private double aucRoc;
    private double loss;
    private long executionTimeMs;
    private int originalFeatures;
    private int selectedFeatures;
    private double precision;
    private double recall;
    private double f1Score;
    private List<Double> fitnessHistory;
    private int[][] confusionMatrix;

    public ExecutionResult() {
    }

    public void setAlgorithmName(String algorithmName) {
        this.algorithmName = algorithmName;
    }

    // Constructor for hyperparameter tuning results
    public ExecutionResult(String algorithmName, double accuracy, double aucRoc, double loss, long executionTimeMs) {
        this.algorithmName = algorithmName;
        this.accuracy = accuracy;
        this.aucRoc = aucRoc;
        this.loss = loss;
        this.executionTimeMs = executionTimeMs;
        this.originalFeatures = 0;
        this.selectedFeatures = (int) loss;
        this.precision = 0.0;
        this.recall = 0.0;
        this.f1Score = 0.0;
        this.fitnessHistory = null;
        this.confusionMatrix = null;
    }

    // Constructor for feature selection results
    public ExecutionResult(String algorithmName, double accuracy, double precision, double recall, double f1Score,
            int originalFeatures, int selectedFeatures, long executionTimeMs, List<Double> fitnessHistory,
            int[][] confusionMatrix) {
        this.algorithmName = algorithmName;
        this.accuracy = accuracy;
        this.precision = precision;
        this.recall = recall;
        this.f1Score = f1Score;
        this.originalFeatures = originalFeatures;
        this.selectedFeatures = selectedFeatures;
        this.executionTimeMs = executionTimeMs;
        this.fitnessHistory = fitnessHistory;
        this.confusionMatrix = confusionMatrix;
        this.aucRoc = Double.NaN;
        this.loss = Double.NaN;
    }

    public String getAlgorithmName() {
        return algorithmName;
    }

    public double getAccuracy() {
        return accuracy;
    }

    public double getAucRoc() {
        return aucRoc;
    }

    public double getLoss() {
        return loss;
    }

    public long getExecutionTimeMs() {
        return executionTimeMs;
    }

    public int getOriginalFeatures() {
        return originalFeatures;
    }

    public int getSelectedFeatures() {
        return selectedFeatures;
    }

    public double getPrecision() {
        return precision;
    }

    public double getRecall() {
        return recall;
    }

    public double getF1Score() {
        return f1Score;
    }

    public List<Double> getFitnessHistory() {
        return fitnessHistory;
    }

    public int[][] getConfusionMatrix() {
        return confusionMatrix;
    }

    @Override
    public String toString() {
        return "ExecutionResult{" + "algo='" + algorithmName + '\'' + ", acc=" + accuracy + '}';
    }
}