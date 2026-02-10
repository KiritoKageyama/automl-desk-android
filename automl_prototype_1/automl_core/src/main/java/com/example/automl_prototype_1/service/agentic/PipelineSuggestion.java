package com.example.automl_prototype_1.service.agentic;

/**
 * Represents a suggested machine learning pipeline.
 */
public class PipelineSuggestion {
    private final String algorithmName;
    private final String classifierName;
    private String reasoning;
    private final double confidence;
    private final double empiricalScore; // New field
    private final String sotaComparison; // New field

    public PipelineSuggestion(String algorithmName, String classifierName, String reasoning, double confidence) {
        this(algorithmName, classifierName, reasoning, confidence, 0.0, "N/A");
    }

    public PipelineSuggestion(String algorithmName, String classifierName, String reasoning, double confidence,
            double empiricalScore, String sotaComparison) {
        this.algorithmName = algorithmName;
        this.classifierName = classifierName;
        this.reasoning = reasoning;
        this.confidence = confidence;
        this.empiricalScore = empiricalScore;
        this.sotaComparison = sotaComparison;
    }

    public String getAlgorithmName() {
        return algorithmName;
    }

    public String getClassifierName() {
        return classifierName;
    }

    public String getReasoning() {
        return reasoning;
    }

    public void setReasoning(String reasoning) {
        this.reasoning = reasoning;
    }

    public double getConfidence() {
        return confidence;
    }

    public double getEmpiricalScore() {
        return empiricalScore;
    }

    public String getSotaComparison() {
        return sotaComparison;
    }

    @Override
    public String toString() {
        return String.format("Suggestion: %s + %s (Confidence: %.2f, Empirical: %.2f) - %s [SOTA: %s]",
                algorithmName, classifierName, confidence, empiricalScore, reasoning, sotaComparison);
    }
}
