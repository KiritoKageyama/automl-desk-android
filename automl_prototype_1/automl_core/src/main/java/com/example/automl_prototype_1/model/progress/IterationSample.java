package com.example.automl_prototype_1.model.progress;

public record IterationSample(
        String algoId,
        int trialIndex,
        int iterIndex,
        double score,
        long wallTimeMs
) {}