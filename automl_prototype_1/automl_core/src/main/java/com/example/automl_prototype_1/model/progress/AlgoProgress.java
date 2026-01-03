package com.example.automl_prototype_1.model.progress;

public record AlgoProgress(
        String algoId,
        Phase phase,
        int currentTrial,
        int totalTrials,
        int currentIter,
        int totalIters,
        double bestScore,
        int bestTrial,
        int bestIter,
        long etaMs
) {}