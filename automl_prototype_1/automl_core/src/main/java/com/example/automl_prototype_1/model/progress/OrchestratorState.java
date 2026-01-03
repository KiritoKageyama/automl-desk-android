package com.example.automl_prototype_1.model.progress;

import java.util.List;

public record OrchestratorState(
        AlgoProgress currentRun,
        List<AlgoProgress> queue,
        String globalBestAlgo,
        double globalBestScore,
        int globalBestTrial, // <-- ADD THIS
        int globalBestIter   // <-- ADD THIS
) {}