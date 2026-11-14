package com.example.automl_prototype_1.service.pipeline;

import com.example.automl_prototype_1.service.IClassifier;

import com.example.automl_prototype_1.algorithm.*;
import com.example.automl_prototype_1.model.Dataset;
import com.example.automl_prototype_1.model.ExecutionResult;
import com.example.automl_prototype_1.registry.AlgorithmRegistry;

public class HyperparameterTuningService {

    public HyperparameterTuningService() {
    }

    public ExecutionResult tune(String algorithmName, int trials, Dataset dataset, IClassifier classifier) {
        System.out.println("--- Starting Hyperparameter Tuning for " + algorithmName + " ---");
        ExecutionResult bestResult = null;

        for (int i = 0; i < trials; i++) {
            System.out.println("--- Trial " + (i + 1) + "/" + trials + " ---");
            // Heartbeat between trials so multi-trial tuning is not flagged stuck
            com.example.automl_prototype_1.service.agentic.TrainingWatchdogService.getInstance().heartbeat();

            OptimizationAlgorithm algorithmInstance = AlgorithmRegistry.getInstance().createRandomized(algorithmName);
            if (algorithmInstance == null) {
                System.err.println("Cannot tune unknown algorithm: " + algorithmName);
                break;
            }

            // Call optimize directly to pass correct trial information for progress
            // tracking
            ExecutionResult currentResult = algorithmInstance.optimize(dataset, classifier, i + 1, trials);

            if (bestResult == null || currentResult.getAccuracy() > bestResult.getAccuracy()) {
                bestResult = currentResult;
                System.out.println(
                        "*** New Best Accuracy Found: " + bestResult.getAccuracy() + " from trial " + (i + 1) + " ***");
            }
        }

        System.out.println("--- Tuning Finished for " + algorithmName + ". Best Result: " + bestResult + " ---");
        // Modify the name to indicate it was a tuned result
        if (bestResult != null) {
            bestResult.setAlgorithmName(bestResult.getAlgorithmName() + " (Tuned)");
        }
        return bestResult;
    }
}