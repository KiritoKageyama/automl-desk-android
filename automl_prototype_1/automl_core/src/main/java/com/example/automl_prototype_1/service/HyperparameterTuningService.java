package com.example.automl_prototype_1.service;

import com.example.automl_prototype_1.algorithm.*;
import com.example.automl_prototype_1.model.Dataset;
import com.example.automl_prototype_1.model.ExecutionResult;
import java.util.concurrent.ThreadLocalRandom;

public class HyperparameterTuningService {

    public HyperparameterTuningService() {
    }

    public ExecutionResult tune(String algorithmName, int trials, Dataset dataset, IClassifier classifier) {
        System.out.println("--- Starting Hyperparameter Tuning for " + algorithmName + " ---");
        ExecutionResult bestResult = null;

        for (int i = 0; i < trials; i++) {
            System.out.println("--- Trial " + (i + 1) + "/" + trials + " ---");
            OptimizationAlgorithm algorithmInstance = createRandomizedAlgorithm(algorithmName);
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

    private OptimizationAlgorithm createRandomizedAlgorithm(String name) {
        var rand = ThreadLocalRandom.current();

        int popSize = rand.nextInt(20, 51); // populationSize between 20-50
        int maxIter = rand.nextInt(50, 151); // maxIterations between 50-150

        return switch (name) {
            case "Genetic Algorithm", "GA" -> new GeneticAlgorithm(popSize, maxIter);
            case "BPSO" -> new BinaryParticleSwarmOptimization(popSize, maxIter);
            case "WWO" -> new WaterWaveOptimization(popSize, maxIter);
            case "ASO" -> new AtomSearchOptimization(popSize, maxIter);
            case "IGPSO" -> new IGPSO(popSize, maxIter);
            case "Simulated Annealing", "SA" -> {
                double initialTemp = rand.nextDouble(0.8, 1.2);
                double coolingRate = rand.nextDouble(0.99, 0.999);
                yield new SimulatedAnnealing(maxIter * 5, initialTemp, coolingRate);
            }
            case "GWO", "Grey Wolf Optimizer" -> new GreyWolfOptimizer(popSize, maxIter);
            case "DE", "Differential Evolution" -> new DifferentialEvolution(popSize, maxIter);
            case "PLS", "Parallel Local Search" -> new ParallelLocalSearchAlgorithm(maxIter);
            default -> null;
        };
    }
}