package com.example.automl_prototype_1.service;

import com.example.automl_prototype_1.model.*;

import java.util.*;
import java.util.stream.Collectors;

public class SimulationService {

    private static SimulationService instance;
    private final PreprocessorService preprocessorService = new PreprocessorService();

    private SimulationService() {
    }

    public static synchronized SimulationService getInstance() {
        if (instance == null) {
            instance = new SimulationService();
        }
        return instance;
    }

    /**
     * Simulates the preprocessing pipeline on the first few rows of the dataset.
     */
    public TransformationPreview simulatePreprocessing(Dataset dataset, String scalingStrategy) {
        if (dataset == null)
            return new TransformationPreview(new ArrayList<>(), "No dataset selected.");

        // Take a sample (first 10 rows)
        int sampleSize = Math.min(10, dataset.getRecords().size());

        // We need a dummy dataset for the preprocessor specific to this sample,
        // OR we can run preprocessor on the whole thing and take a sample.
        // Running on whole is more accurate for Mean/Std calculation.
        // Let's assume we run on the whole dataset to get the "Real" stats, then return
        // sample.

        // NOTE: This might be heavy for very large datasets, but for a simulator on a
        // desktop app it's usually fine.
        // If optimization needed: Calculate stats on sample only. But user wants "Real
        // Time Calculator".

        // Create config from legacy string
        PreprocessingConfig config = new PreprocessingConfig();
        config.setScalingStrategy(scalingStrategy);
        config.setAutoRemoveIdColumns(true); // Default for simulation
        config.setImputeMissingValues(true);
        config.setRemoveOutliers(false);

        Dataset processedDataset = preprocessorService.runPreprocessingPipeline(dataset, "Class", config);

        List<Map<String, String>> processedSample = processedDataset.getRecords().stream()
                .limit(sampleSize)
                .collect(Collectors.toList());

        String explanation = "Applied " + scalingStrategy + ".\n";
        if ("Standardization".equals(scalingStrategy)) {
            explanation += "Formula: z = (x - mean) / std_dev\n";
            explanation += "Calculated Z-Scores for all numeric columns to center around 0.";
        } else if ("Normalization".equals(scalingStrategy)) {
            explanation += "Formula: x_norm = (x - min) / (max - min)\n";
            explanation += "Scaled all numeric columns to range [0, 1].";
        } else {
            explanation += "No scaling applied. Original values retained.";
        }

        return new TransformationPreview(processedSample, explanation);
    }

    public AlgoSimulationData getAlgorithmLogic(String algoName) {
        if (algoName.contains("PSO") || algoName.contains("Particle")) {
            return new AlgoSimulationData(
                    "Binary PSO",
                    "v[i] = w*v[i] + c1*r1*(pBest[i]-x[i]) + c2*r2*(gBest[i]-x[i])",
                    "Updates velocity of particles based on personal and global best positions. Positions are binary discretized via Sigmoid.",
                    "Feature Mask: [1, 0, 1, 1, 0, ...]");
        } else if (algoName.contains("Genetic") || algoName.contains("GA")) {
            return new AlgoSimulationData(
                    "Genetic Algorithm",
                    "Child = Crossover(Parent1, Parent2) + Mutation(Rate)",
                    "Evolves a population of feature subsets using selection, crossover, and mutation operators.",
                    "Chromosome: [0, 1, 0, 1, 1, ...]");
        } else if (algoName.contains("Simulated Annealing") || algoName.contains("SA")) {
            return new AlgoSimulationData(
                    "Simulated Annealing",
                    "P(accept) = exp((fitness_new - fitness_current) / Temp)",
                    "Probabilistic search that accepts worse solutions early on to escape local optima, cooling down over time.",
                    "State: [1, 1, 0, 0, 1, ...]");
        } else {
            // Default / Other
            return new AlgoSimulationData(
                    algoName,
                    "Generic Optimization Formula",
                    "Iterative search for best objective function value.",
                    "Vector: [1, 0, ...]");
        }
    }

    public ClassifierSimulationData getClassifierLogic(String classifierName) {
        return getClassifierLogic(classifierName, null);
    }

    public ClassifierSimulationData getClassifierLogic(String classifierName, List<Map<String, String>> sampleData) {
        if (sampleData == null || sampleData.isEmpty()) {
            return new ClassifierSimulationData(classifierName, "Probability(y|x)", "Minimize Cross-Entropy",
                    "Class: A (Prob: 0.85)");
        }

        // Parse first two rows for demo
        try {
            Map<String, String> row1 = sampleData.get(0);
            // If we have a second row, we can do distance
            Map<String, String> row2 = sampleData.size() > 1 ? sampleData.get(1) : row1;

            if (classifierName.contains("KNN")) {
                double dist = 0.0;
                StringBuilder formulaStep = new StringBuilder("d(Row1, Row2) = sqrt(");
                int count = 0;

                for (String key : row1.keySet()) {
                    if (key.equals("Class"))
                        continue; // Skip target
                    try {
                        double v1 = Double.parseDouble(row1.get(key));
                        double v2 = Double.parseDouble(row2.get(key));
                        double diff = v1 - v2;
                        dist += diff * diff;

                        if (count < 3) { // Show first 3 dims
                            formulaStep.append(String.format("(%.2f - %.2f)^2 + ", v1, v2));
                        }
                        count++;
                    } catch (NumberFormatException ignored) {
                    }
                }
                if (count > 3)
                    formulaStep.append("...");
                formulaStep.append(")");

                double finalDist = Math.sqrt(dist);

                return new ClassifierSimulationData(
                        "K-Nearest Neighbors",
                        "d(p, q) = sqrt(sum((p_i - q_i)^2))",
                        formulaStep.toString(),
                        String.format("Distance: %.4f (Row 1 is %.4f away from Row 2)", finalDist, finalDist));
            } else if (classifierName.contains("SVM")) {
                double dotProd = 0.0;
                StringBuilder formulaStep = new StringBuilder("f(x) = ");
                int count = 0;
                Random rnd = new Random(42); // Deterministic for consistent UI

                for (String key : row1.keySet()) {
                    if (key.equals("Class"))
                        continue;
                    try {
                        double x = Double.parseDouble(row1.get(key));
                        double w = rnd.nextDouble() - 0.5; // Random weight [-0.5, 0.5]
                        dotProd += x * w;

                        if (count < 3) {
                            formulaStep.append(String.format("(%.2f * %.2f) + ", w, x));
                        }
                        count++;
                    } catch (NumberFormatException ignored) {
                    }
                }
                if (count > 3)
                    formulaStep.append("...");
                double bias = 0.1;
                dotProd += bias;
                formulaStep.append(String.format(" + %.2f (bias)", bias));

                String predictedClass = dotProd >= 0 ? "1" : "-1";

                return new ClassifierSimulationData(
                        "Support Vector Machine",
                        "f(x) = sign(w * x + b)",
                        formulaStep.toString(),
                        String.format("Output: %.4f -> Class %s (Distance to Hyperplane)", dotProd, predictedClass));
            } else if (classifierName.contains("Logistic")) {
                double z = 0.0;
                Random rnd = new Random(42);

                for (String key : row1.keySet()) {
                    if (key.equals("Class"))
                        continue;
                    try {
                        double x = Double.parseDouble(row1.get(key));
                        double w = rnd.nextDouble() - 0.5;
                        z += x * w;
                    } catch (NumberFormatException ignored) {
                    }
                }

                double prob = 1.0 / (1.0 + Math.exp(-z));

                return new ClassifierSimulationData(
                        "Logistic Regression",
                        "P(y=1|x) = 1 / (1 + e^-z)",
                        String.format("z = %.4f (linear combination)", z),
                        String.format("Probability: %.2f%% (Class %s)", prob * 100, prob > 0.5 ? "1" : "0"));
            }
        } catch (Exception e) {
            return new ClassifierSimulationData(classifierName, "Error calc", "Error", e.getMessage());
        }

        // Fallback
        return new ClassifierSimulationData(
                classifierName,
                "Probability(y|x)",
                "Minimize Cross-Entropy",
                "Class: A (Prob: 0.85)");
    }

    public TuningSimulationData getTuningLogic() {
        // dynamic link to real progress
        var snapshot = ProgressTrackingService.getInstance().snapshot();
        var current = snapshot.currentRun();

        if (current != null && current.phase() == com.example.automl_prototype_1.model.progress.Phase.RUNNING) {
            String trace = String.format("Status: RUNNING\nTrial: %d/%d\nIteration: %d/%d\nBest Score: %.4f",
                    current.currentTrial(), current.totalTrials(),
                    current.currentIter(), current.totalIters(),
                    current.bestScore());

            return new TuningSimulationData(
                    "Active Optimization (" + current.algoId() + ")",
                    "Live Execution Monitor",
                    "Maximize Accuracy",
                    trace);
        } else if (snapshot.globalBestScore() > 0) {
            // Show result of last run
            String trace = String.format("Status: COMPLETED\nTop Algorithm: %s\nBest Score: %.4f\nFound at: Trial %d",
                    snapshot.globalBestAlgo(), snapshot.globalBestScore(), snapshot.globalBestTrial());

            return new TuningSimulationData(
                    "Optimization Report",
                    "Post-Run Analysis",
                    "Best Result Found",
                    trace);
        }

        return new TuningSimulationData(
                "Monte Carlo Search (Random)",
                "PopSize ~ U(10, 50), MaxIter ~ U(50, 1000)",
                "Maximize Accuracy on Validation Fold",
                "Trial 1: Pop=20, Acc=0.85\nTrial 2: Pop=45, Acc=0.91 (New Best)");
    }

    /**
     * Estimates execution time based on dataset size, algorithm complexity, and
     * hardware.
     */
    public String getTimeEstimation(String algo, String classifier, int rows, int cols, boolean hasGpu) {
        // Base complexity factors (Abstract Units)
        double algoFactor = 1.0;
        if (algo.contains("Genetic"))
            algoFactor = 2.5; // Populations
        if (algo.contains("PSO"))
            algoFactor = 2.0;
        if (algo.contains("Annealing"))
            algoFactor = 1.5;

        double classifierFactor = 1.0;
        if (classifier.contains("SVM"))
            classifierFactor = 3.0; // O(n^3) usually
        if (classifier.contains("Forest"))
            classifierFactor = 2.0;
        if (classifier.contains("XGBoost"))
            classifierFactor = 1.5;
        if (classifier.contains("KNN"))
            classifierFactor = 1.2;

        // Data Volume
        long volume = (long) rows * cols;

        // Setup Base Time (CPU) in seconds
        // Heuristic: 1M cells takes ~10 seconds for base algo/classifier
        double baseTimeSeconds = (volume / 100000.0) * algoFactor * classifierFactor;

        // GPU Speedup Factor
        double speedup = 1.0;
        boolean gpuApplicable = classifier.contains("GPU") || classifier.contains("Aparapi");

        if (hasGpu && gpuApplicable) {
            if (classifier.contains("SVM") && classifier.contains("Python")) {
                speedup = 50.0; // Huge speedup for SVM
            } else if (classifier.contains("XGBoost")) {
                speedup = 20.0;
            } else if (classifier.contains("Forest")) {
                speedup = 15.0;
            } else {
                speedup = 5.0; // Basic Aparapi
            }
        }

        double finalTime = baseTimeSeconds / speedup;
        String timeStr = formatTime(finalTime);

        StringBuilder sb = new StringBuilder();
        sb.append(String.format("Estimated Runtime: %s\n", timeStr));
        if (hasGpu && gpuApplicable) {
            sb.append(String.format("Hardware: NVIDIA RTX (Active) - %.1fx Speedup\n", speedup));
        } else if (hasGpu && !gpuApplicable) {
            sb.append("Hardware: CPU (GPU Available but not used for this classifier)\n");
        } else {
            sb.append("Hardware: CPU (Standard)\n");
        }

        // Power/Cost (Abstract)
        double kwh = (hasGpu ? 0.35 : 0.15) * (finalTime / 3600.0); // 350W GPU vs 150W CPU
        sb.append(String.format("Energy Est: %.5f kWh", kwh));

        return sb.toString();
    }

    private String formatTime(double seconds) {
        if (seconds < 60)
            return String.format("%.1f sec", seconds);
        if (seconds < 3600)
            return String.format("%.1f min", seconds / 60);
        return String.format("%.1f hours", seconds / 3600);
    }
}
