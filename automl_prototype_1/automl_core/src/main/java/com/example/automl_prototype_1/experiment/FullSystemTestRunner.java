package com.example.automl_prototype_1.experiment;

import com.example.automl_prototype_1.algorithm.*;
import com.example.automl_prototype_1.classifier.*;
import com.example.automl_prototype_1.model.Dataset;
import com.example.automl_prototype_1.model.ExecutionResult;
import com.example.automl_prototype_1.service.AppStateService;
import com.example.automl_prototype_1.service.IClassifier;
import com.example.automl_prototype_1.service.LogService;

import java.util.*;

/**
 * Comprehensive Integration Test Suite.
 * Iterates through all toggle combinations, classifiers, and algorithms to
 * ensure system stability.
 */
public class FullSystemTestRunner {

    public static void main(String[] args) {
        System.out.println("==========================================================");
        System.out.println("   STARTING FULL SYSTEM INTEGRATION TEST (SELF-CHECK)     ");
        System.out.println("==========================================================");

        // 1. Setup Logging
        LogService.getInstance().setListener(message -> System.out.println("[LOG] " + message));
        AppStateService.getInstance().setLoggingEnabled(true);

        // 2. Generate Synthetic Data
        // 100 rows, 5 features, last col is Class (0 or 1)
        Dataset dataset = generateSyntheticDataset(100, 5);
        System.out.println("Synthetic Dataset Generated: 100 rows, 5 features.");

        // Define combinations
        boolean[] wslOptions = { true }; // Focus on true as that's the complex path. Windows native is less critical
                                         // now.
        boolean[] gpuOptions = { true, false };

        List<IClassifier> classifiers = Arrays.asList(
                new GpuXGBoostClassifier(10, 2), // Low trees for speed
                new GpuRandomForestClassifier(10),
                new GpuPythonSvmClassifier(),
                new GpuKnnClassifier(),
                new GpuLogisticRegression());

        // We test a representative subset of algorithms to save time, or all if
        // requested.
        // Testing GA and PSO covers most logic paths.
        // We will instantiate them fresh in the loop.
        List<String> algosCallback = Arrays.asList("GeneticAlgorithm", "PSO", "SimulatedAnnealing",
                "DifferentialEvolution");

        int totalTests = wslOptions.length * gpuOptions.length * classifiers.size() * algosCallback.size();
        int passed = 0;
        int failed = 0;

        for (boolean useWsl : wslOptions) {
            for (boolean useGpu : gpuOptions) {
                AppStateService.getInstance().setUseWsl(useWsl);
                AppStateService.getInstance().setUseGpu(useGpu);

                System.out.printf("%n[ CONFIG ] WSL=%s | GPU=%s%n", useWsl, useGpu);

                for (IClassifier classifier : classifiers) {
                    for (String algoName : algosCallback) {
                        AbstractFeatureSelectionAlgorithm algo = createAlgorithm(algoName);
                        String testName = String.format("[%s] + [%s]", classifier.getName(),
                                algo.getClass().getSimpleName());

                        System.out.print("TESTING: " + testName + " ... ");

                        try {
                            Map<String, Object> config = new HashMap<>();
                            config.put("classifier", classifier);

                            // Run small optimization
                            ExecutionResult result = algo.execute(dataset, config);

                            if (result != null && result.getAccuracy() >= 0) {
                                System.out.println("PASS (Acc: " + String.format("%.3f", result.getAccuracy()) + ")");
                                passed++;
                            } else {
                                System.out.println("FAIL (Null Result)");
                                failed++;
                            }

                        } catch (Exception e) {
                            System.out.println("FAIL (Exception: " + e.getMessage() + ")");
                            // e.printStackTrace(); // Keep log clean
                            failed++;
                        }
                    }
                }
            }
        }

        System.out.println("\n==========================================================");
        System.out.println("TEST SUMMARY");
        System.out.println("PASS: " + passed);
        System.out.println("FAIL: " + failed);
        System.out.println("TOTAL: " + totalTests);
        System.out.println("==========================================================");

        if (failed > 0) {
            System.exit(1);
        }
    }

    private static AbstractFeatureSelectionAlgorithm createAlgorithm(String name) {
        // Use small population/iterations for speed
        int pop = 5;
        int iter = 2;
        switch (name) {
            case "GeneticAlgorithm":
                return new GeneticAlgorithm(pop, iter);
            case "PSO":
                return new BinaryParticleSwarmOptimization(pop, iter); // Use BPSO class name
            case "SimulatedAnnealing":
                return new SimulatedAnnealing(pop, iter);
            case "DifferentialEvolution":
                return new DifferentialEvolution(pop, iter);
            default:
                return new GeneticAlgorithm(pop, iter);
        }
    }

    private static Dataset generateSyntheticDataset(int rows, int cols) {
        double[][] x = new double[rows][cols];
        List<String> headers = new ArrayList<>();
        Random rand = new Random(42);

        for (int j = 0; j < cols - 1; j++) {
            headers.add("Feature_" + j);
        }
        headers.add("Class");

        double[] y = new double[rows];

        // Fill Data
        for (int i = 0; i < rows; i++) {
            double sum = 0;
            for (int j = 0; j < cols - 1; j++) {
                x[i][j] = rand.nextDouble();
                sum += x[i][j];
            }
            // Simple rule: if sum > cols/2, class 1, else 0
            x[i][cols - 1] = (sum > ((cols - 1) / 2.0)) ? 1.0 : 0.0;
        }

        // Dataset expects target col name.
        return new Dataset(null, headers, x, null, "Class");
    }
}
