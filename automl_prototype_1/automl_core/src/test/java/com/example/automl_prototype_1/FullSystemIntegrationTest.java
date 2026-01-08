package com.example.automl_prototype_1;

import com.example.automl_prototype_1.algorithm.*;
import com.example.automl_prototype_1.classifier.*;
import com.example.automl_prototype_1.model.Dataset;
import com.example.automl_prototype_1.model.ExecutionResult;
import com.example.automl_prototype_1.service.AppStateService;
import com.example.automl_prototype_1.service.IClassifier;
import com.example.automl_prototype_1.service.LogService;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

import java.util.*;

public class FullSystemIntegrationTest {

    @Test
    public void verifyAllCombinations() {
        System.out.println("==========================================================");
        System.out.println("   STARTING JUNIT FULL INTEGRATION TEST                   ");
        System.out.println("==========================================================");

        // 1. Setup Logging
        LogService.getInstance().setListener(message -> System.out.println("[LOG] " + message));
        AppStateService.getInstance().setLoggingEnabled(true);

        // 2. Generate Data
        Dataset dataset = generateSyntheticDataset(50, 5); // Smaller for speed

        // Define combinations
        boolean[] wslOptions = { true };
        boolean[] gpuOptions = { true }; // Focus on true-true first as it's the critical path

        List<IClassifier> classifiers = Arrays.asList(
                new GpuXGBoostClassifier(5, 2),
                new GpuRandomForestClassifier(5),
                new GpuPythonSvmClassifier(),
                new GpuKnnClassifier(3),
                new GpuLogisticRegression());

        List<String> algosCallback = Arrays.asList("GeneticAlgorithm", "PSO", "SimulatedAnnealing",
                "DifferentialEvolution");

        int failures = 0;

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
                            } else {
                                System.out.println("FAIL (Null Result)");
                                failures++;
                            }

                        } catch (Exception | LinkageError e) {
                            System.out.println(
                                    "FAIL (Error: " + e.getClass().getSimpleName() + " - " + e.getMessage() + ")");
                            e.printStackTrace();
                            failures++;
                        }
                    }
                }
            }
        }

        assertEquals(0, failures, "Some integration tests failed. Check logs.");
    }

    private AbstractFeatureSelectionAlgorithm createAlgorithm(String name) {
        int pop = 5;
        int iter = 2;
        switch (name) {
            case "GeneticAlgorithm":
                return new GeneticAlgorithm(pop, iter);
            case "PSO":
                return new BinaryParticleSwarmOptimization(pop, iter);
            case "SimulatedAnnealing":
                return new SimulatedAnnealing(pop, iter);
            case "DifferentialEvolution":
                return new DifferentialEvolution(pop, iter);
            default:
                return new GeneticAlgorithm(pop, iter);
        }
    }

    private Dataset generateSyntheticDataset(int rows, int cols) {
        double[][] x = new double[rows][cols];
        List<String> headers = new ArrayList<>();
        Random rand = new Random(42);

        for (int j = 0; j < cols - 1; j++) {
            headers.add("Feature_" + j);
        }
        headers.add("Class");

        for (int i = 0; i < rows; i++) {
            double sum = 0;
            for (int j = 0; j < cols - 1; j++) {
                x[i][j] = rand.nextDouble();
                sum += x[i][j];
            }
            x[i][cols - 1] = (sum > ((cols - 1) / 2.0)) ? 1.0 : 0.0;
        }
        return new Dataset(null, headers, x, null, "Class");
    }
}
