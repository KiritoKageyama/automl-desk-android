package com.example.automl_prototype_1.experiment;

import com.example.automl_prototype_1.algorithm.AblationAwareGeneticAlgorithm;
import com.example.automl_prototype_1.classifier.cpu.RandomForestClassifier;
import com.example.automl_prototype_1.dataprovider.CsvDataProvider;
import com.example.automl_prototype_1.model.Dataset;
import com.example.automl_prototype_1.model.ExecutionResult;
import java.util.HashMap;
import java.util.Map;

public class AblationExperimentRunner {

    public static void main(String[] args) {
        System.out.println("Starting Ablation Study Experiments...");

        // 1. Generate/Load Data
        System.out.println("Loading WDBC dataset...");
        // Use the existing wdbc.csv in the project root or core folder
        // Note: The file path needs to be correct relative to execution dir
        String datasetPath = "wdbc.csv";

        // Load Dataset
        CsvDataProvider dataProvider = new CsvDataProvider();
        Dataset dataset = null;
        try (java.io.FileReader reader = new java.io.FileReader(datasetPath)) {
            dataset = dataProvider.loadDataset(reader);
            // Explicitly set target column for WDBC
            dataset.setTargetColumnName("Diagnosis");
        } catch (Exception e) {
            // Fallback if not found in root, try core folder
            try (java.io.FileReader reader = new java.io.FileReader("automl_core/wdbc.csv")) {
                dataset = dataProvider.loadDataset(reader);
                dataset.setTargetColumnName("Diagnosis");
                // Remove ID column if present as it's not a feature
                // dataset.dropColumn("ID"); // Assuming dropColumn exists or just ignore it
                // later?
                // Dataset.getFeatureMatrix actually generally ignores non-numeric?
                // ID is likely unique numeric, which might be treated as a feature.
                // For now, let's just set Target.
            } catch (Exception ex) {
                com.example.automl_prototype_1.service.LogService.error("AblationRunner",
                        "Failed to load dataset: " + ex.getMessage());
                return;
            }
        }

        // 2. Define Configurations
        // Baseline: Random Forest
        // Configs:
        // A: Full System (GA + Proxy + Bias + Advisor)
        // B: No Fast-Proxy
        // C: No Low-Cardinality Bias (No Guidance)

        System.out.println("Dataset Loaded: WDBC (" + dataset.getFeatureCount() + " features, "
                + dataset.getRecordCount() + " instances).");

        // Run Experiments
        runExperiment("Full System", dataset, true, true);
        runExperiment("No Fast-Proxy", dataset, false, true);
        runExperiment("No Low-Cardinality Bias", dataset, true, false);
    }

    private static void runExperiment(String name, Dataset dataset, boolean useProxy, boolean useBias) {
        System.out.println("\n--- Running Experiment: " + name + " ---");
        long start = System.currentTimeMillis();

        // INCREASED BUDGET: 30 Population, 20 Iterations (Better for WDBC)
        AblationAwareGeneticAlgorithm optimizer = new AblationAwareGeneticAlgorithm(30, 20, useProxy, useBias);

        Map<String, Object> config = new HashMap<>();
        config.put("classifier", new RandomForestClassifier(10)); // Base classifier

        ExecutionResult result = optimizer.execute(dataset, config);

        long duration = System.currentTimeMillis() - start;
        System.out.println("Result [" + name + "]:");
        System.out.println("  Accuracy: " + String.format("%.4f", result.getAccuracy()));
        System.out.println("  F1 Score: " + String.format("%.4f", result.getF1Score()));
        System.out.println("  Features Selected: " + result.getSelectedFeatures() + "/" + result.getOriginalFeatures());
        System.out.println("  Total Time: " + duration + " ms");
    }
}
