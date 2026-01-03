package com.example.automl_prototype_1.experiment;

import com.example.automl_prototype_1.algorithm.AblationAwareGeneticAlgorithm;
import com.example.automl_prototype_1.classifier.RandomForestClassifier;
import com.example.automl_prototype_1.dataprovider.CsvDataProvider;
import com.example.automl_prototype_1.model.Dataset;
import com.example.automl_prototype_1.model.ExecutionResult;
import java.util.HashMap;
import java.util.Map;

public class AblationExperimentRunner {

    public static void main(String[] args) {
        System.out.println("Starting Ablation Study Experiments...");

        // 1. Generate Data
        System.out.println("Generating synthetic dataset...");
        // 500 instances, 20 features, 5 informative
        SyntheticDataGenerator.generate(500, 20, 5, true);

        // Load Dataset
        CsvDataProvider dataProvider = new CsvDataProvider();
        Dataset dataset = null;
        try (java.io.FileReader reader = new java.io.FileReader("synthetic_ablation_data.csv")) {
            dataset = dataProvider.loadDataset(reader);

            // For now, let's assume the dataset loaded is good enough since we generated it
            // clean.
        } catch (Exception e) {
            e.printStackTrace();
            System.err.println("Failed to load dataset. Exiting.");
            return;
        }

        // 2. Define Configurations
        // Baseline: Random Forest
        // Configs:
        // A: Full System (GA + Proxy + Bias + Advisor)
        // B: No Advisor (Random Algo - we will simulate this by checking what Full
        // chose vs others,
        // but here we act as if we picked something else, or just run GA without
        // Advisor insight?
        // Actually, 'No Advisor' usually means picking a default or random.
        // Let's compare Optimized GA (Advisor Choice) vs Random Search or Default
        // params.
        // For this study, let's assume 'No Advisor' -> Standard GA without meta-feature
        // tuning?
        // Or just compare against 'Random Search' as the baseline for 'No Advisor'.
        // Let's run: Full (Optimized GA), No Proxy (Slow GA), No Bias (Unbiased GA).

        System.out.println("Dataset Loaded: " + dataset.getFeatureCount() + " features, "
                + dataset.getRecordCount() + " instances.");

        runExperiment("Full System", dataset, true, true);
        runExperiment("No Fast-Proxy", dataset, false, true);
        runExperiment("No Low-Cardinality Bias", dataset, true, false);

        // For "No Advisor", we simulate a bad choice or default.
        // Let's say Advisor picked GA. "No Advisor" might pick something random or less
        // optimal.
        // Or we can just say "No Advisor" means we don't use the specific params
        // suggested.
        // For simplicity, we'll focus on the GA variations here.
    }

    private static void runExperiment(String name, Dataset dataset, boolean useProxy, boolean useBias) {
        System.out.println("\n--- Running Experiment: " + name + " ---");
        long start = System.currentTimeMillis();

        // 10 Population, 5 Iterations (Small budget for test)
        // Real experiment would be larger.
        AblationAwareGeneticAlgorithm optimizer = new AblationAwareGeneticAlgorithm(10, 5, useProxy, useBias);

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
