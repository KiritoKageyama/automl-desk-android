package com.example.automl_prototype_1.experiment;

import com.example.automl_prototype_1.algorithm.*;
import com.example.automl_prototype_1.classifier.cpu.*;
import com.example.automl_prototype_1.dataprovider.CsvDataProvider;
import com.example.automl_prototype_1.model.Dataset;
import com.example.automl_prototype_1.model.ExecutionResult;
import com.example.automl_prototype_1.service.IClassifier;
// import com.example.automl_prototype_1.service.LogService;

import java.util.*;

public class ComprehensiveAblationRunner {

    public static void main(String[] args) {
        System.out.println("Starting Comprehensive Ablation Study...");

        // 1. Load Dataset
        String datasetPath = "wdbc.csv";
        Dataset dataset = null;
        CsvDataProvider dataProvider = new CsvDataProvider();
        try (java.io.FileReader reader = new java.io.FileReader(datasetPath)) {
            dataset = dataProvider.loadDataset(reader);
            dataset.setTargetColumnName("Diagnosis");
            // VITAL: Standardize data for SVM/KNN performance
            dataset.standardize();
            System.out.println("Dataset Loaded & Standardized: WDBC (" + dataset.getFeatureCount() + " predictors)");
        } catch (Exception e) {
            try (java.io.FileReader reader = new java.io.FileReader("automl_core/wdbc.csv")) {
                dataset = dataProvider.loadDataset(reader);
                dataset.setTargetColumnName("Diagnosis");
                dataset.standardize();
                System.out.println("Dataset Loaded & Standardized: WDBC (automl_core) (" + dataset.getFeatureCount()
                        + " predictors)");
            } catch (Exception ex) {
                System.err.println("Fatal: Could not load wdbc.csv. " + ex.getMessage());
                return;
            }
        }

        // 2. Define Combinations
        // Algorithms
        List<AlgoFactory> algos = new ArrayList<>();
        algos.add(new AlgoFactory("GA", (p, i) -> new GeneticAlgorithm(p, i)));
        algos.add(new AlgoFactory("BPSO", (p, i) -> new BinaryParticleSwarmOptimization(p, i))); // Assuming constructor
        algos.add(new AlgoFactory("IGPSO", (p, i) -> new IGPSO(p, i))); // Assuming constructor
        algos.add(new AlgoFactory("WWO", (p, i) -> new WaterWaveOptimization(p, i)));
        algos.add(new AlgoFactory("PLS", (p, i) -> new ParallelLocalSearchAlgorithm(p, i)));
        algos.add(new AlgoFactory("ASO", (p, i) -> new AtomSearchOptimization(p, i)));

        // Classifiers
        List<IClassifier> classifiers = new ArrayList<>();
        // Improved RF: 10 -> 50 trees for better stability/accuracy
        classifiers.add(new RandomForestClassifier(50));
        classifiers.add(new KnnClassifier(5));
        classifiers.add(new SvmClassifier());
        try {
            classifiers.add(new XGBoostClassifier(10));
        } catch (Throwable t) {
            System.out.println("Skipping XGBoost (libs missing): " + t.getMessage());
        }

        // 3. Run Benchmarks
        int pop = 10;
        int iter = 5;

        System.out.printf("%-10s %-10s %-15s %-10s %-10s %-10s %-10s%n",
                "Algo", "Clf", "Case", "Acc", "F1", "Feats", "Time(ms)");
        System.out.println("--------------------------------------------------------------------------------");

        for (AlgoFactory factory : algos) {
            for (IClassifier clf : classifiers) {
                runCase(factory, clf, dataset, "Full", true, true, pop, iter);
                runCase(factory, clf, dataset, "NoProxy", false, true, pop, iter);
                runCase(factory, clf, dataset, "NoGuide", true, false, pop, iter);
            }
        }
    }

    private static void runCase(AlgoFactory factory, IClassifier clf, Dataset ds, String caseName,
            boolean useProxy, boolean useHeuristics, int pop, int iter) {
        try {
            AbstractFeatureSelectionAlgorithm algo = factory.create(pop, iter);
            algo.setUseProxy(useProxy);
            algo.setUseHeuristics(useHeuristics);

            Map<String, Object> config = new HashMap<>();
            config.put("classifier", clf);

            ExecutionResult res = algo.execute(ds, config);

            System.out.printf("%-10s %-10s %-15s %-10.4f %-10.4f %-2d/%-2d     %-10d%n",
                    factory.name, clf.getName(), caseName,
                    res.getAccuracy(), res.getF1Score(),
                    res.getSelectedFeatures(), res.getOriginalFeatures(),
                    res.getExecutionTimeMs());
        } catch (Exception e) {
            System.out.printf("%-10s %-10s %-15s ERROR: %s%n", factory.name, clf.getName(), caseName, e.getMessage());
        }
    }

    interface AlgoCreator {
        AbstractFeatureSelectionAlgorithm create(int p, int i);
    }

    static class AlgoFactory {
        String name;
        AlgoCreator creator;

        AlgoFactory(String n, AlgoCreator c) {
            name = n;
            creator = c;
        }

        AbstractFeatureSelectionAlgorithm create(int p, int i) {
            return creator.create(p, i);
        }
    }
}
