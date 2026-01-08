package com.example.automl_prototype_1;

import com.example.automl_prototype_1.algorithm.*;
import com.example.automl_prototype_1.classifier.*;
import com.example.automl_prototype_1.dataprovider.CsvDataProvider;
import com.example.automl_prototype_1.model.Dataset;
import com.example.automl_prototype_1.model.ExecutionResult;
import com.example.automl_prototype_1.service.AppStateService;
import com.example.automl_prototype_1.service.LogService;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

import java.io.File;
import java.util.*;

public class WdbcIntegrationTest {

    @Test
    public void testWdbcWithGpu() {
        System.out.println("==========================================================");
        System.out.println("   EXTREME UI TEST: WDBC DATASET + RTX 5070 Ti            ");
        System.out.println("==========================================================");

        // 1. Setup Logging
        LogService.getInstance().setListener(message -> System.out.println("[LOG] " + message));
        AppStateService.getInstance().setLoggingEnabled(true);

        // 2. Load Dataset
        File wdbcFile = new File("wdbc.csv");
        assertTrue(wdbcFile.exists(), "wdbc.csv not found in project root!");

        // 3. Configure AppState
        AppStateService.getInstance().setUseWsl(true);
        AppStateService.getInstance().setWslDistributionName("Ubuntu");
        AppStateService.getInstance().setUseGpu(true);
        AppStateService.getInstance().setSelectedGpuName("NVIDIA GeForce RTX 5070 Ti Laptop GPU");
        AppStateService.getInstance().setTargetColumnName("Diagnosis"); // Explicitly set target for Algo

        // 4. Test Configuration
        // Using GpuXGBoost as it's the most intensive
        GpuXGBoostClassifier classifier = new GpuXGBoostClassifier(10, 3);
        GeneticAlgorithm algo = new GeneticAlgorithm(10, 5); // Pop 10, 5 iterations

        System.out.printf("TESTING: [WDBC] + [%s] + [%s]%n", classifier.getName(), algo.getClass().getSimpleName());

        try {
            System.out.println("Loading WDBC Dataset...");
            CsvDataProvider provider = new CsvDataProvider();
            Dataset rawDataset = provider.loadDataset(new java.io.FileReader(wdbcFile));

            // Create dataset with target column "Diagnosis"
            Dataset dataset = new Dataset(
                    wdbcFile,
                    rawDataset.getHeaders(),
                    rawDataset.getFeatures(),
                    rawDataset.getLabels(),
                    "Diagnosis");

            System.out.println("Dataset Loaded: " + dataset.getFeatures().length + " rows, "
                    + dataset.getHeaders().size() + " columns.");

            Map<String, Object> config = new HashMap<>();
            config.put("classifier", classifier);

            // Execute
            ExecutionResult result = algo.execute(dataset, config);

            assertNotNull(result, "Result should not be null");
            System.out.println("------------------------------------------------");
            System.out.println("RESULT: Accuracy = " + String.format("%.4f", result.getAccuracy()));
            System.out.println("Selected Features: " + result.getSelectedFeatures());
            System.out.println("------------------------------------------------");

            // WDBC is easy, expects > 90%
            assertTrue(result.getAccuracy() > 0.90, "Accuracy should be > 90% on WDBC");

        } catch (Exception e) {
            e.printStackTrace();
            fail("Test failed with exception: " + e.getMessage());
        }
    }
}
