package com.example.automl_prototype_1;

import com.example.automl_prototype_1.model.Dataset;
import com.example.automl_prototype_1.service.agentic.AgenticAdvisor;
import com.example.automl_prototype_1.service.agentic.PipelineSuggestion;
import java.util.List;
import java.util.ArrayList;

public class AgenticTest {
    public static void main(String[] args) {
        System.out.println("=== Agentic AI Verification Test ===");
        AgenticAdvisor advisor = new AgenticAdvisor();

        // Test Case 1: Small Dataset
        System.out.println("\n--- Test Case 1: Small Dataset (100 rows, 10 features) ---");
        Dataset smallDataset = createDummyDataset(100, 10, 0.0);
        List<PipelineSuggestion> suggestions1 = advisor.suggestPipelines(smallDataset);
        suggestions1.forEach(System.out::println);

        // Test Case 2: Large Dataset
        System.out.println("\n--- Test Case 2: Large Dataset (2000 rows, 20 features) ---");
        Dataset largeDataset = createDummyDataset(2000, 20, 0.0);
        List<PipelineSuggestion> suggestions2 = advisor.suggestPipelines(largeDataset);
        suggestions2.forEach(System.out::println);

        // Test Case 3: Sparse Dataset
        System.out.println("\n--- Test Case 3: Sparse Dataset (500 rows, 50 features, 80% sparse) ---");
        Dataset sparseDataset = createDummyDataset(500, 50, 0.8);
        List<PipelineSuggestion> suggestions3 = advisor.suggestPipelines(sparseDataset);
        suggestions3.forEach(System.out::println);
    }

    private static Dataset createDummyDataset(int rows, int cols, double sparsity) {
        double[][] features = new double[rows][cols];
        int[] labels = new int[rows];
        List<String> headers = new ArrayList<>();
        for (int j = 0; j < cols; j++)
            headers.add("Feat" + j);
        headers.add("Class");

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (Math.random() > sparsity) {
                    features[i][j] = Math.random();
                } else {
                    features[i][j] = 0.0;
                }
            }
            labels[i] = (i % 2); // Dummy labels
        }
        return new Dataset(null, headers, features, labels, "Class");
    }
}
