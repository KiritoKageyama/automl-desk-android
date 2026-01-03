package com.example.automl_prototype_1;

import com.example.automl_prototype_1.classifier.*;
import com.example.automl_prototype_1.service.IClassifier;
import java.util.Random;
import org.junit.jupiter.api.Test;

public class PerformanceBenchmark {

    @Test
    public void testPerformance() {
        System.out.println("Starting Benchmark...");

        // 1. Generate Dummy Data
        int rows = 2000;
        int cols = 50;
        double[][] features = new double[rows][cols];
        int[] labels = new int[rows];
        Random rand = new Random(42);

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                features[i][j] = rand.nextDouble();
            }
            // Smile LASVM expects -1 and 1
            labels[i] = rand.nextBoolean() ? 1 : -1;
        }

        System.out.println("Dataset: " + rows + " rows, " + cols + " columns.");
        System.out.println("--------------------------------------------------");

        // 2. Test Classifiers
        try {
            testClassifierPair(new SvmClassifier(), new GpuSvmClassifier(), features, labels);
        } catch (Exception e) {
            System.err.println("SVM Test Failed: " + e.getMessage());
        }

        try {
            testClassifierPair(new RandomForestClassifier(50), new GpuRandomForestClassifier(), features, labels);
        } catch (Exception e) {
            System.err.println("RF Test Failed: " + e.getMessage());
        }

        // 3. Test KNN
        // KNN uses 0/1 usually or arbitrary, but let's see if -1/1 works or if we need
        // to remap.
        // GpuKnnClassifier expects integer mapping.
        try {
            testClassifierPair(new KnnClassifier(5), new GpuKnnClassifier(5), features, labels);
        } catch (Exception e) {
            System.err.println("KNN Test Failed: " + e.getMessage());
        }
    }

    private void testClassifierPair(IClassifier cpu, IClassifier gpu, double[][] X, int[] y) {
        System.out.println("Testing " + cpu.getName() + " vs " + gpu.getName());

        long cpuTime = timeTraining(cpu, X, y);
        System.out.println("  CPU Train Time: " + cpuTime + " ms");

        long gpuTime = timeTraining(gpu, X, y);
        System.out.println("  GPU Train Time: " + gpuTime + " ms");

        if (cpuTime > 0) {
            double speedup = (double) cpuTime / Math.max(gpuTime, 1);
            System.out.printf("  Speedup: %.2fx\n", speedup);
        }
        System.out.println("--------------------------------------------------");
    }

    private long timeTraining(IClassifier clf, double[][] X, int[] y) {
        long start = System.currentTimeMillis();
        clf.train(X, y);
        long end = System.currentTimeMillis();
        return end - start;
    }
}
