package com.example.automl_prototype_1.classifier;

import com.example.automl_prototype_1.service.IClassifier;
import com.example.automl_prototype_1.service.PythonExecutionService;
import com.example.automl_prototype_1.service.LogService;
import smile.classification.RandomForest;
import smile.data.DataFrame;
import smile.data.formula.Formula;
import java.io.File;

import java.util.Properties;
import java.util.UUID;
import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;

import java.util.Scanner;

/**
 * A GPU-enabled wrapper for Random Forest Classifier.
 * Uses Python Bridge (cuML/scikit-learn) if available, falls back to CPU Smile
 * implementation.
 */
public class GpuRandomForestClassifier implements IClassifier {
    private final int nTrees;
    private RandomForest javaModel;
    private File pythonModelPath;
    private boolean usePython = false;

    public GpuRandomForestClassifier(int nTrees) {
        this.nTrees = nTrees;
    }

    public GpuRandomForestClassifier() {
        this(100);
    }

    @Override
    public String getName() {
        return usePython ? "Random Forest (Python/GPU Bridge)" : "Random Forest (Java CPU Fallback)";
    }

    @Override
    public void train(double[][] features, int[] labels) {
        if (features.length == 0 || features[0].length == 0) {
            return;
        }

        // 1. Try Python Bridge
        if (PythonExecutionService.getInstance().isAvailable()) {
            try {
                LogService.info("GpuRF", "Attempting training via Python Bridge...");
                File trainFile = createTempCsv(features, labels);
                File modelFile = File.createTempFile("rf_model_" + UUID.randomUUID(), ".pkl");

                // We assume the script is available in a known location or extracted
                // For prototype, checking 'python/train_rf.py' relative to run dir or classpath
                // Ideally PythonExecutionService handles script extraction.
                // We will hardcode a helper check here or assume specific path for now.
                // NOTE: In production, extract from JAR. Here, assuming dev environment path
                // structure.
                // Extract script from resources
                File scriptFile = PythonExecutionService.getInstance().extractScriptToTemp("train_rf.py");

                boolean success = PythonExecutionService.getInstance().trainModel(
                        scriptFile,
                        trainFile,
                        modelFile,
                        "--n_estimators", String.valueOf(nTrees));

                if (success) {
                    this.pythonModelPath = modelFile;
                    this.usePython = true;
                    LogService.info("GpuRF", "Python training successful. Model saved to " + modelFile.getName());
                    trainFile.delete(); // Cleanup data
                    return; // Early exit, success
                }
            } catch (Exception e) {
                LogService.error("GpuRF", "Python training failed: " + e.getMessage());
                // Fallthrough to Java
            }
        }

        // 2. Fallback to Java (Smile)
        LogService.info("GpuRF", "Falling back to pure Java implementation (CPU).");
        this.usePython = false;

        int[][] labelMatrix = new int[labels.length][1];
        for (int i = 0; i < labels.length; i++) {
            labelMatrix[i][0] = labels[i];
        }

        DataFrame dataFrame = DataFrame.of(features).merge(DataFrame.of(labelMatrix, "Class"));
        Formula formula = Formula.of("Class", ".");
        Properties props = new Properties();
        props.setProperty("smile.random.forest.trees", String.valueOf(this.nTrees));
        this.javaModel = RandomForest.fit(formula, dataFrame, props);
    }

    @Override
    public int[] predict(double[][] features) {
        if (this.usePython && this.pythonModelPath != null) {
            try {
                File inputFile = createTempCsv(features, null); // No labels
                File outputFile = File.createTempFile("preds_" + UUID.randomUUID(), ".csv");

                File scriptFile = PythonExecutionService.getInstance().extractScriptToTemp("predict.py");

                boolean success = PythonExecutionService.getInstance().predict(
                        scriptFile,
                        this.pythonModelPath,
                        inputFile,
                        outputFile);

                if (success) {
                    return readPredictions(outputFile);
                } else {
                    LogService.error("GpuRF", "Python prediction failed. Returning zeros.");
                    return new int[features.length];
                }
            } catch (Exception e) {
                LogService.error("GpuRF", "Error during Python prediction: " + e.getMessage());
                return new int[features.length];
            }
        }

        // Java Fallback
        if (this.javaModel == null)
            return new int[features.length];

        DataFrame predictionFrame = DataFrame.of(features);
        return this.javaModel.predict(predictionFrame);
    }

    // Helpers
    private File createTempCsv(double[][] features, int[] labels) throws IOException {
        File temp = File.createTempFile("data_" + UUID.randomUUID(), ".csv");
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(temp))) {
            // Header
            for (int i = 0; i < features[0].length; i++) {
                writer.write("F" + i);
                writer.write(",");
            }
            writer.write("Class");
            writer.newLine();

            for (int i = 0; i < features.length; i++) {
                for (int j = 0; j < features[i].length; j++) {
                    writer.write(String.valueOf(features[i][j]));
                    writer.write(",");
                }
                if (labels != null) {
                    writer.write(String.valueOf(labels[i]));
                } else {
                    writer.write("0"); // Dummy target for X only
                }
                writer.newLine();
            }
        }
        return temp;
    }

    private int[] readPredictions(File csvFile) throws IOException {
        // Read single column CSV
        try (Scanner scanner = new Scanner(csvFile)) {
            String content = scanner.useDelimiter("\\A").next();
            // Split by new line or comma
            String[] tokens = content.trim().split("[,\\s]+");
            int[] preds = new int[tokens.length];
            for (int i = 0; i < tokens.length; i++) {
                try {
                    preds[i] = (int) Double.parseDouble(tokens[i]); // Handle 1.0 vs 1
                } catch (NumberFormatException e) {
                    preds[i] = 0;
                }
            }
            return preds;
        }
    }
}
