package com.example.automl_prototype_1.classifier;

import com.example.automl_prototype_1.service.IClassifier;
import com.example.automl_prototype_1.service.PythonExecutionService;
import com.example.automl_prototype_1.service.LogService;
import java.io.File;
import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.UUID;
import java.util.Scanner;

/**
 * A Python-bridged implementation of XGBoost.
 * This version supports true GPU acceleration via 'gpu_hist' in the Python
 * XGBoost library.
 */
public class GpuPythonXGBoostClassifier implements IClassifier {

    private final int nEstimators;
    private final int maxDepth;
    private File pythonModelPath;

    public GpuPythonXGBoostClassifier() {
        this.nEstimators = 100;
        this.maxDepth = 6;
    }

    public GpuPythonXGBoostClassifier(int nEstimators, int maxDepth) {
        this.nEstimators = nEstimators;
        this.maxDepth = maxDepth;
    }

    @Override
    public String getName() {
        return "XGBoost (Python/GPU Bridge)";
    }

    @Override
    public void train(double[][] features, int[] labels) {
        if (features.length == 0 || features[0].length == 0)
            return;

        if (PythonExecutionService.getInstance().isAvailable()) {
            try {
                LogService.info("GpuPyXGB", "Preparing Python Bridge training...");
                File trainFile = createTempCsv(features, labels);
                File modelFile = File.createTempFile("xgb_model_" + UUID.randomUUID(), ".pkl");

                File scriptFile = new File("automl_core/src/main/resources/python/train_xgb.py");
                if (!scriptFile.exists())
                    scriptFile = new File("src/main/resources/python/train_xgb.py");

                if (scriptFile.exists()) {
                    boolean success = PythonExecutionService.getInstance().trainModel(
                            scriptFile,
                            trainFile,
                            modelFile,
                            "--n_estimators", String.valueOf(nEstimators),
                            "--max_depth", String.valueOf(maxDepth));

                    if (success) {
                        this.pythonModelPath = modelFile;
                        LogService.info("GpuPyXGB", "Training successful. Model saved.");
                    }
                } else {
                    LogService.error("GpuPyXGB", "Script not found: train_xgb.py");
                }
                trainFile.delete();
            } catch (Exception e) {
                LogService.error("GpuPyXGB", "Training failed: " + e.getMessage());
            }
        } else {
            LogService.error("GpuPyXGB", "Python not available. Cannot train.");
        }
    }

    @Override
    public int[] predict(double[][] features) {
        if (this.pythonModelPath != null && PythonExecutionService.getInstance().isAvailable()) {
            try {
                File inputFile = createTempCsv(features, null);
                File outputFile = File.createTempFile("preds_xgb_" + UUID.randomUUID(), ".csv");

                File scriptFile = new File("automl_core/src/main/resources/python/predict.py");
                if (!scriptFile.exists())
                    scriptFile = new File("src/main/resources/python/predict.py");

                boolean success = PythonExecutionService.getInstance().predict(
                        scriptFile,
                        this.pythonModelPath,
                        inputFile,
                        outputFile);

                if (success) {
                    return readPredictions(outputFile);
                }
            } catch (Exception e) {
                LogService.error("GpuPyXGB", "Prediction failed: " + e.getMessage());
            }
        }
        return new int[features.length];
    }

    // Helper (Duplicate logic, ideally moved to util)
    private File createTempCsv(double[][] features, int[] labels) throws IOException {
        File temp = File.createTempFile("data_" + UUID.randomUUID(), ".csv");
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(temp))) {
            // Header
            for (int i = 0; i < features[0].length; i++)
                writer.write("F" + i + ",");
            writer.write("Class");
            writer.newLine();
            for (int i = 0; i < features.length; i++) {
                for (int j = 0; j < features[i].length; j++)
                    writer.write(features[i][j] + ",");
                if (labels != null)
                    writer.write(String.valueOf(labels[i]));
                else
                    writer.write("0");
                writer.newLine();
            }
        }
        return temp;
    }

    private int[] readPredictions(File csvFile) throws IOException {
        try (Scanner scanner = new Scanner(csvFile)) {
            if (!scanner.hasNext())
                return new int[0];
            String content = scanner.useDelimiter("\\A").next();
            String[] tokens = content.trim().split("[,\\s]+");
            int[] preds = new int[tokens.length];
            for (int i = 0; i < tokens.length; i++) {
                try {
                    preds[i] = (int) Double.parseDouble(tokens[i]);
                } catch (Exception e) {
                    preds[i] = 0;
                }
            }
            return preds;
        }
    }
}
