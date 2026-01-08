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
 * A Python-bridged implementation of SVM.
 * This version supports true GPU acceleration for RBF/Poly kernels via cuML.
 */
public class GpuPythonSvmClassifier implements IClassifier {

    private final double C;
    private final String kernel;
    private File pythonModelPath;

    public GpuPythonSvmClassifier() {
        this(1.0, "rbf");
    }

    public GpuPythonSvmClassifier(double C, String kernel) {
        this.C = C;
        this.kernel = kernel;
    }

    @Override
    public String getName() {
        return "SVM (Python/GPU Bridge - " + kernel + ")";
    }

    @Override
    public void train(double[][] features, int[] labels) {
        if (features.length == 0 || features[0].length == 0)
            return;

        if (PythonExecutionService.getInstance().isAvailable()) {
            try {
                LogService.info("GpuPySVM", "Preparing SVM Python training (kernel=" + kernel + ")...");
                File trainFile = createTempCsv(features, labels);
                File modelFile = File.createTempFile("svm_model_" + UUID.randomUUID(), ".pkl");

                File scriptFile = PythonExecutionService.getInstance().extractScriptToTemp("train_svm.py");

                boolean success = PythonExecutionService.getInstance().trainModel(
                        scriptFile,
                        trainFile,
                        modelFile,
                        "--C", String.valueOf(C),
                        "--kernel", kernel);

                if (success) {
                    this.pythonModelPath = modelFile;
                    LogService.info("GpuPySVM", "Training successful. Model saved.");
                }
                trainFile.delete();
            } catch (Exception e) {
                LogService.error("GpuPySVM", "Training failed: " + e.getMessage());
            }
        } else {
            LogService.error("GpuPySVM", "Python not available. Cannot train.");
        }
    }

    @Override
    public int[] predict(double[][] features) {
        if (this.pythonModelPath != null && PythonExecutionService.getInstance().isAvailable()) {
            try {
                File inputFile = createTempCsv(features, null);
                File outputFile = File.createTempFile("preds_svm_" + UUID.randomUUID(), ".csv");

                File scriptFile = PythonExecutionService.getInstance().extractScriptToTemp("predict.py");

                boolean success = PythonExecutionService.getInstance().predict(
                        scriptFile,
                        this.pythonModelPath,
                        inputFile,
                        outputFile);

                if (success) {
                    return readPredictions(outputFile);
                }
            } catch (Exception e) {
                LogService.error("GpuPySVM", "Prediction failed: " + e.getMessage());
            }
        }
        return new int[features.length];
    }

    // Helper
    private File createTempCsv(double[][] features, int[] labels) throws IOException {
        File temp = File.createTempFile("data_" + UUID.randomUUID(), ".csv");
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(temp))) {
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
