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

public class PythonAdaBoostClassifier implements IClassifier {

    protected final int nEstimators;
    protected final double learningRate;
    protected final int maxDepth;
    private File pythonModelPath;

    public PythonAdaBoostClassifier() {
        this.nEstimators = 50;
        this.learningRate = 1.0;
        this.maxDepth = 1; // Default for AdaBoost Stump
    }

    public PythonAdaBoostClassifier(int nEstimators, double learningRate, int maxDepth) {
        this.nEstimators = nEstimators;
        this.learningRate = learningRate;
        this.maxDepth = maxDepth;
    }

    @Override
    public String getName() {
        return "AdaBoost (Python)";
    }

    @Override
    public void train(double[][] features, int[] labels) {
        trainGeneric("adaboost", features, labels);
    }

    @Override
    public int[] predict(double[][] features) {
        return predictGeneric(features);
    }

    protected void trainGeneric(String modelType, double[][] features, int[] labels) {
        if (features.length == 0 || features[0].length == 0)
            return;

        if (PythonExecutionService.getInstance().isAvailable()) {
            try {
                File trainFile = createTempCsv(features, labels);
                File modelFile = File.createTempFile("model_" + modelType + "_" + UUID.randomUUID(), ".pkl");

                File scriptFile = new File("automl_core/src/main/resources/python/train_sklearn.py");
                if (!scriptFile.exists())
                    scriptFile = new File("src/main/resources/python/train_sklearn.py");

                if (scriptFile.exists()) {
                    boolean success = PythonExecutionService.getInstance().trainModel(
                            scriptFile,
                            trainFile,
                            modelFile,
                            "--model_type", modelType,
                            "--n_estimators", String.valueOf(nEstimators),
                            "--learning_rate", String.valueOf(learningRate),
                            "--max_depth", String.valueOf(maxDepth));
                    if (success) {
                        this.pythonModelPath = modelFile;
                    }
                } else {
                    LogService.error("PyBridge", "Script not found: train_sklearn.py");
                }
                trainFile.delete();
            } catch (Exception e) {
                LogService.error("PyBridge", "Training failed: " + e.getMessage());
            }
        }
    }

    protected int[] predictGeneric(double[][] features) {
        if (this.pythonModelPath != null && PythonExecutionService.getInstance().isAvailable()) {
            try {
                File inputFile = createTempCsv(features, null);
                File outputFile = File.createTempFile("preds_" + UUID.randomUUID(), ".csv");

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
                outputFile.delete();
                inputFile.delete();
            } catch (Exception e) {
                LogService.error("PyBridge", "Prediction failed: " + e.getMessage());
            }
        }
        return new int[features.length];
    }

    // Reuse helper logic (should be in Util class but duplicating for speed)
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
