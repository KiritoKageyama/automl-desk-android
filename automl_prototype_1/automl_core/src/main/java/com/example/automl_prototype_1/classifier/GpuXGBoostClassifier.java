package com.example.automl_prototype_1.classifier;

import com.example.automl_prototype_1.service.IClassifier;
import com.example.automl_prototype_1.service.PythonExecutionService;
import com.example.automl_prototype_1.service.LogService;
import smile.classification.GradientTreeBoost;
import smile.data.DataFrame;
import smile.data.formula.Formula;
import java.io.File;
import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Properties;
import java.util.UUID;
import java.util.Scanner;

/**
 * A GPU-enabled wrapper for XGBoost Classifier.
 * Uses Python Bridge (xgboost with tree_method='gpu_hist') if available.
 * Falls back to Smile GradientTreeBoost (CPU) if Python/GPU is unavailable.
 */
public class GpuXGBoostClassifier implements IClassifier {
    private final int ntrees;
    private final int maxDepth;
    private GradientTreeBoost javaModel;
    private File pythonModelPath;
    private boolean usePython = false;

    public GpuXGBoostClassifier() {
        this(200, 3); // Default depth 3 is standard for boosting
    }

    public GpuXGBoostClassifier(int ntrees, int maxDepth) {
        this.ntrees = ntrees;
        this.maxDepth = maxDepth;
    }

    @Override
    public String getName() {
        return usePython ? "XGBoost (Python/GPU Bridge)" : "Gradient Boosting (Java CPU Fallback)";
    }

    @Override
    public void train(double[][] features, int[] labels) {
        if (features.length == 0 || features[0].length == 0) {
            return;
        }

        // 1. Try Python Bridge (Real GPU Support)
        if (PythonExecutionService.getInstance().isAvailable()) {
            try {
                LogService.info("GpuXGB", "Attempting training via Python Bridge (XGBoost GPU)...");
                File trainFile = createTempCsv(features, labels);
                File modelFile = File.createTempFile("xgb_model_" + UUID.randomUUID(), ".pkl");

                // Use the shared sklearn script which we updated to support 'xgboost'
                File scriptFile = PythonExecutionService.getInstance().extractScriptToTemp("train_sklearn.py");

                if (scriptFile.exists()) {
                    boolean success = PythonExecutionService.getInstance().trainModel(
                            scriptFile,
                            trainFile,
                            modelFile,
                            "--model_type", "xgboost",
                            "--n_estimators", String.valueOf(ntrees),
                            "--max_depth", String.valueOf(maxDepth),
                            "--learning_rate", "0.1"); // Default learning rate

                    if (success) {
                        this.pythonModelPath = modelFile;
                        this.usePython = true;
                        LogService.info("GpuXGB", "Python training successful. Model saved.");
                        trainFile.delete();
                        return;
                    }
                } else {
                    LogService.error("GpuXGB", "Python script not found at " + scriptFile.getAbsolutePath());
                }
            } catch (Exception e) {
                LogService.error("GpuXGB", "Python training failed: " + e.getMessage());
            }
        }

        // 2. Fallback to Java (Smile GBM) - CPU Only
        LogService.info("GpuXGB", "Falling back to pure Java implementation (CPU GBM).");
        this.usePython = false;

        int[][] labelMatrix = new int[labels.length][1];
        for (int i = 0; i < labels.length; i++) {
            labelMatrix[i][0] = labels[i];
        }

        DataFrame dataFrame = DataFrame.of(features).merge(DataFrame.of(labelMatrix, "Class"));
        Formula formula = Formula.of("Class", ".");
        Properties props = new Properties();
        props.setProperty("smile.gradient.boost.trees", String.valueOf(this.ntrees));
        props.setProperty("smile.gradient.boost.max.depth", String.valueOf(this.maxDepth));

        this.javaModel = GradientTreeBoost.fit(formula, dataFrame, props);
    }

    @Override
    public int[] predict(double[][] features) {
        // Python Prediction path
        if (this.usePython && this.pythonModelPath != null) {
            try {
                File inputFile = createTempCsv(features, null);
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
                    LogService.error("GpuXGB", "Python prediction failed. Returning zeros.");
                    return new int[features.length];
                }
            } catch (Exception e) {
                LogService.error("GpuXGB", "Error during Python prediction: " + e.getMessage());
                return new int[features.length];
            }
        }

        // Java Fallback
        if (this.javaModel == null) {
            return new int[features.length];
        }
        DataFrame predictionFrame = DataFrame.of(features);
        return this.javaModel.predict(predictionFrame);
    }

    // Helpers
    private File createTempCsv(double[][] features, int[] labels) throws IOException {
        File temp = File.createTempFile("data_" + UUID.randomUUID(), ".csv");
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(temp))) {
            // Header
            for (int i = 0; i < features[0].length; i++) {
                writer.write("F" + i + ",");
            }
            writer.write("Class");
            writer.newLine();

            for (int i = 0; i < features.length; i++) {
                for (int j = 0; j < features[i].length; j++) {
                    writer.write(String.valueOf(features[i][j]) + ",");
                }
                if (labels != null) {
                    writer.write(String.valueOf(labels[i]));
                } else {
                    writer.write("0");
                }
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
                } catch (NumberFormatException e) {
                    preds[i] = 0;
                }
            }
            return preds;
        }
    }
}
