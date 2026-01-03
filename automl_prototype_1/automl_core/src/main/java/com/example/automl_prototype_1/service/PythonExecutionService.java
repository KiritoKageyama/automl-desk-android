package com.example.automl_prototype_1.service;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Service to execute external Python scripts for ML tasks (Bridge to
 * Scikit-learn/CuML).
 */
public class PythonExecutionService {

    private static PythonExecutionService instance;
    private Boolean pythonAvailable = null;
    private String pythonCommand = "python"; // Default, could be python3

    private PythonExecutionService() {
        checkPythonAvailability();
    }

    public static synchronized PythonExecutionService getInstance() {
        if (instance == null) {
            instance = new PythonExecutionService();
        }
        return instance;
    }

    public boolean isAvailable() {
        if (!PlatformService.isDesktop())
            return false;
        return Boolean.TRUE.equals(pythonAvailable);
    }

    private void checkPythonAvailability() {
        if (!PlatformService.isDesktop()) {
            pythonAvailable = false;
            return;
        }
        try {
            Process process = new ProcessBuilder(pythonCommand, "--version").start();
            int exitCode = process.waitFor();
            pythonAvailable = (exitCode == 0);
            if (pythonAvailable) {
                LogService.info("PythonBridge", "Python detected and ready.");
            } else {
                LogService.error("PythonBridge", "Python command failed. Bridge disabled.");
            }
        } catch (Exception e) {
            pythonAvailable = false;
            LogService.error("PythonBridge", "Python not found on PATH. Bridge disabled.");
        }
    }

    /**
     * Executes a training script.
     * 
     * @param scriptPath      Path to the python script (e.g., train_rf.py)
     * @param trainDataPath   Path to CSV (features + label)
     * @param modelOutputPath Path to save the serialized model
     * @param params          JSON-like string of parameters or separate args
     * @return true if successful
     */
    public boolean trainModel(File scriptPath, File trainDataPath, File modelOutputPath, String... args) {
        if (!isAvailable())
            return false;

        List<String> command = new ArrayList<>();
        command.add(pythonCommand);
        command.add(scriptPath.getAbsolutePath());
        command.add("--data");
        command.add(trainDataPath.getAbsolutePath());
        command.add("--output");
        command.add(modelOutputPath.getAbsolutePath());
        if (args != null) {
            command.addAll(Arrays.asList(args));
        }

        return executeProcess(command, "Training");
    }

    /**
     * Executes a prediction script.
     * 
     * @param scriptPath         Path to predict.py
     * @param modelPath          Path to saved model
     * @param inputDataPath      Path to input features CSV
     * @param outputWithPredPath Path to save output CSV
     * @return true if successful
     */
    public boolean predict(File scriptPath, File modelPath, File inputDataPath, File outputWithPredPath) {
        if (!isAvailable())
            return false;

        List<String> command = new ArrayList<>();
        command.add(pythonCommand);
        command.add(scriptPath.getAbsolutePath());
        command.add("--model");
        command.add(modelPath.getAbsolutePath());
        command.add("--input");
        command.add(inputDataPath.getAbsolutePath());
        command.add("--output");
        command.add(outputWithPredPath.getAbsolutePath());

        return executeProcess(command, "Prediction");
    }

    private boolean executeProcess(List<String> command, String operation) {
        try {
            ProcessBuilder pb = new ProcessBuilder(command);
            pb.redirectErrorStream(true); // Merge stderr into stdout
            Process process = pb.start();

            // Read output
            try (BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()))) {
                while (reader.readLine() != null) {
                    // Filter verbose python logs if needed, or just debug log
                    // LogService.debug("PythonBridge", line);
                }
            }

            int exitCode = process.waitFor();
            if (exitCode == 0) {
                LogService.info("PythonBridge", operation + " completed successfully.");
                return true;
            } else {
                LogService.error("PythonBridge", operation + " failed with exit code " + exitCode);
                return false;
            }
        } catch (Exception e) {
            LogService.error("PythonBridge", operation + " failed: " + e.getMessage());
            return false;
        }
    }

    // Helper to extract resource script to temp file
    public File extractScriptToTemp(String resourceName) throws IOException {
        // Implementation later if scripts are embedded in JAR
        // For development, we assume running from IDE or resources are accessible
        return null; // Placeholder
    }
}
