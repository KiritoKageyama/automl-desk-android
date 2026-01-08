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

        boolean useWsl = AppStateService.getInstance().isUseWsl();
        List<String> command = new ArrayList<>();

        if (useWsl) {
            // Dynamic Wrapper Generation to avoid "Script not found" and CRLF issues
            try {
                File wrapperFile = createWslWrapperScript();
                // Explicitly use bash to execute the shell script
                command.add("wsl");
                if (AppStateService.getInstance().getWslDistributionName() != null) {
                    command.add("-d");
                    command.add(AppStateService.getInstance().getWslDistributionName());
                }
                command.add("bash");
                command.add(convertToWslPath(wrapperFile.getAbsolutePath()));
            } catch (IOException e) {
                LogService.error("PythonBridge", "Failed to create temp wrapper script: " + e.getMessage());
                return false;
            }
        } else {
            command.add(pythonCommand);
        }

        // Handle Paths
        String scriptP = useWsl ? convertToWslPath(scriptPath.getAbsolutePath()) : scriptPath.getAbsolutePath();
        String dataP = useWsl ? convertToWslPath(trainDataPath.getAbsolutePath()) : trainDataPath.getAbsolutePath();
        String outP = useWsl ? convertToWslPath(modelOutputPath.getAbsolutePath()) : modelOutputPath.getAbsolutePath();

        command.add(scriptP);
        command.add("--data");
        command.add(dataP);
        command.add("--output");
        command.add(outP);
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

        boolean useWsl = AppStateService.getInstance().isUseWsl();
        List<String> command = new ArrayList<>();

        if (useWsl) {
            // Dynamic Wrapper Generation to avoid "Script not found" and CRLF issues
            try {
                File wrapperFile = createWslWrapperScript();
                // Explicitly use bash to execute the shell script
                command.add("wsl");
                if (AppStateService.getInstance().getWslDistributionName() != null) {
                    command.add("-d");
                    command.add(AppStateService.getInstance().getWslDistributionName());
                }
                command.add("bash");
                command.add(convertToWslPath(wrapperFile.getAbsolutePath()));
            } catch (IOException e) {
                LogService.error("PythonBridge", "Failed to create temp wrapper script: " + e.getMessage());
                return false;
            }
        } else {
            command.add(pythonCommand);
        }

        String scriptP = useWsl ? convertToWslPath(scriptPath.getAbsolutePath()) : scriptPath.getAbsolutePath();
        String modelP = useWsl ? convertToWslPath(modelPath.getAbsolutePath()) : modelPath.getAbsolutePath();
        String inputP = useWsl ? convertToWslPath(inputDataPath.getAbsolutePath()) : inputDataPath.getAbsolutePath();
        String outP = useWsl ? convertToWslPath(outputWithPredPath.getAbsolutePath())
                : outputWithPredPath.getAbsolutePath();

        command.add(scriptP);
        command.add("--model");
        command.add(modelP);
        command.add("--input");
        command.add(inputP);
        command.add("--output");
        command.add(outP);

        return executeProcess(command, "Prediction");
    }

    private boolean executeProcess(List<String> command, String operation) {
        try {
            LogService.info("PythonBridge", "Exec: " + String.join(" ", command));
            ProcessBuilder pb = new ProcessBuilder(command);
            pb.redirectErrorStream(true); // Merge stderr into stdout
            Process process = pb.start();

            // Read output
            try (BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()))) {
                String line;
                while ((line = reader.readLine()) != null) {
                    if (AppStateService.getInstance().isLoggingEnabled()) {
                        LogService.info("PythonBridge", "[DEBUG] " + line);
                    }
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

    /**
     * Converts a Windows path (C:\Foo\Bar) to a WSL path (/mnt/c/Foo/Bar).
     */
    private String convertToWslPath(String windowsPath) {
        String drive = windowsPath.substring(0, 1).toLowerCase();
        // Remove drive letter and colon (e.g. "C:")
        String path = windowsPath.substring(3).replace("\\", "/");
        // User's Ubuntu WSL mounts C: at /mnt/c
        return "/mnt/" + drive + "/" + path;
    }

    private File getProjectTempDir() {
        File projectRoot = new File(System.getProperty("user.dir"));
        File tempDir = new File(projectRoot, "wsl_temp");
        if (!tempDir.exists()) {
            tempDir.mkdirs();
        }
        return tempDir;
    }

    // Helper to extract resource script to temp file
    public File extractScriptToTemp(String resourceName) throws IOException {
        String resourcePath = "/python/" + resourceName;
        java.io.InputStream in = getClass().getResourceAsStream(resourcePath);
        if (in == null) {
            // Try without /python prefix just in case
            resourcePath = "/" + resourceName;
            in = getClass().getResourceAsStream(resourcePath);
            if (in == null) {
                throw new java.io.FileNotFoundException("Resource not found: " + resourceName);
            }
        }

        // Create temp file in project-local temp dir
        File tempFile = File.createTempFile("automl_script_" + resourceName, ".py", getProjectTempDir());
        tempFile.deleteOnExit();

        try (java.io.FileOutputStream out = new java.io.FileOutputStream(tempFile)) {
            byte[] buffer = new byte[1024];
            int bytesRead;
            while ((bytesRead = in.read(buffer)) != -1) {
                out.write(buffer, 0, bytesRead);
            }
        } finally {
            in.close();
        }
        return tempFile;
    }

    private File createWslWrapperScript() throws IOException {
        // Create temp file in project-local temp dir
        File tempFile = File.createTempFile("run_wsl_gpu", ".sh", getProjectTempDir());
        tempFile.deleteOnExit();

        // Write content with explicit \n line endings (LF) for Linux compatibility
        try (java.io.BufferedWriter out = new java.io.BufferedWriter(new java.io.FileWriter(tempFile))) {
            out.write("#!/bin/bash\n");
            out.write("PYTHON_EXEC=\"$HOME/miniconda/envs/automl_gpu_env/bin/python\"\n");
            out.write("if [ ! -f \"$PYTHON_EXEC\" ]; then\n");
            out.write("    PYTHON_EXEC=\"$HOME/miniconda3/envs/automl_gpu_env/bin/python\"\n");
            out.write("fi\n");
            out.write("\n");
            out.write("if [ -f \"$PYTHON_EXEC\" ]; then\n");
            out.write("    \"$PYTHON_EXEC\" \"$@\"\n");
            out.write("else\n");
            out.write(
                    "    source \"$HOME/miniconda/etc/profile.d/conda.sh\" 2>/dev/null || source \"$HOME/miniconda3/etc/profile.d/conda.sh\" 2>/dev/null\n");
            out.write("    conda activate automl_gpu_env\n");
            out.write("    python \"$@\"\n");
            out.write("fi\n");
        }
        return tempFile;
    }
}
