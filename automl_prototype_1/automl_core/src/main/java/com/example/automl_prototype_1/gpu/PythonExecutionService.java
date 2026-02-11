package com.example.automl_prototype_1.gpu;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import com.example.automl_prototype_1.service.LogService;
import com.example.automl_prototype_1.service.AppStateService;
import com.example.automl_prototype_1.service.PlatformService;

/**
 * Service to execute external Python scripts for ML tasks (Bridge to
 * Scikit-learn/CuML).
 */
public class PythonExecutionService {

    private static PythonExecutionService instance;
    private Boolean pythonAvailable = null;
    private String pythonCommand = "python"; // Default, could be python3

    // Lifecycle Management
    private final java.util.concurrent.ScheduledExecutorService scheduler = java.util.concurrent.Executors
            .newSingleThreadScheduledExecutor();
    private final java.util.concurrent.atomic.AtomicLong lastActivityTime = new java.util.concurrent.atomic.AtomicLong(
            System.currentTimeMillis());
    private static final long IDLE_TIMEOUT_MS = 30000; // 30 Seconds Timeout (Increased from 5s to prevent race
                                                       // conditions)
    private volatile boolean isMonitoring = false;

    private PythonExecutionService() {
        cleanupTempDir();
        checkPythonAvailability();
        // Register Shutdown Hook to ensure cleanup if JVM is killed
        Runtime.getRuntime().addShutdownHook(new Thread(() -> {
            System.out.println("[PythonBridge] Shutdown hook triggered. Cleaning up...");
            terminateAll();
            stopGpuServer(); // Explicitly kill server too
        }));
        startIdleMonitor();
    }

    public void recordActivity() {
        lastActivityTime.set(System.currentTimeMillis());
    }

    private void startIdleMonitor() {
        if (isMonitoring)
            return;
        isMonitoring = true;
        scheduler.scheduleAtFixedRate(() -> {
            if (serverProcess != null && serverProcess.isAlive()) {
                // Check if queue is empty via GpuBatchService proxy check
                // We rely on activity timestamp. If no activity for 5s, we imply queue is done.
                long idleTime = System.currentTimeMillis() - lastActivityTime.get();
                if (idleTime > IDLE_TIMEOUT_MS) {
                    if (GpuBatchService.getInstance().isQueueEmpty()) {
                        LogService.info("PythonBridge", "GPU Server idle for 5s. Shutting down (Aggressive Cleanup).");
                        stopGpuServer();
                    } else {
                        // Queue not empty, pretend we are active
                        // (Ideally GpuBatchService calls recordActivity(), but this is a safety net)
                        recordActivity();
                    }
                }
            }
        }, 1, 1, java.util.concurrent.TimeUnit.SECONDS);
    }

    private void cleanupTempDir() {
        try {
            File tempDir = getProjectTempDir();
            if (tempDir.exists() && tempDir.isDirectory()) {
                File[] files = tempDir.listFiles();
                if (files != null) {
                    for (File f : files) {
                        try {
                            if (f.isFile()) {
                                f.delete();
                            }
                        } catch (Exception ignored) {
                        }
                    }
                }
                LogService.info("PythonBridge", "Cleaned up wsl_temp directory.");
            }
        } catch (Exception e) {
            System.err.println("[PythonBridge] Warning: Failed to clean temp dir: " + e.getMessage());
        }
    }

    public static synchronized PythonExecutionService getInstance() {
        if (instance == null) {
            instance = new PythonExecutionService();
        }
        return instance;
    }

    private static final boolean FORCE_VISIBLE_TERMINAL = false;

    private final java.util.Set<Process> activeProcesses = java.util.Collections
            .synchronizedSet(new java.util.HashSet<>());

    public synchronized void terminateAll() {
        synchronized (activeProcesses) {
            if (!activeProcesses.isEmpty()) {
                LogService.info("PythonBridge",
                        "Force termination requested. Killing " + activeProcesses.size()
                                + " active processes (and descendants)...");
                for (Process p : activeProcesses) {
                    if (p.isAlive()) {
                        p.toHandle().descendants().forEach(ProcessHandle::destroyForcibly);
                        p.destroyForcibly();
                    }
                }
                activeProcesses.clear();
            }
            // Explicitly kill any orphaned WSL processes (crucial for visible terminal
            // mode)
            try {
                new ProcessBuilder("taskkill", "/F", "/IM", "wsl.exe", "/IM", "wslhost.exe").start().waitFor();
                LogService.info("PythonBridge", "Enforced cleanup of WSL processes via taskkill.");
            } catch (Exception e) {
                // Ignore if no processes found
            }

            // Clean up temporary files (ONNX models, etc.)
            cleanupTempDir();
        }
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
        File wrapperFile = null;

        try {
            if (useWsl) {
                // Dynamic Wrapper Generation to avoid "Script not found" and CRLF issues
                try {
                    wrapperFile = createWslWrapperScript();

                    if (FORCE_VISIBLE_TERMINAL) {
                        command.add("cmd.exe");
                        command.add("/c");
                        command.add("start");
                        command.add("AutoML_GPU_Job");
                        command.add("/wait");
                    }

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
                command.add("-u"); // Force unbuffered output
            }

            // Handle Paths
            String scriptP = useWsl ? convertToWslPath(scriptPath.getAbsolutePath()) : scriptPath.getAbsolutePath();
            String dataP = useWsl ? convertToWslPath(trainDataPath.getAbsolutePath()) : trainDataPath.getAbsolutePath();
            String outP = useWsl ? convertToWslPath(modelOutputPath.getAbsolutePath())
                    : modelOutputPath.getAbsolutePath();

            command.add(scriptP);
            command.add("--data");
            command.add(dataP);
            command.add("--output");
            command.add(outP);
            if (args != null) {
                command.addAll(Arrays.asList(args));
            }

            return executeProcess(command, "Training");
        } finally {
            if (wrapperFile != null && wrapperFile.exists()) {
                wrapperFile.delete();
            }
        }
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
        File wrapperFile = null;

        try {
            if (useWsl) {
                // Dynamic Wrapper Generation to avoid "Script not found" and CRLF issues
                try {
                    wrapperFile = createWslWrapperScript();

                    if (FORCE_VISIBLE_TERMINAL) {
                        command.add("cmd.exe");
                        command.add("/c");
                        command.add("start");
                        command.add("AutoML_GPU_Pred");
                        command.add("/wait");
                    }

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
            String inputP = useWsl ? convertToWslPath(inputDataPath.getAbsolutePath())
                    : inputDataPath.getAbsolutePath();
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
        } finally {
            if (wrapperFile != null && wrapperFile.exists()) {
                wrapperFile.delete();
            }
        }
    }

    private boolean executeProcess(List<String> command, String operation) {
        Process process = null;
        try {
            LogService.info("PythonBridge", "Exec: " + String.join(" ", command));
            ProcessBuilder pb = new ProcessBuilder(command);
            pb.redirectErrorStream(true); // Merge stderr into stdout
            process = pb.start();
            activeProcesses.add(process);

            List<String> outputBuffer = new ArrayList<>();

            // Read output
            try (BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()))) {
                String line;
                while ((line = reader.readLine()) != null) {
                    outputBuffer.add(line);
                    if (AppStateService.getInstance().isLoggingEnabled()) {
                        LogService.info("PythonBridge", "[DEBUG] " + line);
                    }
                }
            }

            // Wait with timeout (e.g., 2 hours max per script to prevent infinite hangs)
            // This protects against Python script hanging indefinitely without output
            boolean finished = process.waitFor(120, java.util.concurrent.TimeUnit.MINUTES);
            if (!finished) {
                LogService.error("PythonBridge", operation + " timed out after 120 minutes. Terminating...");
                process.destroyForcibly();
                return false;
            }
            int exitCode = process.exitValue();
            if (exitCode == 0) {
                LogService.info("PythonBridge", operation + " completed successfully.");
                return true;
            } else {
                LogService.error("PythonBridge", operation + " failed with exit code " + exitCode);
                LogService.error("PythonBridge", "--- Begin Error Output ---");
                for (String line : outputBuffer) {
                    LogService.error("PythonBridge", line);
                }
                LogService.error("PythonBridge", "--- End Error Output ---");
                return false;
            }
        } catch (InterruptedException ie) {
            LogService.error("PythonBridge", operation + " interrupted (STOPPED BY USER).");
            return false;
        } catch (Exception e) {
            LogService.error("PythonBridge", operation + " failed: " + e.getMessage());
            return false;
        } finally {
            if (process != null) {
                activeProcesses.remove(process);
                // Ensure it's dead if we got here via exception
                if (process.isAlive()) {
                    process.destroyForcibly();
                }
            }
        }
    }

    /**
     * Converts a Windows path (C:\Foo\Bar) to a WSL path (/mnt/c/Foo/Bar).
     */
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

    /**
     * Converts a WSL path (/mnt/c/Foo/Bar) back to a Windows path (C:\Foo\Bar).
     */
    public String convertFromWslPath(String wslPath) {
        if (wslPath == null || !wslPath.startsWith("/mnt/")) {
            return wslPath; // Not a WSL path or unknown mount
        }
        // Format: /mnt/c/Users/...
        // drive char is at index 5 ('c')
        char drive = wslPath.charAt(5);
        String path = wslPath.substring(7); // Skip "/mnt/c/"
        return Character.toUpperCase(drive) + ":\\" + path.replace("/", "\\");
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

        // Phase 6, Step 3: Explicit Resource Caps
        // Limit Python process to 8GB Virtual Memory (Software Cap)
        // This prevents OS Thrashing/Freezing if Python leaks or tries to load too much
        long memoryLimitKb = 8 * 1024 * 1024;

        // Write content with explicit \n line endings (LF) for Linux compatibility
        try (java.io.BufferedWriter out = new java.io.BufferedWriter(new java.io.FileWriter(tempFile))) {
            out.write("#!/bin/bash\n");

            // Hard Cap Memory before launching Python
            out.write("# --- Phase 6: Resource Cap ---\n");
            out.write("ulimit -v " + memoryLimitKb + "\n");
            out.write("export OMP_NUM_THREADS=4\n"); // Limit OpenMP threads
            out.write(
                    "echo \"[Wrapper] Enforced Virtual Memory Limit: " + (memoryLimitKb / 1024) + "MB, Threads: 4\"\n");
            out.write("# -----------------------------\n");

            out.write("PYTHON_EXEC=\"$HOME/miniconda/envs/automl_gpu_env/bin/python\"\n");
            out.write("if [ ! -f \"$PYTHON_EXEC\" ]; then\n");
            out.write("    PYTHON_EXEC=\"$HOME/miniconda3/envs/automl_gpu_env/bin/python\"\n");
            out.write("fi\n");
            out.write("\n");
            out.write("if [ -f \"$PYTHON_EXEC\" ]; then\n");
            out.write("    echo \"[Wrapper] Direct python found at $PYTHON_EXEC\"\n");
            out.write("    \"$PYTHON_EXEC\" -u \"$@\"\n");
            out.write("else\n");
            out.write("    echo \"[Wrapper] Direct python NOT found. Using Conda fallback...\"\n");
            out.write(
                    "    source \"$HOME/miniconda/etc/profile.d/conda.sh\" 2>/dev/null || source \"$HOME/miniconda3/etc/profile.d/conda.sh\" 2>/dev/null\n");
            out.write("    echo \"[Wrapper] Conda sourced. Activating env...\"\n");
            out.write("    conda activate automl_gpu_env\n");
            out.write("    echo \"[Wrapper] Env activated. Running python...\"\n");
            out.write("    python -u \"$@\"\n");
            out.write("fi\n");
            out.write("exit $?\n"); // Propagate exit code
        }
        return tempFile;
    }

    // --- Server Management ---
    private volatile Process serverProcess;
    // private static final int SERVER_PORT = 5000; // Unused

    public boolean isServerRunning() {
        return serverProcess != null && serverProcess.isAlive();
    }

    public synchronized void startGpuServer() {
        if (!isAvailable())
            return;

        recordActivity(); // CRITICAL: Reset idle timer so we don't kill ourselves immediately on startup

        if (serverProcess != null && serverProcess.isAlive()) {
            LogService.info("PythonBridge", "GPU Server already running.");
            return;
        }

        try {
            // Priority Step: Kill any existing server on port 5000 (Zombie cleanup)
            // Priority Step: Kill any existing server on port 5000 (Zombie cleanup)
            if (AppStateService.getInstance().isUseWsl()) {
                try {
                    String distro = AppStateService.getInstance().getWslDistributionName();

                    // Force kill everything on port 5000
                    List<String> killCmd = new ArrayList<>();
                    killCmd.add("wsl");
                    if (distro != null) {
                        killCmd.add("-d");
                        killCmd.add(distro);
                    }
                    killCmd.add("bash");
                    killCmd.add("-c");
                    // Use fuser -k -9 (SIGKILL) to be absolutely sure
                    killCmd.add("fuser -k -9 5006/tcp || true; pkill -9 -f gpu_server.py || true");

                    ProcessBuilder pbKill = new ProcessBuilder(killCmd);
                    pbKill.start().waitFor();
                    LogService.info("PythonBridge", "Aggressively cleaned up port 5000.");

                    // Wait a moment for OS to release port
                    Thread.sleep(1000);

                } catch (Exception e) {
                    // Ignore cleanup errors
                }
            } else {
                // Native Windows Cleanup
                try {
                    // Only kill if we are tracking a process, relying on terminateAll() generally.
                    // The global kill is too dangerous for users with other Python work.
                    if (serverProcess != null && serverProcess.isAlive()) {
                        serverProcess.destroyForcibly();
                    }
                } catch (Exception e) {
                    // Ignore
                }
            }

            LogService.info("PythonBridge", "Starting GPU Server...");
            File scriptFile = extractScriptToTemp("gpu_server.py");

            List<String> command = new ArrayList<>();
            boolean useWsl = AppStateService.getInstance().isUseWsl();

            if (useWsl) {
                // Use the same wrapper generation logic as train/predict
                File wrapperFile = createWslWrapperScript();
                command.add("wsl");
                if (AppStateService.getInstance().getWslDistributionName() != null) {
                    command.add("-d");
                    command.add(AppStateService.getInstance().getWslDistributionName());
                }
                command.add("bash");
                command.add(convertToWslPath(wrapperFile.getAbsolutePath()));
                command.add(convertToWslPath(scriptFile.getAbsolutePath()));
            } else {
                command.add(pythonCommand);
                command.add("-u");
                command.add(scriptFile.getAbsolutePath());
            }

            ProcessBuilder pb = new ProcessBuilder(command);
            pb.redirectErrorStream(true);
            serverProcess = pb.start();
            activeProcesses.add(serverProcess);

            // Consume stdout in background to prevent buffer blocking
            new Thread(() -> {
                try (BufferedReader reader = new BufferedReader(
                        new InputStreamReader(serverProcess.getInputStream()))) {
                    String line;
                    while ((line = reader.readLine()) != null) {
                        if (AppStateService.getInstance().isLoggingEnabled()) {
                            LogService.info("GpuServer", line);
                        }
                    }
                } catch (IOException e) {
                    /* Process died */ }
            }).start();

            // Robust Wait for Server Port
            int retries = 0;
            boolean connected = false;
            LogService.info("PythonBridge", "Waiting for GPU Server port 5006...");

            while (retries < 100 && !connected) { // INCREASED: Wait up to 100 seconds
                try (java.net.Socket s = new java.net.Socket("127.0.0.1", 5006)) {
                    connected = true;
                    LogService.info("PythonBridge", "GPU Server Port 5006 is ACTIVE.");
                } catch (IOException e) {
                    try {
                        Thread.sleep(1000);
                        if (retries % 5 == 0) {
                            LogService.info("PythonBridge", "Waiting for server... (" + retries + "/100)");
                        }
                    } catch (InterruptedException ie) {
                    }
                    retries++;
                    recordActivity(); // CRITICAL: Keep us alive while we wait!
                }
            }

            if (!connected) {
                LogService.error("PythonBridge", "GPU Server did not bind to port 5006 after 60 seconds.");
            }

        } catch (Exception e) {
            LogService.error("PythonBridge", "Failed to start GPU Server: " + e.getMessage());
        }
    }

    // --- Bridge for IClassifier types to GpuBatchService ---

    // --- Arrow & ONNX Integration ---

    public void trainModel(com.example.automl_prototype_1.service.IClassifier classifier, double[][] features,
            int[] labels, java.util.Map<String, Object> params) {
        String algoType;
        if (classifier instanceof com.example.automl_prototype_1.classifier.MlpClassifier) {
            algoType = "MLP";
        } else if (classifier.getName().contains("Random Forest")) {
            algoType = "RF";
        } else if (classifier instanceof com.example.automl_prototype_1.classifier.CustomScriptClassifier) {
            algoType = "CUSTOM_SCRIPT";
            if (params == null)
                params = new java.util.HashMap<>();
            params.put("script_path",
                    ((com.example.automl_prototype_1.classifier.CustomScriptClassifier) classifier).getScriptPath());
        } else if (classifier instanceof com.example.automl_prototype_1.classifier.CnnClassifier) {
            algoType = "CUSTOM_SCRIPT";
            if (params == null)
                params = new java.util.HashMap<>();
            params.put("script_path",
                    ((com.example.automl_prototype_1.classifier.CnnClassifier) classifier).getScriptPath());
        } else if (classifier instanceof com.example.automl_prototype_1.classifier.BertClassifier) {
            algoType = "BERT";
        } else if (classifier instanceof com.example.automl_prototype_1.classifier.LstmClassifier) {
            algoType = "LSTM";
        } else {
            algoType = "RF"; // Default
        }

        // --- Arrow Optimization ---
        // If Arrow is available and the algorithm is standard (RF, SVM, etc), use fast
        // path.
        // Custom scripts or image data (list of paths) might stick to JSON/CSV for now.
        boolean useArrow = true;

        // Disable Arrow for Custom Scripts, CNN (paths), and BERT (paths) as
        // ArrowService is currently double[][] only
        if (algoType.equals("CUSTOM_SCRIPT") || algoType.equals("BERT")) {
            useArrow = false;
        }

        if (useArrow) {
            try {
                trainModelArrow(classifier, features, labels, params, algoType);
                return;
            } catch (Exception e) {
                LogService.error("PythonBridge",
                        "Arrow Training Failed, falling back to JSON Batch: " + e.getMessage());
                // Fallthrough to standard logic
            }
        }

        try {
            java.util.concurrent.CompletableFuture<Object> future = GpuBatchService.getInstance().submitTrain(algoType,
                    features, labels, params);

            Object result = future.get();
            String modelId = (String) result;

            // TODO: GpuBatchService signature doesn't propagate the FULL result object
            // (like onnx_path)
            // It just returns model_id string.
            // To support ONNX via Batch, we'd need to update GpuBatchService.
            // For now, Arrow path below HANDLES ONNX registration directly.
            // Legacy JSON path might miss ONNX registration unless we refactor
            // GpuBatchService.

            modelIdMap.put(classifier, modelId);
            LogService.info("PythonBridge", "Trained " + algoType + " model (Legacy JSON). ID: " + modelId);

        } catch (Exception e) {
            LogService.error("PythonBridge", "Failed to train " + algoType + ": " + e.getMessage());
            throw new RuntimeException(e);
        }
    }

    @SuppressWarnings("unchecked")
    private void trainModelArrow(com.example.automl_prototype_1.service.IClassifier classifier, double[][] features,
            int[] labels, java.util.Map<String, Object> params, String algoType) throws Exception {

        LogService.info("PythonBridge", "Starting Arrow Training for " + algoType);

        // 1. Serialize
        byte[] arrowData = com.example.automl_prototype_1.service.pipeline.ArrowService.getInstance()
                .serializeTrainData(features, labels);

        // 2. HTTP POST to /train_arrow
        java.net.URL url = java.net.URI.create("http://127.0.0.1:5006/train_arrow").toURL();
        java.net.HttpURLConnection con = (java.net.HttpURLConnection) url.openConnection();
        con.setRequestMethod("POST");
        con.setRequestProperty("Content-Type", "application/octet-stream");
        con.setRequestProperty("Content-Length", String.valueOf(arrowData.length));
        con.setRequestProperty("X-Algo-Type", algoType);

        if (params == null)
            params = new java.util.HashMap<>();
        com.fasterxml.jackson.databind.ObjectMapper mapper = new com.fasterxml.jackson.databind.ObjectMapper();
        con.setRequestProperty("X-Params", mapper.writeValueAsString(params));

        con.setDoOutput(true);
        try (java.io.OutputStream os = con.getOutputStream()) {
            os.write(arrowData);
        }

        // 3. Read Response
        if (con.getResponseCode() == 200) {
            try (java.io.BufferedReader br = new java.io.BufferedReader(
                    new java.io.InputStreamReader(con.getInputStream(), "utf-8"))) {
                StringBuilder response = new StringBuilder();
                String line;
                while ((line = br.readLine()) != null)
                    response.append(line.trim());

                java.util.Map<String, Object> respMap = mapper.readValue(response.toString(),
                        new com.fasterxml.jackson.core.type.TypeReference<java.util.Map<String, Object>>() {
                        });

                // Expect key "results": [ { "model_id": ..., "onnx_path": ..., "success": ... }
                // ]
                java.util.List<java.util.Map<String, Object>> results = (java.util.List<java.util.Map<String, Object>>) respMap
                        .get("results");

                if (results != null && !results.isEmpty()) {
                    java.util.Map<String, Object> res = results.get(0);
                    if (Boolean.TRUE.equals(res.get("success"))) {
                        String modelId = (String) res.get("model_id");
                        String onnxPath = (String) res.get("onnx_path");

                        modelIdMap.put(classifier, modelId);

                        if (onnxPath != null) {
                            LogService.info("PythonBridge", "ONNX Model Available: " + onnxPath);
                            try {
                                com.example.automl_prototype_1.service.pipeline.OnnxInferenceService.getInstance()
                                        .loadModel(modelId, onnxPath);
                                LogService.info("PythonBridge", "ONNX Model Loaded for Local Inference.");
                            } catch (Exception e) {
                                LogService.error("PythonBridge", "Failed to load ONNX model: " + e.getMessage());
                            }
                        }
                    } else {
                        throw new RuntimeException("Server Error: " + res.get("error"));
                    }
                }
            }
        } else {
            throw new RuntimeException("HTTP connection failed: " + con.getResponseCode());
        }
    }

    private final java.util.Map<com.example.automl_prototype_1.service.IClassifier, String> modelIdMap = java.util.Collections
            .synchronizedMap(new java.util.WeakHashMap<>());

    public int[] predict(com.example.automl_prototype_1.service.IClassifier classifier, double[][] features) {
        String modelId = modelIdMap.get(classifier);
        if (modelId == null) {
            throw new RuntimeException("Model not trained or ID lost.");
        }

        // 1. Try ONNX Local Inference
        try {
            return com.example.automl_prototype_1.service.pipeline.OnnxInferenceService.getInstance().predict(modelId,
                    features);
        } catch (Exception e) {
            // LogService.debug("PythonBridge", "ONNX Inference unavailable/failed (" +
            // e.getMessage() + "). Using HTTP fallback.");
        }

        // 2. Fallback to HTTP Batch
        try {
            java.util.concurrent.CompletableFuture<Object> future = GpuBatchService.getInstance().submitPredict(modelId,
                    features);
            return (int[]) future.get();
        } catch (Exception e) {
            LogService.error("PythonBridge", "Failed to predict: " + e.getMessage());
            throw new RuntimeException(e);
        }
    }

    public boolean exportModel(com.example.automl_prototype_1.service.IClassifier classifier, String destinationPath) {
        String modelId = modelIdMap.get(classifier);
        if (modelId == null) {
            LogService.error("PythonBridge", "Cannot export: Model ID not found (is it trained?)");
            return false;
        }

        try {
            // We need to send a synchronous request to GpuServer (bypassing BatchService
            // for simplicity, or add explicit action to Batch)
            // Since GpuBatchService handles the queue, we should probably add a generic
            // 'submitAction' to it?
            // Or just piggyback on a custom "EXPORT" job type?
            // Let's do a direct HTTP call for this infrequent operation, BUT we need to be
            // careful of the queue lock.

            // Safer: Add submitExport to GpuBatchService?
            // Or just do a quick ad-hoc sync request if queue is empty?
            // Let's implement a simple direct post here since GpuBatchService isn't easily
            // extensible in this specific Step.
            // Wait, we can use the same pattern:

            java.util.Map<String, Object> payload = new java.util.HashMap<>();
            payload.put("action", "EXPORT_MODEL");
            payload.put("model_id", modelId);

            // We need X_sample for shape inference?
            // We'll rely on server fallback or cached dataset.

            // Perform HTTP POST
            java.net.URL url = java.net.URI.create("http://127.0.0.1:5006").toURL();
            java.net.HttpURLConnection con = (java.net.HttpURLConnection) url.openConnection();
            con.setRequestMethod("POST");
            con.setRequestProperty("Content-Type", "application/json");
            con.setDoOutput(true);

            com.fasterxml.jackson.databind.ObjectMapper mapper = new com.fasterxml.jackson.databind.ObjectMapper();
            String jsonBody = mapper.writeValueAsString(payload);

            try (java.io.OutputStream os = con.getOutputStream()) {
                os.write(jsonBody.getBytes("utf-8"));
            }

            if (con.getResponseCode() == 200) {
                try (java.io.BufferedReader br = new java.io.BufferedReader(
                        new java.io.InputStreamReader(con.getInputStream(), "utf-8"))) {
                    StringBuilder response = new StringBuilder();
                    String line;
                    while ((line = br.readLine()) != null)
                        response.append(line.trim());

                    java.util.Map<String, Object> respMap = mapper.readValue(response.toString(),
                            new com.fasterxml.jackson.core.type.TypeReference<java.util.Map<String, Object>>() {
                            });
                    if (Boolean.TRUE.equals(respMap.get("success"))) {
                        String tempPath = (String) respMap.get("path");

                        // Move file from tempPath to destinationPath
                        File source = new File(tempPath);

                        // Fix WSL path if needed
                        if (AppStateService.getInstance().isUseWsl() && tempPath.startsWith("/mnt/")) {
                            // Need to convert WSL path to Windows path for Java to read?
                            // No, if it's in wsl_temp (project dir), it's visible to Windows.
                            // But server returned a WSL path.
                            String winPath = convertFromWslPath(tempPath);
                            source = new File(winPath);
                        }

                        File dest = new File(destinationPath);
                        if (source.exists()) {
                            java.nio.file.Files.copy(source.toPath(), dest.toPath(),
                                    java.nio.file.StandardCopyOption.REPLACE_EXISTING);
                            LogService.info("PythonBridge", "Model exported to: " + destinationPath);
                            return true;
                        } else {
                            LogService.error("PythonBridge", "Exported file not found at source: " + source);
                        }
                    }
                }
            } else {
                LogService.error("PythonBridge", "Export Request Failed: " + con.getResponseMessage());
            }

        } catch (Exception e) {
            LogService.error("PythonBridge", "Export Exception: " + e.getMessage());
        }
        return false;
    }

    public synchronized void stopGpuServer() {
        if (serverProcess != null) {
            LogService.info("PythonBridge", "Stopping GPU Server...");

            if (AppStateService.getInstance().isUseWsl()) {
                try {
                    String distro = AppStateService.getInstance().getWslDistributionName();
                    List<String> cmd = new ArrayList<>();
                    cmd.add("wsl");
                    if (distro != null) {
                        cmd.add("-d");
                        cmd.add(distro);
                    }
                    cmd.add("bash");
                    cmd.add("-c");
                    // Authoritative Kill: pkill -9 is essential for Python scripts that handle
                    // signals poorly
                    cmd.add("pkill -9 -f gpu_server.py; fuser -k -9 5006/tcp");

                    new ProcessBuilder(cmd).start().waitFor();
                    LogService.info("PythonBridge", "Sent SIGKILL to WSL processes.");
                } catch (Exception e) {
                    LogService.error("PythonBridge", "Failed to kill WSL process: " + e.getMessage());
                }
            }

            serverProcess.destroyForcibly();
            serverProcess = null;
        }
        // Do NOT shutdown GpuBatchService dispatcher generally, just the server
        // connection.
        // But if the app is closing, we might want to.
        // For idle cleanup, we keep GpuBatchService alive to restart server if needed.
    }

    public synchronized void restartGpuServer() {
        // Prevent rapid cycling: if cleaned up recently, wait?
        // Simple logic: Stop then Start
        if (serverProcess != null && serverProcess.isAlive()) {
            // Already running? Maybe check health?
            // For now, assume if this is called, the caller has good reason (failed
            // connection).
            stopGpuServer();
        }
        startGpuServer();
    }
}
