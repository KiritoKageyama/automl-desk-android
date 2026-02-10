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
