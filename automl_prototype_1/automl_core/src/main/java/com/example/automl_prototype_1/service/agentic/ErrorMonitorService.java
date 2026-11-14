package com.example.automl_prototype_1.service.agentic;

import com.example.automl_prototype_1.service.LogService;

import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.function.Consumer;

/**
 * Monitors the application logs for errors and uses the LLM to diagnose them.
 */
public class ErrorMonitorService implements LogService.LogListener {

    private final com.example.automl_prototype_1.service.OllamaService llmService;
    private final ExecutorService analysisExecutor = Executors.newSingleThreadExecutor();
    private Consumer<String> alertCallback; // To push to UI

    // Debounce cache: hash(error_msg) -> timestamp
    private final ConcurrentHashMap<Integer, Long> lastAlertTime = new ConcurrentHashMap<>();
    private static final long COOLDOWN_MS = 60000; // 1 minute cooldown per unique error

    public ErrorMonitorService() {
        this.llmService = com.example.automl_prototype_1.service.OllamaService.getInstance();
        LogService.getInstance().addListener(this);
        LogService.info("ErrorMonitor", "Service started. Watching for critical errors...");
    }

    public void setAlertCallback(Consumer<String> callback) {
        this.alertCallback = callback;
    }

    @Override
    public void onNewLog(String message) {
        if (isCriticalError(message)) {
            processError(message);
        }
    }

    private boolean isCriticalError(String message) {
        return message.contains("ERROR") || message.contains("Exception") || message.contains("Critical");
    }

    private void processError(String logEntry) {
        int hash = logEntry.hashCode();
        long now = System.currentTimeMillis();

        // Check debounce
        if (lastAlertTime.containsKey(hash)) {
            if (now - lastAlertTime.get(hash) < COOLDOWN_MS) {
                return; // Suppress duplicate alert
            }
        }
        lastAlertTime.put(hash, now);

        // Async Analysis
        analysisExecutor.submit(() -> {
            try {
                // Phase 8: Self-Healing Check
                String failingClass = extractFailingGeneratedClass(logEntry);
                if (failingClass != null) {
                    LogService.getInstance().logAI("ErrorMonitor", "Candidate for Repair",
                            "Identified failing generated class: " + failingClass);
                    AgenticArchitectService.getInstance().attemptAutoFix(failingClass, logEntry);
                    return; // Skip diagnosis, we are fixing it.
                }

                String diagnosis = diagnoseError(logEntry);
                if (alertCallback != null && diagnosis != null) {
                    alertCallback.accept(diagnosis);
                }

                // Phase 9: Trigger Self-Correction Loop for algorithm-level failures
                // Extract algorithm name from log entry if present
                String algoName = extractAlgorithmName(logEntry);
                if (algoName != null) {
                    SelfCorrectionLoopService.getInstance()
                            .onExecutionComplete(algoName, false, "RUNTIME_ERROR", 0.0);
                    LogService.info("ErrorMonitor", "Self-correction triggered for: " + algoName);
                }
            } catch (Exception e) {
                System.err.println("ErrorMonitor diagnosis failed: " + e.getMessage());
            }
        });
    }

    /**
     * Attempts to extract an algorithm name from a log entry.
     * Returns null if no known algorithm is mentioned.
     */
    private String extractAlgorithmName(String logEntry) {
        String[] knownAlgos = { "GA", "IGPSO", "WWO", "BPSO", "ASO", "PLS", "SA", "GWO", "DE" };
        String upper = logEntry.toUpperCase();
        for (String algo : knownAlgos) {
            if (upper.contains("[" + algo + "]") || upper.contains(" " + algo + " ") ||
                    upper.contains("=" + algo + "=")) {
                return algo;
            }
        }
        return null;
    }

    private String extractFailingGeneratedClass(String logEntry) {
        if (!logEntry.contains("com.example.automl_prototype_1.rl.extensions")) {
            return null;
        }
        // Very basic parsing: look for "at
        // com.example.automl_prototype_1.rl.extensions.ClassName.method(ClassName.java:Line)"
        // or just regex for class name
        try {
            java.util.regex.Pattern p = java.util.regex.Pattern
                    .compile("com\\.example\\.automl_prototype_1\\.rl\\.extensions\\.([A-Za-z0-9_]+)");
            java.util.regex.Matcher m = p.matcher(logEntry);
            if (m.find()) {
                return m.group(1);
            }
        } catch (Exception e) {
        }
        return null;
    }

    private String diagnoseError(String errorLog) {
        if (!llmService.isAvailable())
            return null;

        String prompt = "SYSTEM: You are an Expert Java Debugger.\n" +
                "TASK: Analyze the following error log. Explain the ROOT CAUSE and suggest a FIX.\n" +
                "keep it concise (max 3 sentences).\n\n" +
                "ERROR LOG:\n" + errorLog;

        String response = llmService.queryOllama(prompt, "active");

        // Format for UI
        return String.format("[AI DIAGNOSIS]\nError: %s\n\nAnalysis: %s",
                errorLog.length() > 100 ? errorLog.substring(0, 97) + "..." : errorLog,
                response);
    }

    public void shutdown() {
        LogService.getInstance().removeListener(this);
        analysisExecutor.shutdownNow();
    }
}
