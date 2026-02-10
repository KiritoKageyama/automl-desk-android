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

    private final LocalLlmService llmService;
    private final ExecutorService analysisExecutor = Executors.newSingleThreadExecutor();
    private Consumer<String> alertCallback; // To push to UI

    // Debounce cache: hash(error_msg) -> timestamp
    private final ConcurrentHashMap<Integer, Long> lastAlertTime = new ConcurrentHashMap<>();
    private static final long COOLDOWN_MS = 60000; // 1 minute cooldown per unique error

    public ErrorMonitorService() {
        this.llmService = new LocalLlmService();
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
                String diagnosis = diagnoseError(logEntry);
                if (alertCallback != null && diagnosis != null) {
                    alertCallback.accept(diagnosis);
                }
            } catch (Exception e) {
                System.err.println("ErrorMonitor diagnosis failed: " + e.getMessage());
            }
        });
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
