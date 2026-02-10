package com.example.automl_prototype_1.service;

import java.util.concurrent.ConcurrentLinkedQueue;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;

/**
 * Singleton service for buffering execution logs and generating post-execution
 * reports.
 * Used to provide transparency without cluttering the UI during runtime.
 */
public class LogService {
    private static final LogService instance = new LogService();
    private final ConcurrentLinkedQueue<String> logBuffer = new ConcurrentLinkedQueue<>();
    private final DateTimeFormatter timeFormatter = DateTimeFormatter.ofPattern("HH:mm:ss.SSS");

    private LogService() {
        log("System", "Log Service Initialized.");
    }

    public static LogService getInstance() {
        return instance;
    }

    /**
     * Clears the log buffer. Call this before a new experiment run.
     */
    public void clear() {
        logBuffer.clear();
        log("System", "Log Buffer Cleared. Ready for new execution.");
    }

    /**
     * Logs a message with a specific source/context.
     * 
     * @param source  The component generating the log (e.g., "GA",
     *                "ExecutionService").
     * @param message The details of the operation.
     */

    /**
     * Helper for informational messages.
     */
    public static void info(String source, String message) {
        getInstance().log(source, message);
    }

    public interface LogListener {
        void onNewLog(String message);
    }

    private final java.util.List<LogListener> listeners = new java.util.concurrent.CopyOnWriteArrayList<>();

    public void addListener(LogListener listener) {
        if (listener != null)
            listeners.add(listener);
    }

    public void removeListener(LogListener listener) {
        listeners.remove(listener);
    }

    /**
     * Helper for error messages.
     */
    public static void error(String source, String message) {
        getInstance().log(source, "ERROR: " + message);
    }

    /**
     * Helper for warning messages.
     */
    public static void warn(String source, String message) {
        getInstance().log(source, "WARN: " + message);
    }

    /**
     * Logs a message with a specific source/context.
     * 
     * @param source  The component generating the log (e.g., "GA",
     *                "ExecutionService").
     * @param message The details of the operation.
     */
    private String lastLogMessage = "";
    private int duplicateCount = 0;
    private static final int MAX_DUPLICATES_SHOWN = 5;

    public void log(String source, String message) {
        // Respect the "Detailed Logging" toggle
        if (AppStateService.hasInstance() && !AppStateService.getInstance().isLoggingEnabled()) {
            if (!message.contains("ERROR") && !message.contains("Exception") && !message.contains("Critical"))
                return;
        }

        // Deduplication Logic
        synchronized (this) {
            if (message.equals(lastLogMessage)) {
                duplicateCount++;
                if (duplicateCount > MAX_DUPLICATES_SHOWN) {
                    // Stop printing after N duplicates to save resources
                    return;
                }
            } else {
                if (duplicateCount > MAX_DUPLICATES_SHOWN) {
                    // Summarize the skipped ones
                    String entry = String.format("[%s] [%s] ... (Skipped %d identical messages)",
                            LocalDateTime.now().format(timeFormatter), source, (duplicateCount - MAX_DUPLICATES_SHOWN));
                    logBuffer.add(entry);
                    for (LogListener l : listeners)
                        l.onNewLog(entry);
                }
                lastLogMessage = message;
                duplicateCount = 0;
            }
        }

        String timestamp = LocalDateTime.now().format(timeFormatter);
        String entry = String.format("[%s] [%s] %s", timestamp, source, message);
        logBuffer.add(entry);

        for (LogListener l : listeners) {
            l.onNewLog(entry);
        }

        // Write to persistent execution.log
        StatusDumperService.getInstance().appendLog(entry);
        // Force print to console for debugging visibility
        // TEMP: Print everything - Removed for production quality
        // System.out.println(entry);
    }

    /**
     * Generates a critical analysis of the logs, highlighting errors and warnings.
     * Useful for post-mortem analysis after a forced stop.
     */
    public String analyzeCriticalErrors() {
        StringBuilder analysis = new StringBuilder();
        analysis.append("--- CRITICAL LOG ANALYSIS ---\n");
        int errorCount = 0;
        int warningCount = 0;

        for (String entry : logBuffer) {
            if (entry.contains("ERROR") || entry.contains("Exception") || entry.contains("Failed")
                    || entry.contains("Exit code")) {
                analysis.append("[CRITICAL] ").append(entry).append("\n");
                errorCount++;
            } else if (entry.contains("Warning") || entry.contains("deprecated")) {
                warningCount++;
            }
        }

        if (errorCount == 0) {
            analysis.append("No explicit errors found in the current log buffer.\n");
        } else {
            analysis.append("Found ").append(errorCount).append(" critical errors.\n");
        }
        if (warningCount > 0) {
            analysis.append("Found ").append(warningCount).append(" warnings.\n");
        }
        analysis.append("-----------------------------\n");
        return analysis.toString();
    }

    public String generateReport() {
        StringBuilder sb = new StringBuilder();
        sb.append("=================================================================\n");
        sb.append("                  AUTOML EXECUTION REPORT                        \n");
        sb.append("Generated: ").append(LocalDateTime.now()).append("\n");
        sb.append("=================================================================\n\n");

        for (String entry : logBuffer) {
            sb.append(entry).append("\n");
        }

        sb.append("\n=================================================================\n");
        sb.append("                       END OF REPORT                             \n");
        sb.append("=================================================================\n");
        return sb.toString();
    }
}
