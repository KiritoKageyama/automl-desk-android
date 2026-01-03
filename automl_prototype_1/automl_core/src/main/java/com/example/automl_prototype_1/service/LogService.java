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

    private LogListener listener;

    public void setListener(LogListener listener) {
        this.listener = listener;
    }

    /**
     * Helper for error messages.
     */
    public static void error(String source, String message) {
        getInstance().log(source, "ERROR: " + message);
    }

    /**
     * Logs a message with a specific source/context.
     * 
     * @param source  The component generating the log (e.g., "GA",
     *                "ExecutionService").
     * @param message The details of the operation.
     */
    public void log(String source, String message) {
        // Respect the "Detailed Logging" toggle
        // Only check if AppStateService is ready to avoid circular dependency
        if (AppStateService.hasInstance() && !AppStateService.getInstance().isLoggingEnabled()) {
            // Optional: Allow critical errors through
            if (!message.contains("ERROR"))
                return;
        }

        String timestamp = LocalDateTime.now().format(timeFormatter);
        String entry = String.format("[%s] [%s] %s", timestamp, source, message);
        logBuffer.add(entry);

        // Notify UI Listener (Fire and Forget)
        if (listener != null) {
            listener.onNewLog(entry);
        }
    }

    /**
     * Generates the full execution report as a single string.
     * 
     * @return The formatted report.
     */
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
