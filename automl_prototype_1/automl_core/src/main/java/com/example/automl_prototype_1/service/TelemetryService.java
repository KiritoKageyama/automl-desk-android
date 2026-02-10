package com.example.automl_prototype_1.service;

import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicReference;

/**
 * Centralized service for Active Data Monitoring and Telemetry.
 * Tracks throughput, error rates, and latest data shapes.
 */
public class TelemetryService {

    private static final TelemetryService instance = new TelemetryService();

    // Metrics
    private final AtomicLong totalRowsProcessed = new AtomicLong(0);
    private final AtomicLong totalBatchesProcessed = new AtomicLong(0);
    private final AtomicLong totalErrors = new AtomicLong(0);

    // Latest Activity Snapshot
    private final AtomicReference<String> lastActivity = new AtomicReference<>("Idle");
    private final AtomicReference<String> lastDataShape = new AtomicReference<>("-");

    private TelemetryService() {
    }

    public static TelemetryService getInstance() {
        return instance;
    }

    public void recordTransaction(String source, int rows, int cols) {
        totalRowsProcessed.addAndGet(rows);
        totalBatchesProcessed.incrementAndGet();
        lastActivity.set("Processed Batch from " + source);
        lastDataShape.set(rows + "x" + cols);
        LogService.info("Telemetry", String.format("Data Monitor: [%s] processed %dx%d. Total Rows: %d", source, rows,
                cols, totalRowsProcessed.get()));
    }

    public void recordError(String source) {
        totalErrors.incrementAndGet();
        lastActivity.set("Error in " + source);
        // We don't log here to avoid infinite loops if LogService calls this
    }

    public String getStatusReport() {
        return String.format(
                "Telemetry Report:\n" +
                        " - Total Rows: %d\n" +
                        " - Total Batches: %d\n" +
                        " - Total Errors: %d\n" +
                        " - Last Activity: %s\n" +
                        " - Last Shape: %s",
                totalRowsProcessed.get(),
                totalBatchesProcessed.get(),
                totalErrors.get(),
                lastActivity.get(),
                lastDataShape.get());
    }

    public void reset() {
        totalRowsProcessed.set(0);
        totalBatchesProcessed.set(0);
        totalErrors.set(0);
        lastActivity.set("Idle (Reset)");
        lastDataShape.set("-");
    }

    /**
     * Attempts to clear application memory.
     * 1. Resets internal counters.
     * 2. Clears LogService buffer.
     * 3. Triggers System.gc() (Suggestion to JVM).
     */
    public void triggerMemoryCleanup() {
        LogService.info("Telemetry", "--- Initiating 'Hacker Mode' Memory Cleanup ---");
        // 1. Clear Logs from Memory
        LogService.getInstance().clear();

        // 2. Clear Telemetry
        reset();

        // 3. Clear Caches
        try {
            // Suggest GC
            System.gc();
            Runtime.getRuntime().gc();
            Thread.sleep(100); // Give it a moment
        } catch (Exception e) {
        }

        long freeMem = Runtime.getRuntime().freeMemory() / 1024 / 1024;
        long totalMem = Runtime.getRuntime().totalMemory() / 1024 / 1024;
        LogService.info("Telemetry", String.format("Memory Cleanup Complete. Free: %dMB / %dMB", freeMem, totalMem));
    }
}
