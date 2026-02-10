package com.example.automl_prototype_1.service;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.StandardOpenOption;
import java.time.LocalDateTime;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;

/**
 * Service to dump the application state to a JSON file periodically.
 * Allows external observers (User/AI) to check status asynchronously.
 */
public class StatusDumperService {

    private static final StatusDumperService instance = new StatusDumperService();
    private final ScheduledExecutorService scheduler = Executors.newSingleThreadScheduledExecutor();
    private final ObjectMapper mapper = new ObjectMapper().enable(SerializationFeature.INDENT_OUTPUT);
    private final File statusFile = new File("status.json");
    private final File fullLogFile = new File("execution.log");

    public static StatusDumperService getInstance() {
        return instance;
    }

    private StatusDumperService() {
        // Auto-Cleanup on Startup
        clearLegacyLogs();

        // Start dumping status every 2 seconds
        scheduler.scheduleAtFixedRate(this::dumpStatus, 1, 2, TimeUnit.SECONDS);
    }

    private void clearLegacyLogs() {
        try {
            if (fullLogFile.exists()) {
                // Check if file is "old" (e.g., > 5MB or just always clear on fresh start?)
                // User requested "clear them every time the pp starts"
                boolean deleted = fullLogFile.delete();
                if (deleted) {
                    // Re-create empty
                    fullLogFile.createNewFile();
                    LogService.info("System", "Legacy execution.log cleared.");
                }
            }
        } catch (Exception e) {
            System.err.println("Failed to clear legacy logs: " + e.getMessage());
        }
    }

    private void dumpStatus() {
        try {
            var state = ProgressTrackingService.getInstance().snapshot();
            if (state != null) {
                // Decorate with timestamp
                var statusNode = mapper.createObjectNode();
                statusNode.put("timestamp", LocalDateTime.now().toString());
                statusNode.putPOJO("state", state);
                statusNode.put("system_memory_mb",
                        (Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory()) / 1024 / 1024);

                // Atomic write (write to temp then rename to avoid read conflicts)
                File tempFile = new File("status.json.tmp");
                mapper.writeValue(tempFile, statusNode);

                // Simple rename (atomic on POSIX, usually fine on Windows for this purpose)
                if (statusFile.exists()) {
                    statusFile.delete();
                }
                tempFile.renameTo(statusFile);
            }
        } catch (Exception e) {
            // calculated silence
        }
    }

    public void appendLog(String message) {
        try {
            String logLine = LocalDateTime.now() + " " + message + System.lineSeparator();
            Files.writeString(fullLogFile.toPath(), logLine, StandardOpenOption.CREATE, StandardOpenOption.APPEND);
        } catch (IOException e) {
            // Ignore log errors
        }
    }
}
