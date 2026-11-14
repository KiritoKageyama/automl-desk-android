package com.example.automl_prototype_1.service.agentic;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.example.automl_prototype_1.service.LogService;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.Map;

/**
 * Tracks the success/failure history of algorithms to enable empirical decision
 * making.
 */
public class PerformanceHistoryService {

    private static final String HISTORY_FILE = "performance_history.json";
    private final Path configPath;
    private final ObjectMapper mapper;
    private HistoryData data;

    public PerformanceHistoryService() {
        this.mapper = new ObjectMapper();
        this.configPath = Paths.get(System.getProperty("user.dir"), "automl_core", "src", "main", "resources",
                HISTORY_FILE);
        loadHistory();
    }

    public synchronized void logResult(String algorithmName, boolean success, String errorType) {
        if (data == null)
            data = new HistoryData();

        AlgoStats stats = data.algorithms.computeIfAbsent(algorithmName, k -> new AlgoStats());
        stats.attempts++;
        if (success) {
            stats.successes++;
        } else {
            stats.failures.merge(errorType != null ? errorType : "UNKNOWN", 1, (a, b) -> a + b);
        }

        saveHistory();
        LogService.info("PerformanceHistory", String.format("Logged result for %s: %s (Failure Rate: %.2f%%)",
                algorithmName, success ? "SUCCESS" : "FAIL", getFailureRate(algorithmName) * 100));
    }

    public double getFailureRate(String algorithmName) {
        if (data == null || !data.algorithms.containsKey(algorithmName))
            return 0.0;
        AlgoStats stats = data.algorithms.get(algorithmName);
        if (stats.attempts == 0)
            return 0.0;
        return 1.0 - ((double) stats.successes / stats.attempts);
    }

    public int getAttempts(String algorithmName) {
        if (data == null || !data.algorithms.containsKey(algorithmName))
            return 0;
        return data.algorithms.get(algorithmName).attempts;
    }

    /** Alias for getAttempts — used by UI controller. */
    public int getAttemptCount(String algorithmName) {
        return getAttempts(algorithmName);
    }

    /** Returns the full history map for UI display. */
    public Map<String, AlgoStats> getAllHistory() {
        if (data == null)
            return new HashMap<>();
        return data.algorithms;
    }

    /** Clears all performance history. */
    public synchronized void clearHistory() {
        data = new HistoryData();
        saveHistory();
        LogService.info("PerformanceHistory", "All history cleared.");
    }

    /**
     * Returns the algorithm with the highest failure rate (after ≥ 3 attempts),
     * or null if no qualifying algorithm exists.
     */
    public String getWorstAlgorithm() {
        if (data == null || data.algorithms.isEmpty())
            return null;
        return data.algorithms.entrySet().stream()
                .filter(e -> e.getValue().attempts >= 3)
                .max(java.util.Comparator.comparingDouble(
                        e -> getFailureRate(e.getKey())))
                .map(Map.Entry::getKey)
                .orElse(null);
    }

    private void loadHistory() {
        try {
            if (Files.exists(configPath)) {
                data = mapper.readValue(configPath.toFile(), HistoryData.class);
            } else {
                // Try loading from classpath resources if file doesn't exist (packaged app)
                // For now, simpler to verify file existence or start new.
                data = new HistoryData();
            }
        } catch (Exception e) {
            LogService.error("PerformanceHistory", "Failed to load history: " + e.getMessage());
            data = new HistoryData();
        }
    }

    private void saveHistory() {
        try {
            Files.createDirectories(configPath.getParent());
            mapper.writerWithDefaultPrettyPrinter().writeValue(configPath.toFile(), data);
        } catch (IOException e) {
            LogService.error("PerformanceHistory", "Failed to save history: " + e.getMessage());
        }
    }

    // Data Structures for JSON mapping
    public static class HistoryData {
        public Map<String, AlgoStats> algorithms = new HashMap<>();
    }

    public static class AlgoStats {
        public int attempts = 0;
        public int successes = 0;
        public Map<String, Integer> failures = new HashMap<>(); // ErrorType -> Count
    }
}
