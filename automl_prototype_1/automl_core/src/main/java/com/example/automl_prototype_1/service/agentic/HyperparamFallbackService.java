package com.example.automl_prototype_1.service.agentic;

import com.example.automl_prototype_1.service.LogService;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * HyperparamFallbackService — Phase 9 Self-Repair Enhancement.
 *
 * Implements a 3-step hyperparameter fallback ladder before an algorithm
 * is banned by SelfCorrectionLoopService. This prevents premature banning
 * due to configuration issues that are recoverable with milder settings.
 *
 * Ladder (applied in order):
 * Step 1 (first failure) — Halve population / reduce iterations by 30%
 * Step 2 (second failure) — Minimal safe config (pop=10, iter=30)
 * Step 3 (third failure) — Full ban via ConstitutionalService
 *
 * Integration: SelfCorrectionLoopService checks getNextConfig() before banning.
 */
public class HyperparamFallbackService {

    private static HyperparamFallbackService instance;

    public static synchronized HyperparamFallbackService getInstance() {
        if (instance == null)
            instance = new HyperparamFallbackService();
        return instance;
    }

    /** Tracks how many times each algorithm has been downstepped. */
    private final Map<String, Integer> downstepCount = new ConcurrentHashMap<>();

    /** Tracks the current fallback config per algorithm. */
    private final Map<String, FallbackConfig> currentConfig = new ConcurrentHashMap<>();

    // -------------------------------------------------------------------------
    // Config DTO
    // -------------------------------------------------------------------------
    public record FallbackConfig(int populationSize, int maxIterations, String notes) {
    }

    // -------------------------------------------------------------------------
    // Default config for each algorithm (starting point)
    // -------------------------------------------------------------------------
    private static final Map<String, FallbackConfig> DEFAULTS = Map.ofEntries(
            Map.entry("GA", new FallbackConfig(50, 100, "GA default")),
            Map.entry("IGPSO", new FallbackConfig(40, 80, "IGPSO default")),
            Map.entry("WWO", new FallbackConfig(40, 80, "WWO default")),
            Map.entry("BPSO", new FallbackConfig(40, 80, "BPSO default")),
            Map.entry("ASO", new FallbackConfig(30, 60, "ASO default")),
            Map.entry("PLS", new FallbackConfig(1, 60, "PLS default (no pop)")),
            Map.entry("SA", new FallbackConfig(1, 200, "SA default")),
            Map.entry("GWO", new FallbackConfig(30, 80, "GWO default")),
            Map.entry("DE", new FallbackConfig(30, 100, "DE default")));

    private HyperparamFallbackService() {
    }

    // -------------------------------------------------------------------------
    // API
    // -------------------------------------------------------------------------

    /**
     * Returns the next fallback config for the algorithm based on failure count.
     * Returns null when ban threshold (3 steps) is reached.
     *
     * @param algorithmName algorithm that failed
     * @return FallbackConfig to try, or null if should be banned
     */
    public FallbackConfig getNextConfig(String algorithmName) {
        int step = downstepCount.compute(algorithmName, (k, v) -> v == null ? 1 : v + 1);

        FallbackConfig base = DEFAULTS.getOrDefault(algorithmName,
                new FallbackConfig(30, 80, "unknown algo defaults"));

        return switch (step) {
            case 1 -> {
                // Step 1: Halve population, reduce iters by 30%
                FallbackConfig cfg = new FallbackConfig(
                        Math.max(5, base.populationSize() / 2),
                        (int) (base.maxIterations() * 0.7),
                        "Fallback L1: reduced params");
                currentConfig.put(algorithmName, cfg);
                LogService.info("HyperparamFallback",
                        algorithmName + " → Fallback L1: pop=" + cfg.populationSize() + " iter=" + cfg.maxIterations());
                yield cfg;
            }
            case 2 -> {
                // Step 2: Minimal safe config
                FallbackConfig cfg = new FallbackConfig(5, 20,
                        "Fallback L2: minimal config");
                currentConfig.put(algorithmName, cfg);
                LogService.info("HyperparamFallback",
                        algorithmName + " → Fallback L2: minimal safe config");
                yield cfg;
            }
            default -> {
                // Step 3+: Signal to ban
                LogService.error("HyperparamFallback",
                        algorithmName + " → BAN threshold reached after " + step + " failures.");
                yield null;
            }
        };
    }

    /**
     * Returns true if the algorithm should be banned (≥ 3 failures).
     */
    public boolean shouldBan(String algorithmName) {
        return downstepCount.getOrDefault(algorithmName, 0) >= 3;
    }

    /**
     * Get the current active fallback config for an algorithm.
     */
    public FallbackConfig getCurrentConfig(String algorithmName) {
        return currentConfig.getOrDefault(algorithmName,
                DEFAULTS.getOrDefault(algorithmName, new FallbackConfig(30, 80, "default")));
    }

    /**
     * Reset an algorithm's failure counter (e.g., after successful run).
     */
    public void resetAlgorithm(String algorithmName) {
        downstepCount.remove(algorithmName);
        currentConfig.remove(algorithmName);
        LogService.info("HyperparamFallback", algorithmName + " reset — failure ladder cleared.");
    }

    /**
     * Reset all algorithms for a fresh session.
     */
    public void reset() {
        downstepCount.clear();
        currentConfig.clear();
    }

    /**
     * Returns a summary of current fallback states.
     */
    public List<String> getSummary() {
        List<String> lines = new ArrayList<>();
        downstepCount.forEach((algo, steps) -> {
            FallbackConfig cfg = currentConfig.get(algo);
            String cfgStr = cfg != null ? "pop=" + cfg.populationSize() + " iter=" + cfg.maxIterations() : "default";
            lines.add(algo + ": " + steps + " failures — active config: " + cfgStr);
        });
        return lines;
    }
}
