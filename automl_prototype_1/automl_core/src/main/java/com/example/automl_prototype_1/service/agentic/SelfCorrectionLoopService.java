package com.example.automl_prototype_1.service.agentic;

import com.example.automl_prototype_1.service.LogService;

/**
 * System Self-Correction Loop (Phase 6, Step 6).
 *
 * Closes the feedback loop between execution failures and system adaptation:
 * 1. After each algorithm run, logs results to PerformanceHistoryService
 * 2. Runs AutoAblationService to check if any algorithm should be dynamically
 * banned
 * 3. If banned, ConstitutionalService rules are updated automatically
 *
 * This creates an autonomous "immune system" that:
 * - Tracks which algorithms fail frequently
 * - Automatically disables unstable algorithms (>50% failure rate after 3+
 * attempts)
 * - Injects dynamic ban rules into the Constitutional AI layer
 */
public class SelfCorrectionLoopService {

    private final PerformanceHistoryService historyService;
    private final AutoAblationService ablationService;
    private final ConstitutionalService constitutionalService;

    private static SelfCorrectionLoopService instance;

    public SelfCorrectionLoopService() {
        this.historyService = new PerformanceHistoryService();
        this.constitutionalService = new ConstitutionalService();
        this.ablationService = new AutoAblationService(historyService, constitutionalService);
        LogService.info("SelfCorrection", "Self-Correction Loop initialized.");
    }

    public static synchronized SelfCorrectionLoopService getInstance() {
        if (instance == null) {
            instance = new SelfCorrectionLoopService();
        }
        return instance;
    }

    /**
     * Called after every algorithm execution to close the feedback loop.
     *
     * @param algorithmName Name of the algorithm that ran
     * @param success       Whether the algorithm executed successfully
     * @param errorType     Type of error if failed (null if success)
     * @param accuracy      The accuracy achieved (0.0 if failed)
     */
    public void onExecutionComplete(String algorithmName, boolean success, String errorType, double accuracy) {
        // Step 1: Log result to performance history
        historyService.logResult(algorithmName, success, errorType);

        // Step 2: Run auto-ablation check (will ban if failure rate > 50% after 3+
        // attempts)
        ablationService.checkAndAblate(algorithmName);

        // Step 3: Log the correction loop activity
        double failureRate = historyService.getFailureRate(algorithmName);
        int attempts = historyService.getAttemptCount(algorithmName);

        if (success) {
            LogService.info("SelfCorrection",
                    String.format("✅ %s succeeded (Accuracy: %.2f%%). [%d attempts, %.0f%% failure rate]",
                            algorithmName, accuracy * 100, attempts, failureRate * 100));
        } else {
            LogService.warn("SelfCorrection",
                    String.format("❌ %s failed (%s). [%d attempts, %.0f%% failure rate]",
                            algorithmName, errorType != null ? errorType : "Unknown", attempts, failureRate * 100));

            if (failureRate > 0.5 && attempts >= 3) {
                LogService.error("SelfCorrection",
                        String.format("🚨 ABLATION ALERT: %s has been flagged for automatic ban!", algorithmName));
            }
        }
    }

    /**
     * Checks if an algorithm is currently banned by the constitutional rules.
     *
     * @param algorithmName The algorithm to check
     * @return true if the algorithm is banned
     */
    public boolean isAlgorithmBanned(String algorithmName) {
        String ruleId = "AUTO_BAN_" + algorithmName.toUpperCase().replaceAll("\\s+", "_");
        return constitutionalService.hasRule(ruleId);
    }

    /**
     * Resets the ban on an algorithm (useful for manual override by user).
     *
     * @param algorithmName The algorithm to unban
     */
    public void resetBan(String algorithmName) {
        String ruleId = "AUTO_BAN_" + algorithmName.toUpperCase().replaceAll("\\s+", "_");
        constitutionalService.removeRule(ruleId);
        LogService.info("SelfCorrection", "🔓 Manual reset: " + algorithmName + " has been unbanned.");
    }

    /** Returns the underlying performance history service. */
    public PerformanceHistoryService getHistoryService() {
        return historyService;
    }
}
