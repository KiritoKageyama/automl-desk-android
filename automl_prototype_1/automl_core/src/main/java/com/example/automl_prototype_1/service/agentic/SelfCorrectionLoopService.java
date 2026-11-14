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
        this.constitutionalService = ConstitutionalService.getInstance();
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
     * Now integrates ErrorClassifier + HyperparamFallbackService (Phase 9).
     */
    public void onExecutionComplete(String algorithmName, boolean success, String errorType, double accuracy) {
        historyService.logResult(algorithmName, success, errorType);
        ablationService.checkAndAblate(algorithmName);

        double failureRate = historyService.getFailureRate(algorithmName);
        int attempts = historyService.getAttemptCount(algorithmName);

        if (success) {
            // Reset fallback ladder on any successful run
            HyperparamFallbackService.getInstance().resetAlgorithm(algorithmName);
            LogService.info("SelfCorrection",
                    String.format("✅ %s succeeded (Accuracy: %.2f%%). [%d attempts, %.0f%% failure rate]",
                            algorithmName, accuracy * 100, attempts, failureRate * 100));
        } else {
            // Classify the error for targeted remediation
            ErrorClassifier.ErrorType errClass = ErrorClassifier.getInstance().classify(errorType);
            String remedy = ErrorClassifier.getInstance().remediation(errClass);
            String errLabel = ErrorClassifier.getInstance().describe(errClass);

            LogService.warn("SelfCorrection",
                    String.format("❌ %s failed [%s]. Remedy: %s [%d attempts, %.0f%% failure rate]",
                            algorithmName, errLabel, remedy, attempts, failureRate * 100));

            // Only consider banning after failure rate > 50% after 3+ attempts
            if (failureRate > 0.5 && attempts >= 3) {
                // Try hyperparameter fallback first before banning
                var fallback = HyperparamFallbackService.getInstance().getNextConfig(algorithmName);
                if (fallback == null) {
                    // Fallback exhausted → ban
                    ablationService.checkAndAblate(algorithmName); // triggers actual ban
                    LogService.error("SelfCorrection",
                            String.format("🚨 BAN: %s is permanently disabled after %d failures.", algorithmName,
                                    attempts));
                } else {
                    LogService.warn("SelfCorrection",
                            String.format("⚠ %s → Fallback applied: pop=%d iter=%d. Note: %s",
                                    algorithmName, fallback.populationSize(), fallback.maxIterations(),
                                    fallback.notes()));
                }
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

    /**
     * Phase 9.1 — Triggered by ScreenWatcherService or ErrorMonitorService when a
     * critical error or stuck training is detected. Applies targeted remediation
     * based on the classified error type.
     *
     * @param rawErrorType Raw error string (e.g. "CUDA_OOM",
     *                     "STUCK_TRAINING_TIMEOUT")
     */
    public void triggerCorrection(String rawErrorType) {
        ErrorClassifier.ErrorType errClass = ErrorClassifier.getInstance().classify(rawErrorType);
        String describe = ErrorClassifier.getInstance().describe(errClass);
        LogService.warn("SelfCorrection", "⚡ triggerCorrection() called: " + rawErrorType + " → " + describe);

        switch (errClass) {
            case CUDA_OOM -> {
                com.example.automl_prototype_1.service.AppStateService.getInstance().setForceGpuOff();
                LogService.warn("SelfCorrection",
                        "🔴 CUDA OOM → Switched to CPU-only mode. Re-run experiment.");
            }
            case MISSING_DEP ->
                LogService.error("SelfCorrection",
                        "📦 Missing dependency. Run: pip install -r requirements.txt");
            case DATA_ERROR ->
                LogService.warn("SelfCorrection",
                        "📂 Data error. Check for NaN/Inf in dataset; verify feature/label alignment.");
            case TIMEOUT ->
                LogService.warn("SelfCorrection",
                        "⏱ Training timed out. Reduce algorithm iterations or increase timeout budget.");
            case COMPILE_ERR ->
                LogService.error("SelfCorrection",
                        "🛠 Compile error in generated code. Retry with a simpler template.");
            case CONVERGENCE ->
                LogService.warn("SelfCorrection",
                        "📉 Algorithm diverged. Reduce learning rate or try a different algorithm.");
            default -> {
                // Generic: look up worst algorithm and ablate it
                String worst = historyService.getWorstAlgorithm();
                if (worst != null) {
                    ablationService.checkAndAblate(worst);
                    LogService.warn("SelfCorrection",
                            "🔧 Unknown error → Ablating worst-performing algorithm: " + worst);
                } else {
                    LogService.warn("SelfCorrection",
                            "⚠ Cannot identify corrective action for error: " + rawErrorType);
                }
            }
        }
    }
}
