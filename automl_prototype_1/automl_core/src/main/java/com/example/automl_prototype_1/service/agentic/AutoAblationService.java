package com.example.automl_prototype_1.service.agentic;

import com.example.automl_prototype_1.service.LogService;
// import java.util.Map;

/**
 * Automatically "ablates" (disables) features or algorithms that show high
 * failure rates.
 * Acts as an immune system, suppressing unstable components.
 */
public class AutoAblationService {

    private final PerformanceHistoryService historyService;
    private final ConstitutionalService constitutionalService;

    // Thresholds
    private static final int MIN_ATTEMPTS = 3;
    private static final double MAX_FAILURE_RATE = 0.50; // 50% failure -> BAN

    public AutoAblationService(PerformanceHistoryService historyService, ConstitutionalService constitutionalService) {
        this.historyService = historyService;
        this.constitutionalService = constitutionalService;
    }

    /**
     * Checks a specific algorithm. If it's unstable, bans it.
     */
    public void checkAndAblate(String algorithmName) {
        double failureRate = historyService.getFailureRate(algorithmName);
        int attempts = historyService.getAttempts(algorithmName);

        if (attempts >= MIN_ATTEMPTS && failureRate > MAX_FAILURE_RATE) {
            String ruleId = "AUTO_BAN_" + algorithmName.toUpperCase().replaceAll("\\s+", "_");
            String reason = String.format("Auto-Ablation: %s has a %.0f%% failure rate over %d attempts.",
                    algorithmName, failureRate * 100, attempts);

            // Dynamic Rule Injection
            // Condition: algorithm == 'algorithmName'
            String condition = "algorithm == '" + algorithmName + "'";
            constitutionalService.addRule(ruleId, condition, "VETO", reason);

            LogService.warn("AutoAblation", "🚨 ABLATION TRIGGERED: " + reason);
        }
    }
}
