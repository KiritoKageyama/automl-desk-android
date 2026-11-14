package com.example.automl_prototype_1.service.agentic;

import com.example.automl_prototype_1.service.LogService;

/**
 * Service for "Introspection" - Self-Monitoring and Analysis.
 * Upgraded to use Statistical Process Control (EMA, CUSUM) for robust trend
 * detection.
 */
public class IntrospectionService {

    private final com.example.automl_prototype_1.service.OllamaService llmService;
    private static final double ALPHA_EMA = 0.2; // Smoothing factor for EMA
    private static final double CUSUM_THRESHOLD = 5.0; // Standard deviations for drift detection

    public IntrospectionService() {
        this.llmService = com.example.automl_prototype_1.service.OllamaService.getInstance();
    }

    /**
     * Structure to hold the result of a performance analysis.
     */
    public static class AnalysisResult {
        public enum Trend {
            IMPROVING, STABLE, DEGRADING
        }

        public final Trend trend;
        public final boolean driftDetected;
        public final double currentScore;
        public final String advice;

        public AnalysisResult(Trend trend, boolean driftDetected, double currentScore, String advice) {
            this.trend = trend;
            this.driftDetected = driftDetected;
            this.currentScore = currentScore;
            this.advice = advice;
        }

        @Override
        public String toString() {
            return String.format("Trend: %s | Drift: %s | Score: %.2f | Advice: %s", trend, driftDetected, currentScore,
                    advice);
        }
    }

    /**
     * Analyzes a specific failure event with LLM context.
     */
    public String analyzeFailure(String context, String actionTaken, String outcome) {
        if (!llmService.isAvailable()) {
            return "Self-Analysis: Failure detected. Recommendation: Increase exploration variance.";
        }

        String prompt = "JOB: Analyze this gameplay failure and provide a correction.\n" +
                "CONTEXT: " + context + "\n" +
                "ACTION: " + actionTaken + "\n" +
                "OUTCOME: " + outcome + "\n" +
                "TASK: Explain WHY this failed and WHAT should have been done instead. Keep it short (1 sentence).";

        LogService.info("Introspection", "Analyzing failure with LLM...");
        return llmService.queryOllama(prompt, "active");
    }

    /**
     * Analyzes a sequence of rewards using Statistical Process Control.
     * Uses EMA for trend and CUSUM for sudden shifts.
     * 
     * @param rewardHistory Recent history of rewards.
     * @return Structured analysis result.
     */
    public AnalysisResult analyzePerformanceTrend(double[] rewardHistory) {
        if (rewardHistory == null || rewardHistory.length < 5) {
            return new AnalysisResult(AnalysisResult.Trend.STABLE, false, 0.0, "Insufficient data.");
        }

        // 1. Calculate Exponential Moving Average (EMA) to smooth noise
        double ema = calculateEMA(rewardHistory);

        // 2. Identify Long-term Trend
        // Compare robust moving average of recent vs older window
        double recentAvg = calculateMean(rewardHistory, rewardHistory.length - 5, rewardHistory.length);
        double baselineAvg = calculateMean(rewardHistory, 0, Math.min(10, rewardHistory.length));

        AnalysisResult.Trend trend = AnalysisResult.Trend.STABLE;
        if (recentAvg > baselineAvg * 1.05)
            trend = AnalysisResult.Trend.IMPROVING;
        else if (recentAvg < baselineAvg * 0.95)
            trend = AnalysisResult.Trend.DEGRADING;

        // 3. Detect Drift using CUSUM (Cumulative Sum Control Chart)
        boolean drift = detectDriftCUSUM(rewardHistory);

        // 4. Formulate Advice
        String advice = "Continue current strategy.";

        if (drift) {
            if (trend == AnalysisResult.Trend.IMPROVING) {
                advice = "BREAKTHROUGH: Significant positive drift detected. Performance is exceeding historical baseline.";
            } else {
                advice = "CRITICAL: Significant negative drift detected. Reset environment or aggressive exploration needed.";
            }
        } else if (trend == AnalysisResult.Trend.DEGRADING) {
            advice = "Performance is slowly degrading. Check for overfitting or local optima.";
        } else if (trend == AnalysisResult.Trend.IMPROVING) {
            advice = "Performance is improving. Consider reducing exploration rate (exploit).";
        }

        return new AnalysisResult(trend, drift, ema, advice);
    }

    private double calculateEMA(double[] data) {
        double ema = data[0];
        for (int i = 1; i < data.length; i++) {
            ema = ALPHA_EMA * data[i] + (1 - ALPHA_EMA) * ema;
        }
        return ema;
    }

    private double calculateMean(double[] data, int start, int end) {
        double sum = 0;
        int count = 0;
        for (int i = start; i < end && i < data.length; i++) {
            sum += data[i];
            count++;
        }
        return count == 0 ? 0 : sum / count;
    }

    private boolean detectDriftCUSUM(double[] data) {
        // Simplified CUSUM for mean shift
        double mean = calculateMean(data, 0, data.length);
        double stdDev = calculateStdDev(data, mean);

        double posSum = 0;
        double negSum = 0;
        double k = 0.5 * stdDev; // Slack
        double h = CUSUM_THRESHOLD * stdDev; // Threshold

        for (double x : data) {
            posSum = Math.max(0, posSum + (x - mean - k));
            negSum = Math.max(0, negSum + (mean - x - k));

            if (posSum > h || negSum > h)
                return true;
        }
        return false;
    }

    private double calculateStdDev(double[] data, double mean) {
        double sumSq = 0;
        for (double x : data)
            sumSq += Math.pow(x - mean, 2);
        return Math.sqrt(sumSq / data.length);
    }
}
