package com.example.automl_prototype_1.service.agentic;

import com.example.automl_prototype_1.service.LogService;
import com.example.automl_prototype_1.service.OllamaService;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.time.LocalDateTime;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.stream.Collectors;

/**
 * The "System 2" Reflective Brain.
 * 
 * This service observes the entire execution log stream and periodically (or on
 * demand)
 * performs a deep metacognitive analysis to find:
 * 1. Logical inconsistencies (e.g. data leakages, redundant steps).
 * 2. Optimization opportunities.
 * 3. Root causes of complex failures.
 * 
 * It outputs its findings to ANTIGRAVITY_INSIGHTS.md.
 */
public class AntigravityObserverService implements LogService.LogListener {

    private static final AntigravityObserverService instance = new AntigravityObserverService();
    private final CopyOnWriteArrayList<String> sessionLogBuffer = new CopyOnWriteArrayList<>();
    private static final int MAX_BUFFER_SIZE = 500; // Keep last N lines for context
    private static final String INSIGHTS_FILE = "ANTIGRAVITY_INSIGHTS.md";

    private AntigravityObserverService() {
        LogService.getInstance().addListener(this);
        LogService.getInstance().logAI("AntigravityObserver", "Activated", "Now observing system state.");
    }

    public static AntigravityObserverService getInstance() {
        return instance;
    }

    @Override
    public void onNewLog(String message) {
        sessionLogBuffer.add(message);
        if (sessionLogBuffer.size() > MAX_BUFFER_SIZE) {
            sessionLogBuffer.remove(0); // Maintain rolling window
        }
    }

    public void clearBuffer() {
        sessionLogBuffer.clear();
        LogService.getInstance().logAI("AntigravityObserver", "Buffer Cleared", "Starting fresh observation session.");
    }

    /**
     * Triggers the "Deep Thought" process.
     * Sends the current log buffer to the High-Intelligence LLM (DeepSeek/Gemma).
     */
    public void analyzeSession() {
        LogService.getInstance().logAI("AntigravityObserver", "Analyzing Session",
                "Sending " + sessionLogBuffer.size() + " log lines to System 2 (DeepSeek)...");

        String logs = sessionLogBuffer.stream().collect(Collectors.joining("\n"));

        String prompt = "You are the Antigravity Observer, the metacognitive supervisor of an AutoML system.\n" +
                "Review the following execution log for logical inconsistencies, redundant steps, and optimization opportunities.\n"
                +
                "Focus on:\n" +
                "1. Data Leakage (e.g., scaling before splitting).\n" +
                "2. Resource Waste (e.g., loading same model twice).\n" +
                "3. Silent Failures (e.g., fallback to CPU without warning).\n\n" +
                "LOGS:\n" + logs + "\n\n" +
                "Provide a concise Markdown report with 'Observations' and 'Recommendations'.";

        // Async call to avoid blocking the main thread
        // Use queryDeep to bypass SLM and use the "Smart" model
        java.util.concurrent.CompletableFuture.supplyAsync(() -> OllamaService.getInstance().queryDeep(prompt))
                .thenAccept(insight -> {
                    if (insight != null) {
                        writeInsight(insight);
                        LogService.getInstance().logAI("AntigravityObserver", "Insight Generated",
                                "Report written to " + INSIGHTS_FILE);
                    } else {
                        LogService.getInstance().logAI("AntigravityObserver", "Analysis Failed",
                                "LLM returned no response.");
                    }
                });
    }

    private synchronized void writeInsight(String insight) {
        try (FileWriter writer = new FileWriter(new File(INSIGHTS_FILE), true)) { // Append mode
            writer.write("\n\n# Antigravity Insight (" + LocalDateTime.now() + ")\n");
            writer.write(insight);
            writer.write("\n\n---");
        } catch (IOException e) {
            LogService.error("AntigravityObserver", "Failed to write insight: " + e.getMessage());
        }
    }
}
