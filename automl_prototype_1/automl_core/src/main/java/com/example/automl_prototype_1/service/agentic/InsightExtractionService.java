package com.example.automl_prototype_1.service.agentic;

import com.example.automl_prototype_1.service.LogService;
import com.example.automl_prototype_1.rl.agent.TabularQLearningAgent;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * Service to translate raw RL Q-Tables into human-readable insights using
 * GenAI.
 */
public class InsightExtractionService {

    private final com.example.automl_prototype_1.service.OllamaService llmService;

    public InsightExtractionService() {
        this.llmService = com.example.automl_prototype_1.service.OllamaService.getInstance();
    }

    /**
     * Analyzes the agent's Q-Table and asks the LLM to explain the strategy.
     */
    public String extractInsights(TabularQLearningAgent agent) {
        Map<String, double[]> qTable = agent.getQTable();
        if (qTable.isEmpty()) {
            return "The agent has not learned anything yet (Q-Table is empty).";
        }

        // 1. Identify Critical States (High Variance)
        StringBuilder criticalStates = new StringBuilder();
        int count = 0;

        List<Map.Entry<String, double[]>> sortedEntries = new ArrayList<>(qTable.entrySet());
        // Sort by "Importance" (max Q - min Q)?
        // For prototype, just take top 10 states with highest Max Q (strongest
        // opinions)
        sortedEntries.sort((e1, e2) -> Double.compare(max(e2.getValue()), max(e1.getValue())));

        for (Map.Entry<String, double[]> entry : sortedEntries) {
            if (count >= 10)
                break; // Limit to 10 states for prompt context window

            String stateKey = entry.getKey();
            double[] values = entry.getValue();

            // Filter out unvisited states (all zeros)
            if (max(values) == 0 && min(values) == 0)
                continue;

            criticalStates.append(String.format("- State [%s]: Actions %s\n", stateKey, arrayToString(values)));
            count++;
        }

        if (count == 0) {
            return "The agent is still exploring (mostly zero Q-values). Train longer!";
        }

        // 2. Query LLM
        String prompt = String.format(
                "You are an RL Analyst. I will give you the top learned states from a Q-Learning agent.\n" +
                        "The format is 'State [features]: Actions [q_val_0, q_val_1, ...]'.\n\n" +
                        "DATA:\n%s\n\n" +
                        "Analyze these values. Which actions are preferred? What pattern do you see?\n" +
                        "Provide 3 concise bullet points explaining the agent's strategy.",
                criticalStates.toString());

        LogService.info("InsightExtraction", "Asking LLM to explain " + count + " states...");
        String explanation = llmService.queryOllama(prompt, "insight_extraction");

        return explanation != null ? explanation : "Failed to generate insights.";
    }

    private double max(double[] arr) {
        double m = -Double.MAX_VALUE;
        for (double v : arr)
            if (v > m)
                m = v;
        return m;
    }

    private double min(double[] arr) {
        double m = Double.MAX_VALUE;
        for (double v : arr)
            if (v < m)
                m = v;
        return m;
    }

    private String arrayToString(double[] arr) {
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < arr.length; i++) {
            sb.append(String.format("%.2f", arr[i]));
            if (i < arr.length - 1)
                sb.append(", ");
        }
        sb.append("]");
        return sb.toString();
    }
}
