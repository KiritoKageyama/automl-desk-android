package com.example.automl_prototype_1.service.agentic;

import com.example.automl_prototype_1.service.LogService;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.util.HashMap;
import java.util.Map;

/**
 * Service to inject "Common Sense" or "Heuristic" knowledge from an LLM
 * into an RL Agent's Q-Table (Priors).
 *
 * Flow:
 * 1. User asks for "Tic-Tac-Toe Strategy".
 * 2. LLM returns JSON map of State-Pattern -> Value.
 * 3. StrategyInjector parses this and returns a Map of priors.
 * 4. RL Agent initializes Q(s,a) with these values.
 */
public class StrategyInjectorService {

    private final LocalLlmService llmService;
    private final ObjectMapper mapper;

    public StrategyInjectorService() {
        this.llmService = new LocalLlmService();
        this.mapper = new ObjectMapper();
    }

    /**
     * Queries the LLM for a strategy guide for the given environment/game.
     * Returns a map of "State Description" -> "Action Value Boost".
     *
     * Note: Since mapping specific hash keys is hard for LLM without seeing the
     * internal hash function,
     * we ask for *concepts* (e.g., "Corner", "Center") and the Agent/Env must map
     * these concepts to concrete states.
     *
     * For the prototype, we will try a simplified approach:
     * - We assume the Environment can provide a "State Description" string.
     * - The LLM provides values for those descriptions.
     *
     * @param gameName e.g., "Tic-Tac-Toe"
     * @return Map of Concept -> Value (0.0 to 1.0)
     */
    public Map<String, Double> fetchStrategyGuide(String gameName) {
        LogService.info("StrategyInjector", "Fetching strategy for: " + gameName);

        String prompt = String.format(
                "You are a Game Theory Expert. Provide a JSON strategy guide for '%s'.\n" +
                        "Output ONLY a JSON object where keys are strategic concepts and values are importance (0.0 to 1.0).\n"
                        +
                        "Example for Tic-Tac-Toe:\n" +
                        "{\n" +
                        "  \"center\": 0.9,\n" +
                        "  \"corner\": 0.7,\n" +
                        "  \"edge\": 0.4,\n" +
                        "  \"block_opponent\": 1.0,\n" +
                        "  \"win_now\": 1.0\n" +
                        "}\n" +
                        "Do not include markdown blocks or explanation. Just the raw JSON.",
                gameName);

        String jsonResponse = llmService.queryOllama(prompt, "strategy_guide");
        if (jsonResponse == null || jsonResponse.isEmpty()) {
            return new HashMap<>();
        }

        return parseJsonToMap(jsonResponse);
    }

    private Map<String, Double> parseJsonToMap(String json) {
        Map<String, Double> result = new HashMap<>();
        try {
            // Clean markdown if present
            if (json.contains("```json")) {
                json = json.split("```json")[1].split("```")[0].trim();
            } else if (json.contains("```")) {
                json = json.split("```")[1].trim();
            }

            JsonNode root = mapper.readTree(json);
            root.fields().forEachRemaining(entry -> {
                result.put(entry.getKey().toLowerCase(), entry.getValue().asDouble());
            });

            LogService.info("StrategyInjector", "Parsed " + result.size() + " strategic concepts.");
        } catch (Exception e) {
            LogService.error("StrategyInjector", "Failed to parse strategy JSON: " + e.getMessage());
        }
        return result;
    }
}
