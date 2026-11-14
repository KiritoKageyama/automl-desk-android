package com.example.automl_prototype_1.service.agentic;

import com.example.automl_prototype_1.service.LogService;

/**
 * Service to query general knowledge about games (controls, objectives) using
 * the Local LLM.
 * This effectively gives the agent "Wikipedia" knowledge.
 */
public class GameKnowledgeService {

    private final com.example.automl_prototype_1.service.OllamaService llmService;

    public GameKnowledgeService() {
        this.llmService = com.example.automl_prototype_1.service.OllamaService.getInstance();
    }

    /**
     * Asks the LLM for the control scheme of a specific game.
     * 
     * @param gameName The name of the game (e.g., "Super Mario Bros", "Need for
     *                 Speed").
     * @return A string description or JSON-like mapping of controls.
     */
    public String getGameControls(String gameName) {
        if (!llmService.isAvailable()) {
            LogService.warn("GameKnowledge", "LLM unavailble. Using default controls.");
            return "Default: W, A, S, D, Space";
        }

        String prompt = "JOB: List the PC keyboard controls for the game '" + gameName + "'.\n" +
                "FORMAT: JSON object { \"Action\": \"Key\" }.\n" +
                "EXAMPLE: { \"Jump\": \"Space\", \"Move Forward\": \"W\" }.\n" +
                "Keep it simple and focusing on default bindings.";

        LogService.info("GameKnowledge", "Querying knowledge base for: " + gameName);
        String response = llmService.queryOllama(prompt, "active");

        return response;
    }

    /**
     * Asks the LLM for the objective/goal of the game.
     */
    public String getGameObjective(String gameName) {
        if (!llmService.isAvailable())
            return "Unknown Objective";

        String prompt = "JOB: Briefly explain the main objective of the game '" + gameName + "' in 1 sentence.";
        return llmService.queryOllama(prompt, "active");
    }
}
