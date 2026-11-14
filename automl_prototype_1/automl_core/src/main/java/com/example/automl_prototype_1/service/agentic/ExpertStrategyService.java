package com.example.automl_prototype_1.service.agentic;

import com.example.automl_prototype_1.service.LogService;

/**
 * Service to fetch "Expert" strategies and techniques for specific games.
 * Acts as a bridge between the Agent and external knowledge (LLM/Web).
 */
public class ExpertStrategyService {

    private final com.example.automl_prototype_1.service.OllamaService llmService;

    public ExpertStrategyService() {
        this.llmService = com.example.automl_prototype_1.service.OllamaService.getInstance();
    }

    /**
     * Queries for advanced techniques for a specific game and scenario.
     * 
     * @param gameName  The game (e.g., "NFS Most Wanted").
     * @param technique The specific technique to learn (e.g., "Drifting",
     *                  "Cornering").
     * @return A step-by-step guide or heuristic provided by the LLM.
     */
    public String getProTechnique(String gameName, String technique) {
        if (!llmService.isAvailable()) {
            String fallback = "Expert Service Unavailable. Analysis: Drive fast, don't crash.";
            KnowledgeBaseService.getInstance().addInsight("Strategies", fallback);
            return fallback;
        }

        String prompt = "JOB: Acting as a professional gamer/speedrunner, explain detailed techniques for '"
                + technique + "' in the game '" + gameName + "'.\n" +
                "CONTEXT: The user wants to optimize their play agent.\n" +
                "OUTPUT: A concise list of 3-5 bullet points focusing on inputs (brake, gas, steer).";

        LogService.info("ExpertStrategy", "Consulting Pro Gamers about '" + technique + "' in " + gameName + "...");
        String response = llmService.queryOllama(prompt, "active");

        if (response != null) {
            KnowledgeBaseService.getInstance().addInsight("Strategies", response);
        }
        return response;
    }

    /**
     * Gets a general strategy guide for the game.
     */
    public String getGeneralStrategy(String gameName) {
        String prompt = "JOB: Provide a high-level strategy guide for dominating '" + gameName + "'.";
        String response = llmService.queryOllama(prompt, "active");
        if (response != null) {
            KnowledgeBaseService.getInstance().addInsight("Strategies", response);
        }
        return response;
    }
}
