package com.example.automl_prototype_1.service.agentic;

import com.example.automl_prototype_1.service.LogService;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;

/**
 * Service to store "Knowledge" acquired by the agent.
 * Acts as a database for Rules, Strategies, and Insights.
 * Used for Visualization and Editing in the UI.
 */
public class KnowledgeBaseService {

    private static KnowledgeBaseService instance;
    private final List<Insight> insights;

    private KnowledgeBaseService() {
        this.insights = new CopyOnWriteArrayList<>();
        // Seed with some default knowledge
        addInsight("General", "If you crash, try not to do that again.");
    }

    public static synchronized KnowledgeBaseService getInstance() {
        if (instance == null) {
            instance = new KnowledgeBaseService();
        }
        return instance;
    }

    public void addInsight(String category, String text) {
        insights.add(new Insight(category, text));
        LogService.info("KnowledgeBase", "New Insight: [" + category + "] " + text);
    }

    public List<Insight> getAllInsights() {
        return new ArrayList<>(insights);
    }

    public void updateInsight(int index, String newText) {
        if (index >= 0 && index < insights.size()) {
            insights.get(index).setText(newText);
            LogService.info("KnowledgeBase", "Insight updated at index " + index);
        }
    }

    // --- Inner Class ---
    public static class Insight {
        private String category;
        private String text;

        public Insight(String category, String text) {
            this.category = category;
            this.text = text;
        }

        public String getCategory() {
            return category;
        }

        public String getText() {
            return text;
        }

        public void setText(String text) {
            this.text = text;
        }

        @Override
        public String toString() {
            return "[" + category + "] " + text;
        }
    }
}
