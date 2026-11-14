package com.example.automl_prototype_1.model;

/**
 * Represents the distinct machine learning and AI domains supported by the
 * application.
 * Used for routing, state isolation, and context-specific AI advisor
 * instructions.
 */
public enum Domain {
    ML("Tabular Machine Learning"),
    DL("Deep Learning (Dense/Sequence)"),
    CNN("Computer Vision (CNN)"),
    LLM("Large Language Models / SLM"),
    GENAI("Generative AI & Scripts"),
    RL("Reinforcement Learning");

    private final String displayName;

    Domain(String displayName) {
        this.displayName = displayName;
    }

    public String getDisplayName() {
        return displayName;
    }
}
