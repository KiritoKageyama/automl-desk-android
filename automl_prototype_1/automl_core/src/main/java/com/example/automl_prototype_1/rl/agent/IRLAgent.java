package com.example.automl_prototype_1.rl.agent;

/**
 * Common interface for Reinforcement Learning agents.
 */
public interface IRLAgent {

    /**
     * Chooses an action based on the current state.
     * 
     * @param state The current observation from the environment.
     * @return The chosen action (integer index).
     */
    int chooseAction(double[] state);

    /**
     * Learning update step.
     * 
     * @param state     The previous state.
     * @param action    The action taken.
     * @param reward    The reward received.
     * @param nextState The new state.
     * @param done      Whether the episode ended.
     */
    void learn(double[] state, int action, double reward, double[] nextState, boolean done);

    /**
     * @return The name of the agent (e.g., "Tabular Q-Learning").
     */
    String getName();

    // Optional: Persistence
    default void save(String filepath) {
    }

    default void load(String filepath) {
    }
}
