package com.example.automl_prototype_1.rl;

/**
 * Interface representing a Reinforcement Learning Environment.
 * Modeled after OpenAI Gym / Gymnasium API.
 */
public interface IEnvironment {

    /**
     * Resets the environment to an initial state.
     * 
     * @return The initial observation/state.
     */
    double[] reset();

    /**
     * Takes a step in the environment using the given action.
     * 
     * @param action The action to take (integer for discrete action spaces).
     * @return A StepResult containing the new state, reward, done flag, and info.
     */
    StepResult step(int action);

    /**
     * Renders the current state of the environment.
     * 
     * @return A string representation or visualization data.
     */
    String render();

    /**
     * Gets the size of the observation space.
     * 
     * @return dimension of the state vector.
     */
    int getObservationSpaceSize();

    /**
     * Gets the size of the action space.
     * 
     * @return number of possible actions (for discrete spaces).
     */
    int getActionSpaceSize();

    /**
     * Returns the name of the environment.
     */
    String getName();

    /**
     * Renders the environment graphically onto the provided context.
     * We use Object to allow flexibility (e.g. JavaFX GraphicsContext),
     * but implementations should expect the platform's specific context.
     * 
     * @param graphicsContext The graphics context (e.g.
     *                        javafx.scene.canvas.GraphicsContext).
     */
    default void draw(Object graphicsContext) {
        // Default no-op
    }

    // Helper class for step returns
    public static class StepResult {
        public final double[] state;
        public final double reward;
        public final boolean done;
        public final String info;

        public StepResult(double[] state, double reward, boolean done, String info) {
            this.state = state;
            this.reward = reward;
            this.done = done;
            this.info = info;
        }
    }
}
