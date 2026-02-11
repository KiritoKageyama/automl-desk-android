package com.example.automl_prototype_1.service.agentic;

import com.example.automl_prototype_1.service.LogService;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Service for Active Exploration (Curiosity-Driven Learning).
 * Uses a Forward Dynamics Model: f(state, action) -> next_state
 * Intrinsic Reward = ||Predicted_Next_State - Actual_Next_State||^2
 */
public class ActiveExplorationService {

    private final InputSimService inputSim;
    private final ScreenWatcherService screenWatcher;
    private final AtomicBoolean isExploring;
    private final Random random;

    // Simple Dynamics Model: Map<StateHash + Action, PredictedChange>
    // In a full deep RL system, this would be a Neural Network (ICM).
    // Here we use a tabular-like approximation for the prototype.
    private final Map<String, double[]> dynamicsModel;

    public ActiveExplorationService() {
        this.inputSim = new InputSimService();
        this.screenWatcher = new ScreenWatcherService();
        this.isExploring = new AtomicBoolean(false);
        this.random = new Random();
        this.dynamicsModel = new HashMap<>();
    }

    public void setProfile(String profile) {
        inputSim.setKeyProfile(profile);
    }

    public void startExploring() {
        if (isExploring.get())
            return;
        isExploring.set(true);
        LogService.info("ActiveExploration", "Starting Curiosity-Driven Exploration...");

        new Thread(() -> {
            try {
                Thread.sleep(3000); // 3s buffer
                explorationLoop();
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        }).start();
    }

    public void stopExploring() {
        isExploring.set(false);
        LogService.info("ActiveExploration", "Stopped exploration.");
    }

    private void explorationLoop() {
        double[] state = screenWatcher.captureState();
        int step = 0;

        while (isExploring.get()) {
            step++;
            String stateKey = encodeState(state);

            // 1. Choose Action: Maximize Prediction Error (Curiosity)
            // Or Epsilon-Greedy to balance random exploration vs uncertainty sampling
            int action = chooseUncertainAction(stateKey);

            // 2. Predict Outcome (Mental Model)
            double[] predictedChange = dynamicsModel.getOrDefault(stateKey + "_" + action, new double[state.length]);

            // 3. Execute
            inputSim.performAction(action);
            try {
                Thread.sleep(200);
            } catch (Exception e) {
            } // Wait for world to react

            // 4. Observe
            double[] nextState = screenWatcher.captureState();
            double[] actualChange = diff(nextState, state); // delta = s' - s

            // 5. Compute Intrinsic Reward (Prediction Error)
            double error = mse(predictedChange, actualChange);

            // 6. Learn (Update Model)
            // Simple Exponential Moving Average (Learning Rate = 0.5)
            updateModel(stateKey, action, actualChange);

            // 7. Log "Aha!" Moments
            if (error > 0.05) {
                String discovery = String.format(
                        "Step %d: Surprise! Action %d caused unexpected change (Error: %.4f). Learning physics...",
                        step, action, error);
                LogService.info("Curiosity", discovery);

                // If highly surprising, save to Knowledge Base
                if (error > 0.2) {
                    KnowledgeBaseService.getInstance().addInsight("Physics Discovery",
                            "Action " + action + " has strong effect in state " + stateKey + " (Magnitude: "
                                    + magnitude(actualChange) + ")");
                }
            }

            state = nextState;
        }
    }

    /**
     * Chooses the action about which the model is most uncertain (or random).
     */
    private int chooseUncertainAction(String stateKey) {
        // 10% Random noise
        if (random.nextDouble() < 0.1)
            return random.nextInt(6);

        // Else check model existence
        int bestAction = random.nextInt(6);
        // In this simple tabular model, "uncertainty" = "haven't tried it yet"
        for (int a = 0; a < 6; a++) {
            if (!dynamicsModel.containsKey(stateKey + "_" + a)) {
                return a; // Try unvisited action immediately
            }
        }
        return bestAction; // If all visited, random (or could pick highest variance if we tracked it)
    }

    private void updateModel(String stateKey, int action, double[] actualChange) {
        String key = stateKey + "_" + action;
        double[] predicted = dynamicsModel.getOrDefault(key, new double[actualChange.length]);

        // Learning Rate 0.5
        for (int i = 0; i < actualChange.length; i++) {
            predicted[i] = predicted[i] + 0.5 * (actualChange[i] - predicted[i]);
        }
        dynamicsModel.put(key, predicted);
    }

    private String encodeState(double[] s) {
        // Discretize for tabular hashing
        StringBuilder sb = new StringBuilder();
        for (double v : s)
            sb.append(Math.round(v * 5.0) / 5.0).append(",");
        return sb.toString();
    }

    private double[] diff(double[] s_next, double[] s) {
        double[] d = new double[s.length];
        for (int i = 0; i < s.length; i++)
            d[i] = s_next[i] - s[i];
        return d;
    }

    private double mse(double[] a, double[] b) {
        double sum = 0;
        for (int i = 0; i < a.length; i++)
            sum += Math.pow(a[i] - b[i], 2);
        return sum / a.length;
    }

    private double magnitude(double[] v) {
        double sum = 0;
        for (double d : v)
            sum += d * d;
        return Math.sqrt(sum);
    }
}
