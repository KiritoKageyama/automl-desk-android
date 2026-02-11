package com.example.automl_prototype_1.rl.agent;

import com.example.automl_prototype_1.service.LogService;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.File;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;

/**
 * A standard Tabular Q-Learning Agent.
 * State is hashed to a String key to support arbitrary observation dimensions.
 * Formula: Q(s,a) = Q(s,a) + alpha * [r + gamma * max(Q(s',a')) - Q(s,a)]
 */
public class TabularQLearningAgent implements IRLAgent {

    private final int actionSpaceSize;
    private final double alpha; // Learning rate (0.1)
    private final double gamma; // Discount factor (0.99)
    private double epsilon; // Exploration rate
    private final double epsilonDecay; // Decay per episode (0.995)
    private final double epsilonMin; // Min epsilon (0.01)

    // Q-Table: Map<StateKey, double[NumActions]>
    private Map<String, double[]> qTable;
    private final Random random;
    private final ObjectMapper mapper;

    public TabularQLearningAgent(int actionSpaceSize) {
        this.actionSpaceSize = actionSpaceSize;
        this.alpha = 0.1;
        this.gamma = 0.95;
        this.epsilon = 1.0;
        this.epsilonDecay = 0.995;
        this.epsilonMin = 0.05;

        this.qTable = new HashMap<>();
        this.random = new Random();
        this.mapper = new ObjectMapper();
    }

    @Override
    public int chooseAction(double[] state) {
        String stateKey = encodeState(state);

        // Exploration: Epsilon-Greedy
        if (random.nextDouble() < epsilon) {
            return random.nextInt(actionSpaceSize);
        }

        // Exploitation: ArgMax Q(s, :)
        return getBestAction(stateKey);
    }

    @Override
    public void learn(double[] state, int action, double reward, double[] nextState, boolean done) {
        String stateKey = encodeState(state);
        String nextStateKey = encodeState(nextState);

        double[] qValues = getQValues(stateKey);
        double[] nextQValues = getQValues(nextStateKey);

        // Max Q(s', a')
        double maxNextQ = -Double.MAX_VALUE;
        for (double q : nextQValues) {
            if (q > maxNextQ)
                maxNextQ = q;
        }
        // If done, future reward is 0
        double target = reward + (done ? 0 : gamma * maxNextQ);

        // Q(s,a) = Q(s,a) + alpha * (Target - Q(s,a))
        qValues[action] += alpha * (target - qValues[action]);

        // Decay Epsilon (only if we learned something i.e. episode steps ideally, but
        // calling here is fine)
        // Usually decay is per episode, handled by trainer or here if 'done' is true.
        if (done) {
            decayEpsilon();
        }
    }

    private void decayEpsilon() {
        if (epsilon > epsilonMin) {
            epsilon *= epsilonDecay;
        }
    }

    private int getBestAction(String stateKey) {
        double[] qValues = getQValues(stateKey);
        int bestAction = 0;
        double maxVal = -Double.MAX_VALUE;

        // Simple ArgMax, randomized tie-breaking can be added if needed
        for (int i = 0; i < actionSpaceSize; i++) {
            if (qValues[i] > maxVal) {
                maxVal = qValues[i];
                bestAction = i;
            }
        }
        return bestAction;
    }

    // --- GenAI Injection ---

    private Map<String, Double> priorKnowledge = new HashMap<>();

    /**
     * Injects prior knowledge (concepts) into the agent.
     * The Agent uses these to initialize Q-values for matching states.
     *
     * @param priors Map of Concept -> Value (e.g., "center" -> 0.9)
     */
    public void injectPriors(Map<String, Double> priors) {
        this.priorKnowledge.putAll(priors);
        LogService.info("QLearning", "Injected " + priors.size() + " prior concepts into Agent.");
    }

    /**
     * Helper to get Q-Values, initiating with Priors if empty.
     */
    private double[] getQValues(String stateKey) {
        if (qTable.containsKey(stateKey)) {
            return qTable.get(stateKey);
        }

        double[] qValues = new double[actionSpaceSize];
        // In a real system, we would map stateKey -> concept -> prior value.
        // For now, initialize with 0.0.
        qTable.put(stateKey, qValues);
        return qValues;
    }

    /**
     * Converts continuous double[] state into a discrete string key.
     * Rounds to 2 decimal places to bucket similar states.
     */
    private String encodeState(double[] state) {
        if (state == null)
            return "null";
        StringBuilder sb = new StringBuilder();
        for (double v : state) {
            sb.append(Math.round(v * 10.0) / 10.0).append(",");
        }
        return sb.toString();
    }

    public Map<String, double[]> getQTable() {
        return qTable;
    }

    @Override
    public String getName() {
        return "Tabular Q-Learning";
    }

    public double getEpsilon() {
        return epsilon;
    }

    // --- Persistence ---

    @Override
    public void save(String filepath) {
        try {
            mapper.writeValue(new File(filepath), qTable);
            LogService.info("QLearning", "Agent saved to " + filepath);
        } catch (IOException e) {
            LogService.error("QLearning", "Failed to save agent: " + e.getMessage());
        }
    }

    @Override
    @SuppressWarnings("unchecked")
    public void load(String filepath) {
        try {
            File f = new File(filepath);
            if (f.exists()) {
                this.qTable = mapper.readValue(f, HashMap.class);
                LogService.info("QLearning", "Agent loaded from " + filepath);
            }
        } catch (IOException e) {
            LogService.error("QLearning", "Failed to load agent: " + e.getMessage());
        }
    }
}
