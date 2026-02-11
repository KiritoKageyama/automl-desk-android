package com.example.automl_prototype_1.rl;

import com.example.automl_prototype_1.rl.agent.IRLAgent;
import com.example.automl_prototype_1.rl.agent.TabularQLearningAgent;
import com.example.automl_prototype_1.service.LogService;
import java.util.concurrent.atomic.AtomicBoolean;

public class RlTrainingService {

    private final AtomicBoolean isRunning = new AtomicBoolean(false);
    private IEnvironment environment;
    private IRLAgent agent;

    public void startTraining(IEnvironment env, int episodes) {
        if (isRunning.get()) {
            LogService.warn("RL", "Training already in progress.");
            return;
        }
        this.environment = env;
        // Initialize Q-Learning Agent
        this.agent = new TabularQLearningAgent(env.getActionSpaceSize());

        this.isRunning.set(true);

        new Thread(() -> {
            LogService.info("RL",
                    "Starting Q-Learning on " + env.getName() + " for " + episodes + " episodes.");

            for (int i = 0; i < episodes; i++) {
                if (!isRunning.get())
                    break;

                runEpisode(i);

                // Optional: Slow down for visualization if needed, but for training we want
                // speed.
                // We typically run headless training fast, demo slow.
                try {
                    Thread.sleep(10); // 10ms delay to allow UI to breathe
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                }
            }

            LogService.info("RL", "Training complete. Final Epsilon: "
                    + String.format("%.3f", ((TabularQLearningAgent) agent).getEpsilon()));
            isRunning.set(false);

            // Auto-save model
            agent.save("trained_model_" + env.getName() + ".json");
        }).start();
    }

    public void stopTraining() {
        isRunning.set(false);
    }

    private void runEpisode(int episodeNum) {
        double[] state = environment.reset();
        boolean done = false;
        double totalReward = 0;

        while (!done && isRunning.get()) {
            // 1. Agent chooses action
            int action = agent.chooseAction(state);

            // 2. Env executes action
            IEnvironment.StepResult result = environment.step(action);

            // 3. Agent learns
            agent.learn(state, action, result.reward, result.state, result.done);

            // Update state
            state = result.state;
            totalReward += result.reward;
            done = result.done;
        }

        if (episodeNum % 25 == 0) {
            LogService.info("RL", String.format("Episode %d | Reward: %.2f | Epsilon: %.3f",
                    episodeNum, totalReward, ((TabularQLearningAgent) agent).getEpsilon()));
        }
    }

    /**
     * Phase 7: Kickstart Training from Behavioral Clone.
     * Runs episodes using the User Policy to guide the agent, populating the
     * Q-Table.
     */
    public void pretrainFromCloning(IEnvironment env, com.example.automl_prototype_1.service.IClassifier userPolicy,
            int episodes) {
        if (isRunning.get()) {
            LogService.warn("RL", "Training active. Stop first.");
            return;
        }
        this.environment = env;
        // Initialize if null, else keep existing to continue learning
        if (this.agent == null) {
            this.agent = new TabularQLearningAgent(env.getActionSpaceSize());
        }

        this.isRunning.set(true);

        new Thread(() -> {
            LogService.info("RL", "Starting Kickstart (Pre-training) for " + episodes + " episodes...");

            for (int i = 0; i < episodes; i++) {
                if (!isRunning.get())
                    break;

                // Run episode driven by User Policy
                double[] state = environment.reset();
                boolean done = false;

                while (!done && isRunning.get()) {
                    // 1. Ask User Policy what to do
                    int action = 0;
                    try {
                        // Adapter: double[] -> double[][]
                        double[][] input = new double[1][state.length];
                        input[0] = state;
                        int[] preds = userPolicy.predict(input);
                        if (preds.length > 0)
                            action = preds[0];
                    } catch (Exception e) {
                        LogService.error("RL", "Kickstart Policy Error: " + e.getMessage());
                    }

                    // 2. Execute
                    IEnvironment.StepResult result = environment.step(action);

                    // 3. Agent learns from this experience (Off-Policy / Demonstration)
                    agent.learn(state, action, result.reward, result.state, result.done);

                    state = result.state;
                    done = result.done;
                }

                try {
                    Thread.sleep(5);
                } catch (Exception e) {
                }
            }

            LogService.info("RL", "Kickstart complete. Q-Table seeded.");
            isRunning.set(false);
        }).start();
    }

    public IRLAgent getAgent() {
        return agent;
    }
}
