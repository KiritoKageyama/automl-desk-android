package com.example.automl_prototype_1.service.agentic;

import com.example.automl_prototype_1.service.LogService;
import com.example.automl_prototype_1.service.IClassifier;
import com.example.automl_prototype_1.classifier.cpu.KnnClassifier;
// or from registry if preferred, but direct CPU usage is safe for prototype logic

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Service for Behavioral Cloning (Imitation Learning).
 * Records (State, Action) pairs from the user and trains a policy (KNN) to
 * mimic them.
 */
public class BehavioralCloningService {

    private final ScreenWatcherService screenWatcher;
    private final List<DemonstrationStep> currentDemo;
    private final AtomicBoolean isRecording;

    // The current action being pressed by the user (injected from UI)
    private volatile int currentAction = 0;

    // The trained model
    private IClassifier trainedPolicy;

    public BehavioralCloningService() {
        this.screenWatcher = new ScreenWatcherService();
        this.currentDemo = new ArrayList<>();
        this.isRecording = new AtomicBoolean(false);
    }

    public void startRecording(String profileName) {
        currentDemo.clear();
        isRecording.set(true);
        LogService.info("BehavioralCloning", "Started recording demonstration (" + profileName + ")...");
        new Thread(this::recordLoop).start();
    }

    public void stopRecording() {
        isRecording.set(false);
        LogService.info("BehavioralCloning", "Stopped recording. Captured " + currentDemo.size() + " frames.");
    }

    /**
     * Called by UI when user presses a key.
     */
    public void setCurrentAction(int action) {
        this.currentAction = action;
    }

    private void recordLoop() {
        while (isRecording.get()) {
            long startTime = System.currentTimeMillis();

            // 1. Capture State
            double[] state = screenWatcher.captureState();

            // 2. Capture Action (from volatile field updated by UI)
            int action = currentAction;

            if (state != null && state.length > 0) {
                currentDemo.add(new DemonstrationStep(state, action));
            }

            // Target ~10 FPS
            long elapsed = System.currentTimeMillis() - startTime;
            long sleep = Math.max(0, 100 - elapsed);
            try {
                Thread.sleep(sleep);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        }
    }

    /**
     * Trains a specific classifier to clone behavioral patterns.
     * Uses KNN by default for speed and simplicity in prototype.
     * 
     * @return The trained model.
     */
    public IClassifier trainPolicy() {
        if (currentDemo.isEmpty()) {
            LogService.warn("BehavioralCloning", "No data to train on.");
            return null;
        }

        LogService.info("BehavioralCloning", "Training policy on " + currentDemo.size() + " samples...");

        // Convert to arrays
        int nObj = currentDemo.size();
        int nFeat = currentDemo.get(0).state.length;

        double[][] X = new double[nObj][nFeat];
        int[] y = new int[nObj];

        for (int i = 0; i < nObj; i++) {
            X[i] = currentDemo.get(i).state;
            y[i] = currentDemo.get(i).action;
        }

        // Use KNN for fast learning
        this.trainedPolicy = new KnnClassifier();
        // Or: ClassifierRegistry.getInstance().getClassifier("KNN");

        try {
            this.trainedPolicy.train(X, y);
            LogService.info("BehavioralCloning", "Training complete. Policy is ready.");
        } catch (Exception e) {
            LogService.error("BehavioralCloning", "Training failed: " + e.getMessage());
            e.printStackTrace();
        }

        return this.trainedPolicy;
    }

    /**
     * Predicts the next action based on the current state using the cloned policy.
     */
    public int predict(double[] state) {
        if (trainedPolicy == null)
            return 0; // Default or Random fallback

        try {
            double[][] input = new double[1][state.length];
            input[0] = state;
            int[] result = trainedPolicy.predict(input);
            return result.length > 0 ? result[0] : 0;
        } catch (Exception e) {
            LogService.error("BehavioralCloning", "Prediction error: " + e.getMessage());
            return 0;
        }
    }

    public static class DemonstrationStep {
        public final double[] state;
        public final int action;

        public DemonstrationStep(double[] state, int action) {
            this.state = state;
            this.action = action;
        }
    }
}
