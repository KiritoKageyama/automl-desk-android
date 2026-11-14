package com.example.automl_prototype_1.service.agentic;

import com.example.automl_prototype_1.service.LogService;
import com.example.automl_prototype_1.service.IClassifier;
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
    private volatile int currentMouseButton = 0; // 0=None, 1=Left, 3=Right

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

    public void setCurrentMouseButton(int button) {
        this.currentMouseButton = button;
    }

    private final VisionStateDetector visionDetector = new VisionStateDetector();

    private void recordLoop() {
        while (isRecording.get()) {
            long startTime = System.currentTimeMillis();

            // 1. Capture State (Color Image for Vision, Tensor for RL)
            java.awt.image.BufferedImage img = screenWatcher.captureImage();
            double[] state = screenWatcher.processImageToTensor(img);

            // 2. Vision Analysis (Drift/Nitrous detection)
            VisionStateDetector.GameState gameState = visionDetector.analyze(img);
            if (gameState.isNitrous) {
                LogService.info("Vision", "NITROUS DETECTED! (Blue Flame)");
            }
            // For prototype optimization, only log others if interesting
            // if (gameState.isDrifting) LogService.info("Vision", "Drifting...");

            // 3. Capture Action (from volatile field updated by UI/Hooks)
            int action = currentAction;

            // 4. Capture Mouse (Polling AWT if no hook event, or use volatile from hook)
            // For now, let's Poll java.awt.MouseInfo for position, but rely on hook for
            // Clicks (via currentMouseButton)
            java.awt.Point mousePos = java.awt.MouseInfo.getPointerInfo().getLocation();
            int mx = mousePos.x;
            int my = mousePos.y;
            int mb = currentMouseButton; // volatile field

            if (state != null && state.length > 0) {
                currentDemo.add(new DemonstrationStep(state, action, mx, my, mb));
            }

            // Reset click after capture (so we don't record continuous clicks for one
            // press)
            // Actually, keep it until released?
            // Better: The recorder runs at 10fps. If user holds click, we record "Click
            // Down".
            // Implementation choice: GlobalHook sets currentMouseButton=1 on Press, 0 on
            // Release.

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

    // We store the data for Lazy Learning (KNN) directly
    private double[][] X_train;
    private int[] y_action_train;
    private int[] y_mx_train;
    private int[] y_my_train;
    private int[] y_mb_train;

    public void trainPolicy() {
        if (currentDemo.isEmpty()) {
            LogService.warn("BehavioralCloning", "No data to train on.");
            return;
        }

        LogService.info("BehavioralCloning", "Training (indexing) " + currentDemo.size() + " samples...");

        int n = currentDemo.size();
        int d = currentDemo.get(0).state.length;

        X_train = new double[n][d];
        y_action_train = new int[n];
        y_mx_train = new int[n];
        y_my_train = new int[n];
        y_mb_train = new int[n];

        for (int i = 0; i < n; i++) {
            DemonstrationStep step = currentDemo.get(i);
            X_train[i] = step.state;
            y_action_train[i] = step.action;
            y_mx_train[i] = step.mouseX;
            y_my_train[i] = step.mouseY;
            y_mb_train[i] = step.mouseButton;
        }

        LogService.info("BehavioralCloning", "Training complete. Ready to Imitate.");
    }

    public static class AgentAction {
        public int keyAction;
        public int mouseX;
        public int mouseY;
        public int mouseButton;
    }

    public AgentAction predictAction(double[] state) {
        if (X_train == null || X_train.length == 0)
            return new AgentAction();

        // 1. Find K nearest neighbors (K=5)
        // Simple linear scan for prototype (O(N)). For large N, use KD-Tree (Smile).
        // Since N is small (< 10000 frames), linear is OK.
        int k = 5;
        k = Math.min(k, X_train.length);

        int[] nearestIndices = new int[k];
        double[] nearestDists = new double[k];
        for (int i = 0; i < k; i++)
            nearestDists[i] = Double.MAX_VALUE;

        for (int i = 0; i < X_train.length; i++) {
            double dist = euclidean(state, X_train[i]);
            // Insert into sorted list if closer
            if (dist < nearestDists[k - 1]) {
                // Find position
                int j = k - 2;
                while (j >= 0 && nearestDists[j] > dist) {
                    nearestDists[j + 1] = nearestDists[j];
                    nearestIndices[j + 1] = nearestIndices[j];
                    j--;
                }
                nearestDists[j + 1] = dist;
                nearestIndices[j + 1] = i;
            }
        }

        // 2. Aggregate
        AgentAction result = new AgentAction();

        // Regression for Mouse X/Y (Average)
        long sumX = 0, sumY = 0;
        // Voting for Discrete Action/Button
        java.util.Map<Integer, Integer> actionCounts = new java.util.HashMap<>();
        java.util.Map<Integer, Integer> buttonCounts = new java.util.HashMap<>();

        for (int idx : nearestIndices) {
            sumX += y_mx_train[idx];
            sumY += y_my_train[idx];

            actionCounts.put(y_action_train[idx], actionCounts.getOrDefault(y_action_train[idx], 0) + 1);
            buttonCounts.put(y_mb_train[idx], buttonCounts.getOrDefault(y_mb_train[idx], 0) + 1);
        }

        result.mouseX = (int) (sumX / k);
        result.mouseY = (int) (sumY / k);
        result.keyAction = getMaxVote(actionCounts);
        result.mouseButton = getMaxVote(buttonCounts);

        return result;
    }

    private int getMaxVote(java.util.Map<Integer, Integer> counts) {
        int max = -1;
        int best = 0;
        for (java.util.Map.Entry<Integer, Integer> e : counts.entrySet()) {
            if (e.getValue() > max) {
                max = e.getValue();
                best = e.getKey();
            }
        }
        return best;
    }

    private double euclidean(double[] a, double[] b) {
        double sum = 0;
        for (int i = 0; i < a.length; i++)
            sum += (a[i] - b[i]) * (a[i] - b[i]);
        return Math.sqrt(sum);
    }

    /**
     * Legacy support for RlVisualizerController (GridWorld).
     * Returns just the key action.
     */
    public int predict(double[] state) {
        return predictAction(state).keyAction;
    }

    public static class DemonstrationStep {
        public final double[] state;
        public final int action; // Key Action
        public final int mouseX;
        public final int mouseY;
        public final int mouseButton; // 0=None, 1=Left, 3=Right

        public DemonstrationStep(double[] state, int action, int mouseX, int mouseY, int mouseButton) {
            this.state = state;
            this.action = action;
            this.mouseX = mouseX;
            this.mouseY = mouseY;
            this.mouseButton = mouseButton;
        }
    }
}
