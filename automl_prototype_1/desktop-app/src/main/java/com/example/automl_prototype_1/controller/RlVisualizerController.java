package com.example.automl_prototype_1.controller;

import com.example.automl_prototype_1.rl.IEnvironment;
import com.example.automl_prototype_1.rl.extensions.GridWorldEnv;
import com.example.automl_prototype_1.service.LogService;
import com.example.automl_prototype_1.service.agentic.ActiveExplorationService;
import com.example.automl_prototype_1.service.agentic.BehavioralCloningService;
import com.example.automl_prototype_1.service.agentic.ExpertStrategyService;
import com.example.automl_prototype_1.service.agentic.IntrospectionService;
import com.example.automl_prototype_1.service.agentic.KnowledgeBaseService;
import com.example.automl_prototype_1.rl.RlTrainingService;
import com.example.automl_prototype_1.service.agentic.StrategyInjectorService;
import com.example.automl_prototype_1.service.agentic.InsightExtractionService;
import javafx.animation.AnimationTimer;
import javafx.application.Platform;
import javafx.fxml.FXML;
import javafx.scene.canvas.Canvas;
import javafx.scene.canvas.GraphicsContext;
import javafx.scene.control.Alert;
import javafx.scene.control.Button;
import javafx.scene.control.ComboBox;
import javafx.scene.control.Label;
import javafx.scene.control.ListView;
import javafx.scene.control.TextInputDialog;
import javafx.scene.input.KeyCode;
import java.util.Optional;
import java.util.ArrayList;
import java.util.List;

import java.util.Random;
import java.util.concurrent.atomic.AtomicBoolean;

public class RlVisualizerController {

    @FXML
    private Canvas envCanvas;
    @FXML
    private Button startBtn;
    @FXML
    private Button stopBtn;
    @FXML
    private Button resetBtn;
    @FXML
    private Label statusLabel;
    @FXML
    private Label rewardLabel;

    // External Agent Controls
    @FXML
    private Button exploreBtn;
    @FXML
    private Button expertBtn;
    @FXML
    private ComboBox<String> profileCombo;
    @FXML
    private ListView<String> insightsList;
    @FXML
    private Button editInsightBtn;

    // Behavioral Cloning / User Pattern Learning
    @FXML
    private Button recordBtn;
    @FXML
    private Button stopRecordBtn;
    @FXML
    private Button trainClonedBtn;
    @FXML
    private Button runClonedBtn;

    // External Services
    private final RlTrainingService trainingService = new RlTrainingService();
    private final StrategyInjectorService strategyInjector = new StrategyInjectorService();
    private final InsightExtractionService insightService = new InsightExtractionService();
    private final ActiveExplorationService explorationService = new ActiveExplorationService();
    private final ExpertStrategyService expertService = new ExpertStrategyService();
    private final BehavioralCloningService cloningService = new BehavioralCloningService();
    private final IntrospectionService introspectionService = new IntrospectionService();

    private IEnvironment currentEnv;
    private final Random random = new Random();
    private final AtomicBoolean isRunning = new AtomicBoolean(false);
    private boolean useClonedPolicy = false;

    // Performance Tracking
    private final List<Double> recentRewards = new ArrayList<>();
    private AnimationTimer loop;
    private double totalReward = 0;
    private int stepCount = 0;

    // State Tracking
    private double[] lastState;

    @FXML
    private ComboBox<String> envCombo;

    @FXML
    public void initialize() {
        startBtn.setOnAction(e -> startTraining());
        stopBtn.setOnAction(e -> stopTraining());
        resetBtn.setOnAction(e -> resetEnv());

        // Setup Environment Selection
        if (envCombo != null) {
            reloadEnvironments();
            envCombo.getSelectionModel().selectedItemProperty().addListener((obs, oldVal, newVal) -> {
                if (newVal != null) {
                    loadEnvironment(newVal);
                }
            });
            // Select Default
            envCombo.getSelectionModel().select("GridWorldEnv");
        } else {
            // Fallback if FXML not updated yet (though we just did)
            currentEnv = new GridWorldEnv();
            lastState = currentEnv.reset();
        }

        LogService.info("Visualizer",
                "Initialized. Env: " + (currentEnv != null ? currentEnv.getClass().getSimpleName() : "None"));

        // External Agent
        if (exploreBtn != null) {
            exploreBtn.setOnAction(e -> handleExplore());
        }
        if (expertBtn != null) {
            expertBtn.setOnAction(e -> handleExpert());
        }

        if (editInsightBtn != null) {
            editInsightBtn.setOnAction(e -> handleEditInsight());
        }

        // Behavioral Cloning Handlers
        if (recordBtn != null) {
            recordBtn.setOnAction(e -> handleRecord());
            stopRecordBtn.setOnAction(e -> handleStopRecord());
            trainClonedBtn.setOnAction(e -> handleTrainCloned());
            runClonedBtn.setOnAction(e -> handleRunCloned());
            if (kickstartBtn != null) {
                kickstartBtn.setOnAction(e -> handleKickstart());
            }
        }

        // Input Handling
        envCanvas.setFocusTraversable(true);
        envCanvas.setOnKeyPressed(e -> {
            int action = mapKeyToAction(e.getCode());
            if (action != -1) {
                cloningService.setCurrentAction(action);
            }
        });
        envCanvas.setOnKeyReleased(e -> cloningService.setCurrentAction(0));

        // Start UI Update Loop for Insights
        startInsightUpdater();

        // Initial Draw
        draw();

        // Responsive Canvas
        if (envCanvas.getParent() instanceof javafx.scene.layout.Region) {
            javafx.scene.layout.Region parent = (javafx.scene.layout.Region) envCanvas.getParent();
            envCanvas.widthProperty().bind(parent.widthProperty());
            envCanvas.heightProperty().bind(parent.heightProperty());

            // Redraw on resize
            envCanvas.widthProperty().addListener(evt -> draw());
            envCanvas.heightProperty().addListener(evt -> draw());
        }
    }

    private void reloadEnvironments() {
        com.example.automl_prototype_1.registry.EnvironmentRegistry registry = com.example.automl_prototype_1.registry.EnvironmentRegistry
                .getInstance();
        registry.refresh();
        List<String> names = registry.getEnvironmentNames();
        envCombo.setItems(javafx.collections.FXCollections.observableArrayList(names));
    }

    private void loadEnvironment(String name) {
        stopTraining();
        IEnvironment newEnv = com.example.automl_prototype_1.registry.EnvironmentRegistry.getInstance()
                .createEnvironment(name);
        if (newEnv != null) {
            currentEnv = newEnv;
            lastState = currentEnv.reset();
            LogService.info("Visualizer", "Switched to environment: " + name);
            draw();
            statusLabel.setText("Status: Loaded " + name);
            totalReward = 0;
            rewardLabel.setText("Reward: 0.0");
        } else {
            LogService.error("Visualizer", "Failed to load environment: " + name);
        }
    }

    private int mapKeyToAction(KeyCode code) {
        // GridWorld: 0=Up, 1=Right, 2=Down, 3=Left
        return switch (code) {
            case W, UP -> 0;
            case D, RIGHT -> 1;
            case S, DOWN -> 2;
            case A, LEFT -> 3;
            default -> -1;
        };
    }

    private void startInsightUpdater() {
        new AnimationTimer() {
            private long lastUpdate = 0;

            @Override
            public void handle(long now) {
                if (now - lastUpdate >= 1_000_000_000) { // Update every 1s
                    updateInsightsList();
                    lastUpdate = now;
                }
            }
        }.start();
    }

    private void updateInsightsList() {
        if (insightsList == null)
            return;
        var insights = KnowledgeBaseService.getInstance().getAllInsights();
        Platform.runLater(() -> {
            if (insightsList.getItems().size() != insights.size()) {
                insightsList.getItems().clear();
                for (var insight : insights) {
                    insightsList.getItems().add(insight.toString());
                }
            }
        });
    }

    private void handleEditInsight() {
        String selected = insightsList.getSelectionModel().getSelectedItem();
        int index = insightsList.getSelectionModel().getSelectedIndex();
        if (selected == null || index < 0)
            return;

        TextInputDialog dialog = new TextInputDialog(selected);
        dialog.setTitle("Edit Insight");
        dialog.setHeaderText("Modify the Agent's Knowledge");
        dialog.setContentText("Rule:");

        Optional<String> result = dialog.showAndWait();
        result.ifPresent(newText -> {
            KnowledgeBaseService.getInstance().updateInsight(index, newText);
            updateInsightsList(); // Force refresh
        });
    }

    private void startTraining() {
        if (isRunning.get())
            return;
        isRunning.set(true);
        useClonedPolicy = false; // Default to random/RL
        statusLabel.setText("Status: Running (Training)");

        startLoop();
    }

    private void startLoop() {
        loop = new AnimationTimer() {
            private long lastUpdate = 0;

            @Override
            public void handle(long now) {
                if (!isRunning.get())
                    return;

                if (now - lastUpdate >= 200_000_000) { // 200ms delay per step for readability
                    stepEnv();
                    lastUpdate = now;
                }
            }
        };
        loop.start();
    }

    private void stopTraining() {
        isRunning.set(false);
        if (loop != null)
            loop.stop();
        statusLabel.setText("Status: Stopped");
    }

    private void resetEnv() {
        stopTraining();
        lastState = currentEnv.reset();
        totalReward = 0;
        stepCount = 0;
        recentRewards.clear();
        rewardLabel.setText("Reward: 0.0");
        draw();
        statusLabel.setText("Status: Reset");
    }

    private void stepEnv() {
        stepCount++;
        int action;

        // 1. Choose Action (Cloned Policy vs Random)
        if (useClonedPolicy && lastState != null) {
            // Use the trained KNN policy to predict action from state
            action = cloningService.predict(lastState);
        } else {
            // Random Agent Logic (Placeholder for actual AgenticAdvisor)
            action = random.nextInt(currentEnv.getActionSpaceSize());
        }

        IEnvironment.StepResult result = currentEnv.step(action);
        lastState = result.state; // Update state for next step
        totalReward += result.reward;

        // 2. Track Performance for Introspection
        recentRewards.add(result.reward);
        if (recentRewards.size() > 50)
            recentRewards.remove(0);

        // 3. Introspection Check (Every 20 steps)
        if (stepCount % 20 == 0 && recentRewards.size() >= 5) {
            double[] hist = recentRewards.stream().mapToDouble(d -> d).toArray();
            IntrospectionService.AnalysisResult analysis = introspectionService.analyzePerformanceTrend(hist);

            if (analysis.driftDetected || analysis.trend == IntrospectionService.AnalysisResult.Trend.DEGRADING) {
                LogService.warn("Introspection", "Issue detected: " + analysis.toString());
                KnowledgeBaseService.getInstance().addInsight("Self-Correction", analysis.advice);
                // Flash status
                Platform.runLater(() -> statusLabel.setText("Status: " + analysis.advice));
            }
        }

        rewardLabel.setText(String.format("Reward: %.2f", totalReward));
        draw();

        if (result.done) {
            LogService.info("Visualizer", "Episode Done! Reward: " + totalReward);
            lastState = currentEnv.reset();
            totalReward = 0;
        }
    }

    private void draw() {
        if (currentEnv != null) {
            GraphicsContext gc = envCanvas.getGraphicsContext2D();
            currentEnv.draw(gc);
        }
    }

    // --- External Agent Methods ---

    private void handleExplore() {
        if (exploreBtn.getText().contains("Start")) {
            // Set Profile
            String selected = "WASD";
            if (profileCombo != null && profileCombo.getValue() != null) {
                if (profileCombo.getValue().contains("ARROWS"))
                    selected = "ARROWS";
                else if (profileCombo.getValue().contains("TREX"))
                    selected = "TREX";
            }
            explorationService.setProfile(selected);

            explorationService.startExploring();
            exploreBtn.setText("Stop Exploration");
            statusLabel.setText("Status: Starting in 3s... SWITCH WINDOW!");
            LogService.info("Visualizer", "Active Exploration Started. Switch to game window!");
        } else {
            explorationService.stopExploring();
            exploreBtn.setText("Start Active Exploration");
            statusLabel.setText("Status: Stopped");
        }
    }

    private void handleExpert() {
        statusLabel.setText("Status: Asking Expert...");
        new Thread(() -> {
            String strategy = expertService.getProTechnique("Chrome T-Rex Runner", "General Strategy");
            Platform.runLater(() -> {
                LogService.info("Expert", "Strategy received: " + strategy);
                Alert alert = new Alert(Alert.AlertType.INFORMATION);
                alert.setTitle("Expert Strategy");
                alert.setHeaderText("AI Expert Advice");
                alert.setContentText(strategy);
                alert.showAndWait();
                statusLabel.setText("Status: Strategy Received");
            });
        }).start();
    }

    // --- Behavioral Cloning Methods ---

    private void handleRecord() {
        // Assume default profile or get from combo
        String profile = "WASD";
        if (profileCombo != null && profileCombo.getValue() != null) {
            if (profileCombo.getValue().contains("ARROWS"))
                profile = "ARROWS";
            else if (profileCombo.getValue().contains("TREX"))
                profile = "TREX";
        }

        cloningService.startRecording(profile);

        // Ensure we capture keys
        envCanvas.requestFocus();

        recordBtn.setDisable(true);
        stopRecordBtn.setDisable(false);
        trainClonedBtn.setDisable(true);
        runClonedBtn.setDisable(true);

        statusLabel.setText("Status: Recording... Play with WASD!");
        LogService.info("Visualizer", "Started recording user actions.");
    }

    private void handleStopRecord() {
        cloningService.stopRecording();

        recordBtn.setDisable(false);
        stopRecordBtn.setDisable(true);
        trainClonedBtn.setDisable(false); // Now we can train

        statusLabel.setText("Status: Recording Stopped. Ready to Train.");
        LogService.info("Visualizer", "Stopped recording.");
    }

    @FXML
    private Button kickstartBtn;

    // ...

    private void handleTrainCloned() {
        statusLabel.setText("Status: Training Policy...");
        trainClonedBtn.setDisable(true);

        new Thread(() -> {
            cloningService.trainPolicy();
            Platform.runLater(() -> {
                statusLabel.setText("Status: Training Complete");
                runClonedBtn.setDisable(false); // Enable cloning run
                kickstartBtn.setDisable(false); // Enable Kickstart
                LogService.info("Visualizer", "Cloned Policy Trained.");

                Alert alert = new Alert(Alert.AlertType.INFORMATION);
                alert.setTitle("Training Complete");
                alert.setHeaderText("Pattern Learned");
                alert.setContentText(
                        "The AI has learned your playing style.\nYou can now 'Run Cloned Policy' OR 'Kickstart RL Agent' to seed the learning process.");
                alert.showAndWait();
            });
        }).start();
    }

    private void handleKickstart() {
        if (trainingService == null)
            return;

        statusLabel.setText("Status: Kickstarting RL...");
        kickstartBtn.setDisable(true);

        // Use the cloned policy from cloningService
        com.example.automl_prototype_1.service.IClassifier policy = cloningService.trainPolicy(); // Get the trained one
                                                                                                  // (it caches)

        trainingService.pretrainFromCloning(currentEnv, policy, 100); // 100 Episodes

        // Wait loop or just assume it starts
        new Thread(() -> {
            try {
                Thread.sleep(2000);
            } catch (Exception e) {
            }
            Platform.runLater(() -> {
                statusLabel.setText("Status: Kickstart Complete. Q-Table Seeded.");
                LogService.info("Visualizer", "Kickstart finished.");
            });
        }).start();
    }

    private void handleRunCloned() {
        if (isRunning.get())
            return;

        useClonedPolicy = true;
        isRunning.set(true);
        statusLabel.setText("Status: Running Cloned Policy (Learned Behavior)");
        LogService.info("Visualizer", "Running Cloned Policy...");

        startLoop();
    }
}
