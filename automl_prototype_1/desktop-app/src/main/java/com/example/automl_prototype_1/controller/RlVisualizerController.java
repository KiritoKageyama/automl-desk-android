package com.example.automl_prototype_1.controller;

import com.example.automl_prototype_1.rl.IEnvironment;
import com.example.automl_prototype_1.rl.extensions.GridWorldEnv;
import com.example.automl_prototype_1.rl.agent.TabularQLearningAgent;
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
    private final ActiveExplorationService explorationService = new ActiveExplorationService();
    private final ExpertStrategyService expertService = new ExpertStrategyService();
    private final BehavioralCloningService cloningService = new BehavioralCloningService();
    private final IntrospectionService introspectionService = new IntrospectionService();

    private IEnvironment currentEnv;
    private TabularQLearningAgent qAgent;
    private final AtomicBoolean isRunning = new AtomicBoolean(false);
    private boolean useClonedPolicy = false;
    private int episodeCount = 0;

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
            ensureQAgent();
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

        if (injectStrategyBtn != null) {
            injectStrategyBtn.setOnAction(e -> handleInjectStrategy());
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
            ensureQAgent(); // fresh Q-table for new env
            episodeCount = 0;
            LogService.info("Visualizer", "Switched to environment: " + name);
            draw();
            statusLabel.setText("Status: Loaded " + name + " (Q-Learning ready)");
            totalReward = 0;
            rewardLabel.setText("Reward: 0.0");
        } else {
            LogService.error("Visualizer", "Failed to load environment: " + name);
        }
    }

    private void ensureQAgent() {
        if (currentEnv == null)
            return;
        qAgent = new TabularQLearningAgent(currentEnv.getActionSpaceSize());
        LogService.info("Visualizer", "Initialized Tabular Q-Learning agent (ε="
                + String.format("%.2f", qAgent.getEpsilon()) + ")");
    }

    private int mapKeyToAction(KeyCode code) {
        // Must match extensions.GridWorldEnv: 0=Up, 1=Down, 2=Left, 3=Right
        return switch (code) {
            case W, UP -> 0;
            case S, DOWN -> 1;
            case A, LEFT -> 2;
            case D, RIGHT -> 3;
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
        if (currentEnv == null) {
            statusLabel.setText("Status: No environment loaded");
            return;
        }
        if (qAgent == null)
            ensureQAgent();
        isRunning.set(true);
        useClonedPolicy = false; // Default to Q-Learning
        statusLabel.setText(String.format("Status: Q-Learning (ε=%.2f)", qAgent.getEpsilon()));

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
        ensureQAgent(); // reset Q-table with environment
        episodeCount = 0;
        totalReward = 0;
        stepCount = 0;
        recentRewards.clear();
        rewardLabel.setText("Reward: 0.0");
        draw();
        statusLabel.setText("Status: Reset (new Q-table)");
    }

    private void stepEnv() {
        if (currentEnv == null || lastState == null)
            return;

        stepCount++;
        int action;
        double[] stateBefore = lastState;

        // 1. Choose Action (Cloned Policy vs Q-Learning)
        if (useClonedPolicy) {
            action = cloningService.predict(lastState);
        } else {
            if (qAgent == null)
                ensureQAgent();
            action = qAgent.chooseAction(lastState);
        }

        IEnvironment.StepResult result = currentEnv.step(action);
        lastState = result.state;
        totalReward += result.reward;

        // Q-Learning update (skip when running cloned demo-only mode)
        if (!useClonedPolicy && qAgent != null) {
            qAgent.learn(stateBefore, action, result.reward, result.state, result.done);
        }

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
                Platform.runLater(() -> statusLabel.setText("Status: " + analysis.advice));
            }
        }

        rewardLabel.setText(String.format("Reward: %.2f", totalReward));
        if (!useClonedPolicy && qAgent != null && stepCount % 10 == 0) {
            statusLabel.setText(String.format("Status: Q-Learning ep=%d ε=%.2f", episodeCount, qAgent.getEpsilon()));
        }
        draw();

        if (result.done) {
            episodeCount++;
            LogService.info("Visualizer", String.format(
                    "Episode %d done | Reward: %.2f | ε=%.3f",
                    episodeCount, totalReward,
                    qAgent != null ? qAgent.getEpsilon() : -1));
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
            Alert confirm = new Alert(Alert.AlertType.CONFIRMATION);
            confirm.setTitle("Desktop automation");
            confirm.setHeaderText("Active Exploration presses keys via Robot.");
            confirm.setContentText("Enable Robot automation and start?");
            if (confirm.showAndWait().orElse(javafx.scene.control.ButtonType.CANCEL)
                    != javafx.scene.control.ButtonType.OK) {
                return;
            }
            com.example.automl_prototype_1.service.DesktopAutomationGate.getInstance().setEnabled(true);

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
            com.example.automl_prototype_1.service.DesktopAutomationGate.getInstance().setEnabled(false);
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

    @FXML
    private Button injectStrategyBtn;

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
        if (currentEnv == null) {
            statusLabel.setText("Status: No environment loaded");
            return;
        }
        statusLabel.setText("Status: Kickstarting RL from clone...");
        kickstartBtn.setDisable(true);
        cloningService.trainPolicy();
        if (qAgent == null)
            ensureQAgent();

        final TabularQLearningAgent agent = qAgent;
        final IEnvironment env = currentEnv;
        new Thread(() -> {
            int seeded = 0;
            try {
                for (int ep = 0; ep < 50; ep++) {
                    double[] state = env.reset();
                    boolean done = false;
                    int steps = 0;
                    while (!done && steps < 200) {
                        int action = cloningService.predict(state);
                        IEnvironment.StepResult result = env.step(action);
                        agent.learn(state, action, result.reward, result.state, result.done);
                        state = result.state;
                        done = result.done;
                        steps++;
                    }
                    seeded++;
                }
                LogService.info("Visualizer", "Kickstart seeded Q-table from clone (" + seeded + " episodes).");
            } catch (Exception e) {
                LogService.error("Visualizer", "Kickstart failed: " + e.getMessage());
            }
            Platform.runLater(() -> {
                statusLabel.setText(String.format("Status: Kickstart done (ε=%.2f). Press Start.",
                        agent.getEpsilon()));
                kickstartBtn.setDisable(false);
                // Reset env visual state after background episodes
                lastState = currentEnv.reset();
                totalReward = 0;
                draw();
            });
        }, "rl-kickstart").start();
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

    private void handleInjectStrategy() {
        String game = envCombo != null && envCombo.getValue() != null
                ? envCombo.getValue()
                : "GridWorld";
        statusLabel.setText("Status: Asking LLM for strategy priors...");
        injectStrategyBtn.setDisable(true);

        new Thread(() -> {
            try {
                StrategyInjectorService injector = new StrategyInjectorService();
                java.util.Map<String, Double> priors = injector.fetchStrategyGuide(game);
                Platform.runLater(() -> {
                    ensureQAgent();
                    if (priors.isEmpty()) {
                        statusLabel.setText("Status: No strategy returned (LLM offline?)");
                    } else {
                        qAgent.injectPriors(priors);
                        statusLabel.setText("Status: Injected " + priors.size() + " strategy priors");
                        if (insightsList != null) {
                            insightsList.getItems().clear();
                            priors.forEach((k, v) -> insightsList.getItems()
                                    .add(String.format("%s → %.2f", k, v)));
                        }
                    }
                    injectStrategyBtn.setDisable(false);
                });
            } catch (Exception e) {
                Platform.runLater(() -> {
                    statusLabel.setText("Status: Strategy inject failed: " + e.getMessage());
                    injectStrategyBtn.setDisable(false);
                });
            }
        }, "rl-strategy-inject").start();
    }
}
