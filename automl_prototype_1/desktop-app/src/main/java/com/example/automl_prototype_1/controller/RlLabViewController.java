package com.example.automl_prototype_1.controller;

import com.example.automl_prototype_1.service.DesktopAutomationGate;
import com.example.automl_prototype_1.service.LogService;
import com.example.automl_prototype_1.rl.IEnvironment;
import com.example.automl_prototype_1.rl.RlTrainingService;
import com.example.automl_prototype_1.rl.TicTacToeEnv;

import javafx.fxml.FXML;
import javafx.scene.control.Alert;
import javafx.scene.control.Button;
import javafx.scene.control.ButtonType;
import javafx.scene.control.ComboBox;
import javafx.scene.control.TextArea;
import javafx.scene.control.TextField;
import javafx.application.Platform;

public class RlLabViewController {

    @FXML
    private ComboBox<String> envComboBox;
    @FXML
    private TextField episodesField;
    @FXML
    private Button trainBtn;
    @FXML
    private Button stopBtn;
    @FXML
    private TextArea renderArea;
    @FXML
    private TextArea logArea;

    private final RlTrainingService trainingService;
    private IEnvironment currentEnv;

    public RlLabViewController() {
        this.trainingService = new RlTrainingService();
    }

    @FXML
    public void initialize() {
        // Populate Environments
        envComboBox.getItems().addAll("Tic-Tac-Toe", "GridWorld", "NFS (Imitation Mode)");
        envComboBox.getSelectionModel().selectFirst();

        // Button Actions
        trainBtn.setOnAction(e -> {
            if (envComboBox.getValue().contains("Imitation") || envComboBox.getValue().contains("Demonstration")) {
                startCloning();
            } else {
                startTraining();
            }
        });
        stopBtn.setOnAction(e -> {
            if (envComboBox.getValue().contains("Imitation") || envComboBox.getValue().contains("Demonstration")) {
                stopCloning();
            } else {
                stopTraining();
            }
        });

        // --- Feature 2: Record Demonstration Button (Programmatic Injection) ---
        // Since we cannot edit FXML easily, we repurposed the 'Train' button logic
        // but let's add a clear way to invoke it.
        // We will rename the "NFS (Imitation Mode)" to "Record Demonstration (NFS)"
        envComboBox.getItems().add("Record Demonstration (NFS)");

        // Tooltips
        trainBtn.setTooltip(new javafx.scene.control.Tooltip("Start Training or Recording"));
        stopBtn.setTooltip(new javafx.scene.control.Tooltip("Stop current process"));
        envComboBox.setTooltip(new javafx.scene.control.Tooltip("Select Environment or Mode"));

        // Log redirection (simple poll for now or listener)
        // ideally LogService has a listener we can hook into.
        // For prototype, we'll just log manually or hook into a specialized logger if
        // available.
        // Or we can periodic poll? No.
        // better: let's add a simple static consumer to LogService for this view?
        // or just rely on RlTrainingService to return info?

        // For now, let's just show basic status updates
        log("RL Lab Initialized.");

        // Connect to Global Log Service
        com.example.automl_prototype_1.service.LogService.getInstance().addListener(logListener);
    }

    private void startTraining() {
        String envName = envComboBox.getValue();
        int episodes = 1000;
        try {
            episodes = Integer.parseInt(episodesField.getText());
        } catch (NumberFormatException e) {
            log("Invalid episodes number.");
            return;
        }

        if ("Tic-Tac-Toe".equals(envName)) {
            currentEnv = new TicTacToeEnv();
        } else if ("GridWorld".equals(envName)) {
            currentEnv = new com.example.automl_prototype_1.rl.GridWorldEnv(8, 10, 200);
            log("GridWorld 8×8 loaded: navigate 🤖 to goal 🎯 avoiding walls ██");
        } else {
            log("Environment '" + envName + "' is not available in Q-learning mode. Try TicTacToe or GridWorld.");
            return;
        }

        log("Starting training on " + envName + "...");
        trainBtn.setDisable(true);
        stopBtn.setDisable(false);
        envComboBox.setDisable(true);

        // Start async
        final int finalEpisodes = episodes;
        new Thread(() -> {
            trainingService.startTraining(currentEnv, finalEpisodes);

            // Simulation loop for UI updates (since RlTrainingService runs on its own
            // thread)
            // We want to see progress.
            // Actually RlTrainingService.startTraining spawns a thread.
            // We need to know when it finishes.
            // RlTrainingService is basic right now.
            // Let's just wait a bit and re-enable buttons or update UI periodically.

            // For the demo purpose, we can use a Timer to update render
            displayLoop();

        }).start();
    }

    private void stopTraining() {
        trainingService.stopTraining();
        log("Training stopped by user.");
        resetUi();
    }

    // Simple UI loop to show "Rendering" of the environment state occasionally
    private void handleSafetyInterrupt(String type) {
        if (!isAgentRunning)
            return;

        if (type.contains("MOUSE")) {
            // Granular: Stop Mouse Control ONLY
            isMouseControlActive = false;
            Platform.runLater(() -> {
                // if (statusLabel != null) statusLabel.setText("Status: Mouse Paused (User
                // Control)"); // statusLabel is not defined
                log("User Moved Mouse -> Agent Mouse Control PAUSED.");
            });
            return; // Continue loop, but without mouse
        }

        // KEYBOARD_INTERRUPT -> Stop EVERYTHING & Switch to Background Learning
        isAgentRunning = false;
        Platform.runLater(() -> {
            LogService.warn("Safety", "Agent Interrupted by User: " + type);
            log("Status: PAUSED by " + type);

            log("User Key Press Detected -> Switching to BACKGROUND LEARNING...");

            // Auto-Switch to Recording (Background Learning)
            // We stop the agent loop (done via isAgentRunning=false)
            // And start recording immediately.

            // Reuse startCloning logic but maybe without the 3-second delay if possible?
            // Or just call startCloning() which has the delay.
            startCloning();
        });
    }

    private void displayLoop() {
        while (stopBtn.isDisable() == false) { // while running
            try {
                Thread.sleep(500);
            } catch (InterruptedException e) {
                break;
            }

            if (currentEnv != null) {
                final String render = currentEnv.render();
                Platform.runLater(() -> renderArea.setText(render));
            }
            // Check if done?
            // We need a callback from service really.
        }
    }

    private void resetUi() {
        Platform.runLater(() -> {
            trainBtn.setDisable(false);
            stopBtn.setDisable(true);
            envComboBox.setDisable(false);
            log("Status: Ready");
        });
    }

    private void log(String msg) {
        Platform.runLater(() -> logArea.appendText(msg + "\n"));
    }

    // --- LogService Integration ---
    private final com.example.automl_prototype_1.service.LogService.LogListener logListener = this::onNewLog;

    // --- Behavioral Cloning (Learn from User) ---
    private final com.example.automl_prototype_1.service.agentic.BehavioralCloningService cloningService = new com.example.automl_prototype_1.service.agentic.BehavioralCloningService();
    private GlobalInputListener globalHook;

    // UI Elements (Injected dynamically or assumed present - we need to add them to
    // FXML really,
    // but for prototype we can add programmatically or assume they are mapped if we
    // edited FXML.
    // Wait, I can't edit FXML easily without view_file.
    // I will Assume standard buttons exist or I will create a small toolbar
    // programmatically in initialize?
    // Let's add them to the existing layout if possible.
    // The FXML has `trainBtn`, `stopBtn`. I'll add a Toolbar dynamically to
    // `renderArea`'s parent or similar?
    // Better: RlLabView.fxml likely exists.
    // For now, to suffice the user request without blocking on FXML editing (which
    // is binary/complex):
    // I will use `stopBtn` behavior extension or just add a ToggleButton if I can
    // find the parent container.
    // Actually, I can use the `envComboBox` to switch modes?
    // "Tic-Tac-Toe", "GridWorld", "NFS (Learn from User)"

    // Let's modify startTraining to handle "NFS (Learn from User)" mode.

    private void startCloning() {
        if (globalHook == null) {
            globalHook = new GlobalInputListener(cloningService);
        }

        String mode = envComboBox.getValue();
        if (mode.contains("NFS") || mode.contains("User") || mode.contains("Demonstration")) {
            log("Use ARROW KEYS to drive. Recording in 3 seconds...");
            globalHook.setProfile("ARROWS");
        } else {
            log("Use WASD to control. Recording in 3 seconds...");
            globalHook.setProfile("WASD");
        }

        new Thread(() -> {
            try {
                Thread.sleep(3000);
                Platform.runLater(() -> log("GO! Recording... (Press STOP to finish)"));
                cloningService.startRecording(mode);
                globalHook.start();
            } catch (InterruptedException e) {
            }
        }).start();

        trainBtn.setDisable(true);
        stopBtn.setDisable(false);
    }

    private void stopCloning() {
        if (globalHook != null)
            globalHook.stop();
        cloningService.stopRecording();
        log("Recording finished.");

        // Auto-Train and Run
        new Thread(() -> {
            log("Training Universal Agent...");
            cloningService.trainPolicy();
            // Start Execution Loop
            runUniversalAgent();
        }).start();

        resetUi();
    }

    // Flag to stop the agent
    private volatile boolean isAgentRunning = false;
    private final com.example.automl_prototype_1.service.agentic.InputSimService inputSim = new com.example.automl_prototype_1.service.agentic.InputSimService();
    private final com.example.automl_prototype_1.service.agentic.ScreenWatcherService screenWatcher = new com.example.automl_prototype_1.service.agentic.ScreenWatcherService();
    private com.example.automl_prototype_1.controller.GlobalInputListener safetyListener;
    // --- Window Focus Service ---
    private final com.example.automl_prototype_1.service.agentic.WindowFocusService focusService = new com.example.automl_prototype_1.service.agentic.WindowFocusService();
    private String targetAppName = "";

    // Safety Flags
    private volatile boolean isMouseControlActive = true;

    @FXML
    private TextField targetAppInput; // We need to add this to FXML or inject programmatically

    private void runUniversalAgent() {
        Platform.runLater(() -> {
            Alert confirm = new Alert(Alert.AlertType.CONFIRMATION);
            confirm.setTitle("Desktop automation");
            confirm.setHeaderText("This agent can move the mouse and type keys.");
            confirm.setContentText("Enable Robot automation and continue?");
            if (confirm.showAndWait().orElse(ButtonType.CANCEL) != ButtonType.OK) {
                log("Agent start cancelled — Robot gate OFF.");
                return;
            }

            javafx.scene.control.TextInputDialog dialog = new javafx.scene.control.TextInputDialog("Chrome");
            dialog.setTitle("Target Application");
            dialog.setHeaderText("Privacy Check (required)");
            dialog.setContentText("Enter the window title fragment to control (required, e.g. Chrome):");
            java.util.Optional<String> result = dialog.showAndWait();
            if (result.isEmpty() || result.get().isBlank()) {
                log("Agent start blocked — target app name is required.");
                return;
            }
            targetAppName = result.get().trim();
            DesktopAutomationGate.getInstance().setEnabled(true);
            isAgentRunning = true;
            isMouseControlActive = true;
            log("Agent STARTED. Target=" + targetAppName + ". Press STOP to Abort.");
            trainBtn.setDisable(true);
            stopBtn.setDisable(false);
            stopBtn.setOnAction(e -> stopAgent());
            new Thread(this::startAgentWorker, "rl-universal-agent").start();
        });
    }

    private void startAgentWorker() {
        // Start Safety Listener
        if (safetyListener == null) {
            safetyListener = new com.example.automl_prototype_1.controller.GlobalInputListener(cloningService);
        }
        safetyListener.setSafetyCallback(this::handleSafetyInterrupt);
        safetyListener.start();

        LogService.info("UniversalAgent", "Agent taking control in 3 seconds...");
        try {
            Thread.sleep(3000);
        } catch (Exception e) {
        }

        inputSim.setKeyProfile("ARROWS");

        while (isAgentRunning) {
            // 0. Privacy Check
            if (!targetAppName.isEmpty() && !focusService.isAppFocused(targetAppName)) {
                // Target App NOT in focus. Sleep and Continue.
                try {
                    Thread.sleep(500);
                } catch (Exception e) {
                }
                continue;
            }

            // 1. Capture
            double[] state = screenWatcher.captureState();

            // 2. Think
            var action = cloningService.predictAction(state);

            // 3. Act
            // Mouse (Only if Active)
            if (isMouseControlActive) {
                if (action.mouseX > 0 && action.mouseY > 0) {
                    inputSim.mouseMove(action.mouseX, action.mouseY);
                }
                if (action.mouseButton == 1)
                    inputSim.mouseLeftClick();
                if (action.mouseButton == 3)
                    inputSim.mouseRightClick();
            }

            // Keyboard (Always active unless stopped completely)
            if (action.keyAction != 0) {
                inputSim.performAction(action.keyAction);
            }

            try {
                Thread.sleep(100);
            } catch (Exception e) {
            }
        }

        if (safetyListener != null)
            safetyListener.stop();
        Platform.runLater(() -> log("Agent Stopped."));
        resetUi();
    }

    private void stopAgent() {
        isAgentRunning = false;
        DesktopAutomationGate.getInstance().setEnabled(false);
    }

    private void onNewLog(String message) {
        // Filter only RL related logs to avoid noise
        if (message.contains("[RL]") || message.contains("[QLearning]") || message.contains("BehavioralCloning")
                || message.contains("GlobalHook")) {
            Platform.runLater(() -> logArea.appendText(message + "\n"));
        }
    }

    public void cleanup() {
        if (globalHook != null)
            globalHook.stop();
        com.example.automl_prototype_1.service.LogService.getInstance().removeListener(logListener);
    }
}
