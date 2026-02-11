package com.example.automl_prototype_1.controller;

import com.example.automl_prototype_1.rl.IEnvironment;
import com.example.automl_prototype_1.rl.RlTrainingService;
import com.example.automl_prototype_1.rl.TicTacToeEnv;

import javafx.fxml.FXML;
import javafx.scene.control.Button;
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
        envComboBox.getItems().addAll("Tic-Tac-Toe", "GridWorld (Coming Soon)");
        envComboBox.getSelectionModel().selectFirst();

        // Button Actions
        trainBtn.setOnAction(e -> startTraining());
        stopBtn.setOnAction(e -> stopTraining());

        // Log redirection (simple poll for now or listener)
        // ideally LogService has a listener we can hook into.
        // For prototype, we'll just log manually or hook into a specialized logger if
        // available.
        // Or we can periodic poll? No.
        // better: let's add a simple static consumer to LogService for this view?
        // or just rely on RlTrainingService to return info?

        // For now, let's just show basic status updates
        log("RL Lab Initialized.");
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
        } else {
            log("Environment not implemented.");
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
        });
    }

    private void log(String msg) {
        Platform.runLater(() -> logArea.appendText(msg + "\n"));
    }
}
