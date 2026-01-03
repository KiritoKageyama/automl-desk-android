package com.example.automl_prototype_1.controller;

import com.example.automl_prototype_1.model.progress.OrchestratorState;
import com.example.automl_prototype_1.service.ProgressTrackingService;

import javafx.application.Platform;
import javafx.concurrent.Task;
import javafx.fxml.FXML;
import javafx.scene.control.*;
import java.util.concurrent.*;

public class ProgressViewController implements Cleanable {

    @FXML
    private Label mainStatusLabel, globalBestAccuracyLabel, globalBestAlgoLabel, currentAlgoLabel, etaLabel;
    @FXML
    private ProgressBar trialProgressBar, iterationProgressBar;
    @FXML
    private ProgressIndicator mainProgressIndicator;
    @FXML
    private TextArea logConsole;

    private ScheduledExecutorService detailPoller;
    private final ProgressTrackingService progressService = ProgressTrackingService.getInstance();
    private final com.example.automl_prototype_1.service.LogService.LogListener logListener = this::appendLog;

    public ProgressViewController() {
        // No-arg constructor
    }

    /**
     * This method allows the DashboardController to connect the UI directly to the
     * background task.
     */
    public void bindToTask(Task<?> task) {
        // Bind the main status label and spinner directly to the task's properties
        mainStatusLabel.textProperty().bind(task.messageProperty());
        mainProgressIndicator.progressProperty().bind(task.progressProperty());
    }

    @FXML
    public void initialize() {
        // Connect to LogService
        com.example.automl_prototype_1.service.LogService.getInstance().setListener(logListener);

        // This poller is now ONLY for the detailed card data
        startDetailPolling();
    }

    // --- Log Console Logic ---
    private final StringBuilder pendingLogs = new StringBuilder();
    private long lastLogUpdate = 0;

    private void appendLog(String message) {
        synchronized (pendingLogs) {
            pendingLogs.append(message).append("\n");
        }

        // Throttled UI Update (Max once per 100ms)
        long now = System.currentTimeMillis();
        if (now - lastLogUpdate > 100) {
            flushLogs();
            lastLogUpdate = now;
        }
    }

    private void flushLogs() {
        Platform.runLater(() -> {
            synchronized (pendingLogs) {
                if (pendingLogs.length() > 0) {
                    logConsole.appendText(pendingLogs.toString());
                    pendingLogs.setLength(0); // Clear buffer
                }
            }
        });
    }

    private void startDetailPolling() {
        detailPoller = Executors.newSingleThreadScheduledExecutor();
        detailPoller.scheduleAtFixedRate(this::updateDetails, 0, 500, TimeUnit.MILLISECONDS);
    }

    // ... existing updateDetails ...

    private void updateDetails() {
        OrchestratorState currentState = progressService.snapshot();
        Platform.runLater(() -> {
            // Flush any remaining logs periodically
            synchronized (pendingLogs) {
                if (pendingLogs.length() > 0) {
                    logConsole.appendText(pendingLogs.toString());
                    pendingLogs.setLength(0);
                }
            }

            if (currentState.currentRun() != null) {
                var current = currentState.currentRun();
                currentAlgoLabel.setText(String.format("%s (Trial %d/%d)",
                        current.algoId(), current.currentTrial(), current.totalTrials()));

                trialProgressBar.setProgress(
                        current.totalTrials() > 0 ? (double) current.currentTrial() / current.totalTrials() : 0.0);
                iterationProgressBar.setProgress(
                        current.totalIters() > 0 ? (double) current.currentIter() / current.totalIters() : 0.0);

                long etaSeconds = current.etaMs() / 1000;
                etaLabel.setText(etaSeconds >= 0
                        ? String.format("%02d:%02d:%02d", etaSeconds / 3600, (etaSeconds % 3600) / 60, etaSeconds % 60)
                        : "Calculating...");
            } else {
                currentAlgoLabel.setText("-");
                trialProgressBar.setProgress(0);
                iterationProgressBar.setProgress(0);
                etaLabel.setText("Queued...");
            }

            if (currentState.globalBestScore() > Double.NEGATIVE_INFINITY) {
                globalBestAccuracyLabel.setText(String.format("%.4f", currentState.globalBestScore()));
                globalBestAlgoLabel.setText(String.format("%s (Trial %d, Iteration %d)",
                        currentState.globalBestAlgo(), currentState.globalBestTrial(), currentState.globalBestIter()));
            } else {
                globalBestAccuracyLabel.setText("N/A");
                globalBestAlgoLabel.setText("N/A");
            }
        });
    }

    @Override
    public void cleanup() {
        System.out.println("ProgressViewController: Shutting down detail poller.");
        // Detach listener to prevent memory leaks/zombie updates
        com.example.automl_prototype_1.service.LogService.getInstance().setListener(null);

        if (detailPoller != null && !detailPoller.isShutdown()) {
            detailPoller.shutdownNow();
        }
    }
}