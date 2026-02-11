package com.example.automl_prototype_1.controller;

import com.example.automl_prototype_1.model.progress.OrchestratorState;
import com.example.automl_prototype_1.service.progress.ProgressTrackingService;

import javafx.application.Platform;
import javafx.concurrent.Task;
import javafx.fxml.FXML;
import javafx.scene.control.*;

public class ProgressViewController implements Cleanable {

    @FXML
    private Label mainStatusLabel, globalBestAccuracyLabel, globalBestAlgoLabel, currentAlgoLabel, etaLabel;
    @FXML
    private ProgressBar trialProgressBar, iterationProgressBar;
    @FXML
    private ProgressIndicator mainProgressIndicator;
    @FXML
    private TextArea logConsole;

    private javafx.animation.Timeline timeline;
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
    private TextField logSearchField;

    @FXML
    public void initialize() {
        // Connect to LogService
        com.example.automl_prototype_1.service.LogService.getInstance().addListener(logListener);

        // Filter Listener
        if (logSearchField != null) {
            logSearchField.textProperty().addListener((obs, oldVal, newVal) -> refreshLogs());
        }

        // This poller is now ONLY for the detailed card data
        startDetailPolling();
    }

    // --- Log Console Logic ---
    private final StringBuilder fullLogHistory = new StringBuilder(); // Store everything
    private final StringBuilder pendingLogs = new StringBuilder();
    private long lastLogUpdate = 0;

    private void appendLog(String message) {
        synchronized (pendingLogs) {
            pendingLogs.append(message).append("\n");
        }
        synchronized (fullLogHistory) {
            fullLogHistory.append(message).append("\n");
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
                    // Only append if matches search (optimization: if empty search, append all)
                    String filter = (logSearchField != null) ? logSearchField.getText().toLowerCase() : "";

                    if (filter.isEmpty()) {
                        logConsole.appendText(pendingLogs.toString());
                    } else {
                        // Must re-filter everything if we really want consistency,
                        // but for "Efficiency" actively appending just the new matching lines is okay
                        // However, simplest "Hacker Mode" is: if filter exists, we might need to
                        // re-render all from history
                        // actually, let's just refreshLogs() if filter is active, otherwise append.
                        refreshLogs();
                    }
                    pendingLogs.setLength(0); // Clear buffer
                }
            }
        });
    }

    private void refreshLogs() {
        String filter = (logSearchField != null) ? logSearchField.getText().toLowerCase() : "";

        if (filter.isEmpty()) {
            // If full history is huge, this might lag. But for <1MB logs it's fine.
            synchronized (fullLogHistory) {
                logConsole.setText(fullLogHistory.toString());
                logConsole.setScrollTop(Double.MAX_VALUE);
            }
        } else {
            StringBuilder filtered = new StringBuilder();
            synchronized (fullLogHistory) {
                String[] lines = fullLogHistory.toString().split("\\n");
                for (String line : lines) {
                    if (line.toLowerCase().contains(filter)) {
                        filtered.append(line).append("\n");
                    }
                }
            }
            logConsole.setText(filtered.toString());
            logConsole.setScrollTop(Double.MAX_VALUE);
        }
    }

    @FXML
    private void onCopyLogsClicked() {
        final javafx.scene.input.Clipboard clipboard = javafx.scene.input.Clipboard.getSystemClipboard();
        final javafx.scene.input.ClipboardContent content = new javafx.scene.input.ClipboardContent();
        synchronized (fullLogHistory) {
            content.putString(fullLogHistory.toString());
        }
        clipboard.setContent(content);
        // Visual feedback?
        com.example.automl_prototype_1.service.LogService.info("UI", "Logs copied to clipboard.");
    }

    @FXML
    private void onGarbageCollectClicked() {
        com.example.automl_prototype_1.service.TelemetryService.getInstance().triggerMemoryCleanup();
        // Clear local history too to match backend
        synchronized (fullLogHistory) {
            fullLogHistory.setLength(0);
        }
        logConsole.clear();
        com.example.automl_prototype_1.service.LogService.info("UI", "Local Log View Cleared.");
    }

    private void startDetailPolling() {
        if (timeline != null) {
            timeline.stop();
        }
        timeline = new javafx.animation.Timeline(
                new javafx.animation.KeyFrame(javafx.util.Duration.millis(500), event -> updateDetails()));
        timeline.setCycleCount(javafx.animation.Animation.INDEFINITE);
        timeline.play();
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
        System.out.println("ProgressViewController: Stopping timeline.");
        // Detach listener to prevent memory leaks/zombie updates
        com.example.automl_prototype_1.service.LogService.getInstance().removeListener(logListener);

        if (timeline != null) {
            timeline.stop();
        }
    }
}