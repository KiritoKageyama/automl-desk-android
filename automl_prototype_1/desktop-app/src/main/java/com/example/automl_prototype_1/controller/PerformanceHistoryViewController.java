package com.example.automl_prototype_1.controller;

import com.example.automl_prototype_1.service.agentic.PerformanceHistoryService;
import javafx.beans.property.SimpleDoubleProperty;
import javafx.beans.property.SimpleIntegerProperty;
import javafx.beans.property.SimpleStringProperty;
import javafx.collections.FXCollections;
import javafx.collections.ObservableList;
import javafx.fxml.FXML;
import javafx.scene.control.*;

import java.util.Map;

/**
 * Controller for the Performance History view.
 * Reads algorithm success/failure stats from PerformanceHistoryService
 * and displays them in a TableView.
 */
public class PerformanceHistoryViewController {

    @FXML
    private TableView<AlgoStatsRow> historyTable;
    @FXML
    private TableColumn<AlgoStatsRow, String> algorithmCol;
    @FXML
    private TableColumn<AlgoStatsRow, Number> attemptsCol;
    @FXML
    private TableColumn<AlgoStatsRow, Number> successRateCol;
    @FXML
    private TableColumn<AlgoStatsRow, Number> failureRateCol;
    @FXML
    private Button refreshBtn;
    @FXML
    private Button clearHistoryBtn;
    @FXML
    private Label statusLabel;

    private final PerformanceHistoryService historyService = new PerformanceHistoryService();

    @FXML
    private void initialize() {
        algorithmCol.setCellValueFactory(cd -> cd.getValue().nameProperty());
        attemptsCol.setCellValueFactory(cd -> cd.getValue().attemptsProperty());
        successRateCol.setCellValueFactory(cd -> cd.getValue().successRateProperty());
        failureRateCol.setCellValueFactory(cd -> cd.getValue().failureRateProperty());

        refreshBtn.setOnAction(e -> loadData());
        clearHistoryBtn.setOnAction(e -> handleClearHistory());

        loadData();
    }

    private void loadData() {
        ObservableList<AlgoStatsRow> items = FXCollections.observableArrayList();

        Map<String, PerformanceHistoryService.AlgoStats> allHistory = historyService.getAllHistory();

        if (allHistory == null || allHistory.isEmpty()) {
            statusLabel.setText("No performance history recorded yet. Run some experiments first!");
            historyTable.setItems(items);
            return;
        }

        for (Map.Entry<String, PerformanceHistoryService.AlgoStats> entry : allHistory.entrySet()) {
            String algoName = entry.getKey();
            PerformanceHistoryService.AlgoStats stats = entry.getValue();
            int attempts = stats.attempts;
            double failureRate = historyService.getFailureRate(algoName);
            double successPercent = Math.max(0, 100.0 - (failureRate * 100.0));
            double failurePercent = failureRate * 100.0;

            items.add(new AlgoStatsRow(algoName, attempts, successPercent, failurePercent));
        }

        historyTable.setItems(items);
        statusLabel.setText("Loaded " + items.size() + " algorithm records.");
    }

    private void handleClearHistory() {
        Alert confirm = new Alert(Alert.AlertType.CONFIRMATION,
                "Clear all performance history? This cannot be undone.",
                ButtonType.YES, ButtonType.NO);
        confirm.setTitle("Confirm Clear");
        confirm.showAndWait().ifPresent(response -> {
            if (response == ButtonType.YES) {
                historyService.clearHistory();
                loadData();
                statusLabel.setText("History cleared.");
            }
        });
    }

    // --- Row Model ---

    public static class AlgoStatsRow {
        private final SimpleStringProperty name;
        private final SimpleIntegerProperty attempts;
        private final SimpleDoubleProperty successRate;
        private final SimpleDoubleProperty failureRate;

        public AlgoStatsRow(String name, int attempts, double successRate, double failureRate) {
            this.name = new SimpleStringProperty(name);
            this.attempts = new SimpleIntegerProperty(attempts);
            this.successRate = new SimpleDoubleProperty(Math.round(successRate * 10.0) / 10.0);
            this.failureRate = new SimpleDoubleProperty(Math.round(failureRate * 10.0) / 10.0);
        }

        public SimpleStringProperty nameProperty() {
            return name;
        }

        public SimpleIntegerProperty attemptsProperty() {
            return attempts;
        }

        public SimpleDoubleProperty successRateProperty() {
            return successRate;
        }

        public SimpleDoubleProperty failureRateProperty() {
            return failureRate;
        }
    }
}
