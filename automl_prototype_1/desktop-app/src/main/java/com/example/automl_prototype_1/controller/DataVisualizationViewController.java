package com.example.automl_prototype_1.controller;

import com.example.automl_prototype_1.service.AppStateService;
import com.example.automl_prototype_1.service.LogService;
import com.example.automl_prototype_1.service.RealTimeDataService;
import com.example.automl_prototype_1.model.Dataset;
import javafx.fxml.FXML;
import javafx.scene.chart.LineChart;
import javafx.scene.chart.NumberAxis;
import javafx.scene.chart.ScatterChart;
import javafx.scene.chart.XYChart;
import javafx.scene.control.Label;
import javafx.scene.control.Tab;
import javafx.scene.control.TabPane;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import javafx.scene.layout.TilePane;
import javafx.stage.FileChooser;
import javafx.application.Platform;
import java.io.File;
import java.util.List;

public class DataVisualizationViewController {

    @FXML
    private TabPane vizTabPane;
    @FXML
    private Tab numericTab;
    @FXML
    private Tab imageTab;
    @FXML
    private Tab timeSeriesTab;
    @FXML
    private Tab realTimeTab;

    // Numeric
    @FXML
    private ScatterChart<Number, Number> scatterChart;
    @FXML
    private NumberAxis scatterX;
    @FXML
    private NumberAxis scatterY;

    // Images
    @FXML
    private TilePane imageGallery;

    // Time Series
    @FXML
    private LineChart<Number, Number> timeSeriesChart;

    // Real-Time
    @FXML
    private LineChart<Number, Number> realTimeChart;
    @FXML
    private Label rtCounterLabel;

    private final AppStateService appState = AppStateService.getInstance();
    private final RealTimeDataService rtService = RealTimeDataService.getInstance();

    @FXML
    public void initialize() {
        // Load Buffer Size Listener for Real-Time
        rtService.bufferSizeProperty().addListener((obs, old, val) -> {
            Platform.runLater(() -> {
                rtCounterLabel.setText("Buffer: " + val + " records");
                updateRealTimeChart();
            });
        });

        // Initial Load
        loadCurrentDataset();
    }

    public void loadCurrentDataset() {
        Dataset ds = appState.getCurrentDataset();
        if (ds == null) {
            LogService.warn("Viz", "No dataset loaded.");
            return;
        }

        switch (ds.getDataType()) {
            case IMAGE:
                vizTabPane.getSelectionModel().select(imageTab);
                loadImages(ds);
                break;
            case TIME_SERIES:
                vizTabPane.getSelectionModel().select(timeSeriesTab);
                loadTimeSeries(ds);
                break;
            default: // TABULAR / TEXT
                vizTabPane.getSelectionModel().select(numericTab);
                loadScatter(ds);
                break;
        }
    }

    private void loadScatter(Dataset ds) {
        scatterChart.getData().clear();
        double[][] X = ds.getFeatures();
        // int[] y = ds.getLabels(); // Unused for basic scatter logic currently

        if (X.length == 0)
            return;

        // Simple Plot: Feature 0 vs Feature 1
        XYChart.Series<Number, Number> series = new XYChart.Series<>();
        series.setName("Data Distribution");

        int limit = Math.min(500, X.length); // Limit points
        for (int i = 0; i < limit; i++) {
            double xVal = X[i].length > 0 ? X[i][0] : 0;
            double yVal = X[i].length > 1 ? X[i][1] : 0;
            series.getData().add(new XYChart.Data<>(xVal, yVal));
        }
        scatterChart.getData().add(series);
    }

    private void loadImages(Dataset ds) {
        imageGallery.getChildren().clear();
        List<String> paths = ds.getFilePaths();
        if (paths == null)
            return;

        int limit = Math.min(50, paths.size());
        for (int i = 0; i < limit; i++) {
            File f = new File(paths.get(i));
            if (f.exists()) {
                Image img = new Image(f.toURI().toString(), 100, 100, true, true);
                ImageView iv = new ImageView(img);
                iv.setFitWidth(100);
                iv.setFitHeight(100);
                imageGallery.getChildren().add(iv);
            }
        }
    }

    private void loadTimeSeries(Dataset ds) {
        timeSeriesChart.getData().clear();
        double[][] X = ds.getFeatures();
        if (X.length == 0)
            return;

        // Plot first 5 samples
        int limit = Math.min(5, X.length);
        for (int i = 0; i < limit; i++) {
            XYChart.Series<Number, Number> series = new XYChart.Series<>();
            series.setName("Sample " + i);
            for (int t = 0; t < X[i].length; t++) {
                series.getData().add(new XYChart.Data<>(t, X[i][t]));
            }
            timeSeriesChart.getData().add(series);
        }
    }

    private void updateRealTimeChart() {
        realTimeChart.getData().clear();
        java.util.List<com.example.automl_prototype_1.service.RealTimeDataService.DataPoint> buffer = rtService
                .getSnapshot();
        if (buffer.isEmpty())
            return;

        // Plot last 100 points of Feature 0
        XYChart.Series<Number, Number> series = new XYChart.Series<>();
        series.setName("Live Input (Feat 0)");

        int start = Math.max(0, buffer.size() - 50);
        for (int i = start; i < buffer.size(); i++) {
            double val = buffer.get(i).values()[0]; // Feature 0
            series.getData().add(new XYChart.Data<>(i, val));
        }
        realTimeChart.getData().add(series);
    }

    @FXML
    private void handleSaveStream() {
        FileChooser fc = new FileChooser();
        fc.setTitle("Save Stream Buffer");
        fc.setInitialFileName("stream_capture.csv");
        File f = fc.showSaveDialog(vizTabPane.getScene().getWindow());
        if (f != null) {
            rtService.saveBufferToCsv(f);
        }
    }

    @FXML
    private void handleClearStream() {
        rtService.clearBuffer();
    }

    @FXML
    private void handleSimulateData() {
        // Debug helper to push random data
        double[] pt = new double[] { Math.random(), Math.random() };
        rtService.pushDataPoint(pt);
    }
}
