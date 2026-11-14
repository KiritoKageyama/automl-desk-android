package com.example.automl_prototype_1.controller;

import com.example.automl_prototype_1.dataprovider.CsvDataProvider;
import com.example.automl_prototype_1.gpu.PythonExecutionService;
import com.example.automl_prototype_1.model.Dataset;
import com.example.automl_prototype_1.model.Domain;
import com.example.automl_prototype_1.service.AppStateService;
import com.example.automl_prototype_1.service.LogService;
import com.fasterxml.jackson.databind.ObjectMapper;
import javafx.application.Platform;
import javafx.collections.FXCollections;
import javafx.fxml.FXML;
import javafx.scene.control.*;
import javafx.stage.FileChooser;

import java.io.File;
import java.io.FileReader;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.atomic.AtomicBoolean;

public class DlWorkspaceController implements Cleanable {

    @FXML
    private Button browseButton;
    @FXML
    private Label selectedFileLabel;
    @FXML
    private ChoiceBox<String> targetColumnChoiceBox;

    @FXML
    private ChoiceBox<String> topologyChoiceBox;
    @FXML
    private TextField layersField;
    @FXML
    private ChoiceBox<String> activationChoiceBox;

    @FXML
    private ChoiceBox<String> optimizerChoiceBox;
    @FXML
    private TextField lrField;
    @FXML
    private TextField epochsField;
    @FXML
    private CheckBox chkUseGpu, chkUseWsl;

    @FXML
    private Button runTrainingBtn, stopTrainingBtn;
    @FXML
    private ProgressBar trainingProgress;
    @FXML
    private Label trainingStatusLabel;
    @FXML
    private TextArea trainingLogArea;

    private final AppStateService appState = AppStateService.getInstance();
    private final CsvDataProvider csvDataProvider = new CsvDataProvider();
    private final ObjectMapper mapper = new ObjectMapper();
    private final AtomicBoolean trainingRunning = new AtomicBoolean(false);
    private volatile String activeJobId;

    @FXML
    private void initialize() {
        appState.setCurrentDomain(Domain.DL);

        topologyChoiceBox.setItems(FXCollections.observableArrayList("Multi-Layer Perceptron (MLP)",
                "Recurrent NN (LSTM)", "Autoencoder (preview→MLP)"));
        topologyChoiceBox.getSelectionModel().select(0);

        activationChoiceBox.setItems(FXCollections.observableArrayList("ReLU", "GELU", "Tanh", "Sigmoid"));
        activationChoiceBox.getSelectionModel().select(0);

        optimizerChoiceBox.setItems(FXCollections.observableArrayList("Adam", "SGDW", "RMSprop", "AdamW"));
        optimizerChoiceBox.getSelectionModel().select(0);

        if (layersField.getText() == null || layersField.getText().isBlank()) {
            layersField.setText("128,64");
        }
        if (lrField.getText() == null || lrField.getText().isBlank()) {
            lrField.setText("0.001");
        }
        if (epochsField.getText() == null || epochsField.getText().isBlank()) {
            epochsField.setText("50");
        }

        chkUseGpu.setSelected(appState.isUseGpu());
        chkUseGpu.selectedProperty().addListener((obs, o, n) -> appState.setUseGpu(n));
        chkUseWsl.setSelected(appState.isUseWsl());
        chkUseWsl.selectedProperty().addListener((obs, o, n) -> appState.setUseWsl(n));

        targetColumnChoiceBox.getSelectionModel().selectedItemProperty().addListener((obs, o, n) -> {
            if (n != null)
                appState.setTargetColumnName(n);
        });

        if (appState.getSelectedDatasetFile() != null) {
            selectedFileLabel.setText(appState.getSelectedDatasetFile().getName());
            loadHeadersToChoiceBox(appState.getSelectedDatasetFile(), appState.getTargetColumnName());
        }

        if (stopTrainingBtn != null)
            stopTrainingBtn.setDisable(true);
    }

    @FXML
    private void handleBrowseData() {
        FileChooser fileChooser = new FileChooser();
        fileChooser.getExtensionFilters().add(new FileChooser.ExtensionFilter("CSV Files", "*.csv"));
        File file = fileChooser.showOpenDialog(browseButton.getScene().getWindow());

        if (file != null) {
            selectedFileLabel.setText(file.getName());
            appState.setSelectedDatasetFile(file);
            loadHeadersToChoiceBox(file, null);
        }
    }

    private void loadHeadersToChoiceBox(File file, String preselect) {
        try (FileReader reader = new FileReader(file)) {
            Dataset ds = csvDataProvider.loadDataset(reader);
            List<String> headers = ds.getHeaders();
            appState.setDatasetHeaders(headers);
            targetColumnChoiceBox.setItems(FXCollections.observableArrayList(headers));
            targetColumnChoiceBox.setDisable(false);
            if (preselect != null && headers.contains(preselect)) {
                targetColumnChoiceBox.getSelectionModel().select(preselect);
            } else if (!headers.isEmpty()) {
                targetColumnChoiceBox.getSelectionModel().select(headers.size() - 1);
            }
        } catch (Exception e) {
            appendLog("Failed to load dataset: " + e.getMessage() + "\n");
        }
    }

    @FXML
    private void handleRunTraining() {
        if (appState.getSelectedDatasetFile() == null) {
            trainingStatusLabel.setText("No dataset selected.");
            return;
        }
        if (appState.getTargetColumnName() == null) {
            trainingStatusLabel.setText("No target column selected.");
            return;
        }
        if (!trainingRunning.compareAndSet(false, true)) {
            appendLog("Training already running.\n");
            return;
        }

        String topology = topologyChoiceBox.getValue();
        if (topology == null)
            topology = "Multi-Layer Perceptron (MLP)";
        String algoKey = topology.contains("LSTM") ? "LSTM" : topology.contains("Auto") ? "AUTOENCODER" : "MLP";

        String layers = layersField.getText() == null || layersField.getText().isBlank() ? "128,64"
                : layersField.getText().trim();
        String activation = activationChoiceBox.getValue() != null ? activationChoiceBox.getValue() : "ReLU";
        String optimizer = optimizerChoiceBox.getValue() != null ? optimizerChoiceBox.getValue() : "Adam";
        double lr = parseDouble(lrField.getText(), 0.001);
        int epochs = parseInt(epochsField.getText(), 50);
        activeJobId = UUID.randomUUID().toString();

        runTrainingBtn.setDisable(true);
        stopTrainingBtn.setDisable(false);
        trainingProgress.setProgress(-1);
        trainingStatusLabel.setText("Training " + algoKey + "...");
        trainingLogArea.clear();
        appendLog("Starting " + algoKey + " layers=" + layers + " act=" + activation + " opt=" + optimizer
                + " lr=" + lr + " epochs=" + epochs + "\n");
        if ("AUTOENCODER".equals(algoKey)) {
            appendLog("NOTE: Autoencoder uses preview mode (trains MLP). Real autoencoders are not implemented.\n");
        }
        LogService.info("DL", "Starting DL training: " + algoKey);

        Map<String, Object> payload = new HashMap<>();
        payload.put("action", "TRAIN_WORKSPACE");
        payload.put("job_id", activeJobId);
        payload.put("algo", algoKey);
        if ("AUTOENCODER".equals(algoKey)) {
            payload.put("allow_preview", true);
        }
        payload.put("layers", layers);
        payload.put("activation", activation);
        payload.put("optimizer", optimizer);
        payload.put("learning_rate", lr);
        payload.put("epochs", epochs);
        payload.put("use_gpu", chkUseGpu.isSelected());
        payload.put("use_wsl", chkUseWsl.isSelected());
        payload.put("target_column", appState.getTargetColumnName());
        payload.put("dataset_path", appState.getSelectedDatasetFile().getAbsolutePath().replace("\\", "/"));

        new Thread(() -> runWorkspaceJob(payload), "dl-workspace-train").start();
    }

    @FXML
    private void handleStopTraining() {
        if (!trainingRunning.get())
            return;
        trainingStatusLabel.setText("Stopping...");
        appendLog("Sending CANCEL_TRAINING...\n");
        new Thread(() -> {
            try {
                Map<String, Object> cancel = new HashMap<>();
                cancel.put("action", "CANCEL_TRAINING");
                cancel.put("job_id", activeJobId);
                PythonExecutionService.getInstance().postRaw(mapper.writeValueAsString(cancel));
            } catch (Exception e) {
                Platform.runLater(() -> appendLog("Cancel request failed: " + e.getMessage() + "\n"));
            }
        }, "dl-cancel").start();
    }

    private void runWorkspaceJob(Map<String, Object> payload) {
        try {
            PythonExecutionService py = PythonExecutionService.getInstance();
            if (!py.isServerRunning()) {
                Platform.runLater(() -> appendLog("Starting GPU server...\n"));
                py.startGpuServer();
            }
            String result = py.postRaw(mapper.writeValueAsString(payload));
            @SuppressWarnings("unchecked")
            Map<String, Object> resp = mapper.readValue(result, Map.class);
            Platform.runLater(() -> {
                if (Boolean.TRUE.equals(resp.get("cancelled"))) {
                    trainingStatusLabel.setText("Stopped by user.");
                    trainingProgress.setProgress(0);
                } else if (Boolean.TRUE.equals(resp.get("success"))) {
                    trainingProgress.setProgress(1.0);
                    trainingStatusLabel.setText("Training complete.");
                } else {
                    trainingStatusLabel.setText("Error: " + resp.getOrDefault("error", "unknown"));
                    trainingProgress.setProgress(0);
                }
                appendLog("[Server]\n" + result + "\n");
                resetTrainingUi();
            });
        } catch (Exception e) {
            Platform.runLater(() -> {
                trainingStatusLabel.setText("Error: " + e.getMessage());
                appendLog("ERROR: " + e.getMessage() + "\n");
                resetTrainingUi();
            });
        }
    }

    private void resetTrainingUi() {
        trainingRunning.set(false);
        runTrainingBtn.setDisable(false);
        stopTrainingBtn.setDisable(true);
    }

    private void appendLog(String msg) {
        if (trainingLogArea != null)
            trainingLogArea.appendText(msg);
    }

    private static int parseInt(String s, int def) {
        try {
            return Integer.parseInt(s.trim());
        } catch (Exception e) {
            return def;
        }
    }

    private static double parseDouble(String s, double def) {
        try {
            return Double.parseDouble(s.trim());
        } catch (Exception e) {
            return def;
        }
    }

    @Override
    public void cleanup() {
        if (trainingRunning.get()) {
            try {
                Map<String, Object> cancel = new HashMap<>();
                cancel.put("action", "CANCEL_TRAINING");
                PythonExecutionService.getInstance().postRaw(mapper.writeValueAsString(cancel));
            } catch (Exception ignored) {
            }
        }
        trainingRunning.set(false);
    }
}
