package com.example.automl_prototype_1.controller;

import com.example.automl_prototype_1.gpu.PythonExecutionService;
import com.example.automl_prototype_1.model.Domain;
import com.example.automl_prototype_1.service.AppStateService;
import com.example.automl_prototype_1.service.LogService;
import com.fasterxml.jackson.databind.ObjectMapper;
import javafx.application.Platform;
import javafx.collections.FXCollections;
import javafx.fxml.FXML;
import javafx.scene.control.*;
import javafx.stage.DirectoryChooser;

import java.io.File;
import java.util.HashMap;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.atomic.AtomicBoolean;

public class CnnWorkspaceController implements Cleanable {

    @FXML
    private Button browseFolderButton;
    @FXML
    private Label selectedFolderLabel;

    @FXML
    private CheckBox chkFlip, chkRotate, chkCrop;

    @FXML
    private ChoiceBox<String> cnnBackboneChoiceBox;
    @FXML
    private RadioButton radFeatureExtractor, radFineTune, radTrainScratch;
    private ToggleGroup tlGroup;

    @FXML
    private ChoiceBox<String> imageSizeChoiceBox;
    @FXML
    private TextField batchSizeField, epochsField;
    @FXML
    private CheckBox chkMixedPrecision, chkUseGpu;

    @FXML
    private Button runTrainingBtn, stopTrainingBtn;
    @FXML
    private ProgressBar trainingProgress;
    @FXML
    private Label trainingStatusLabel;
    @FXML
    private TextArea trainingLogArea;

    private final AppStateService appState = AppStateService.getInstance();
    private final ObjectMapper mapper = new ObjectMapper();
    private final AtomicBoolean trainingRunning = new AtomicBoolean(false);
    private volatile String activeJobId;

    @FXML
    private void initialize() {
        appState.setCurrentDomain(Domain.CNN);

        cnnBackboneChoiceBox.setItems(FXCollections.observableArrayList(
                "Custom CNN / SimpleCNN (implemented)",
                "ResNet-50 (not implemented)",
                "EfficientNet-B0 (not implemented)",
                "MobileNet-V3 (not implemented)",
                "YOLOv8 (not implemented)"));
        cnnBackboneChoiceBox.getSelectionModel().select(0);

        tlGroup = new ToggleGroup();
        radFeatureExtractor.setToggleGroup(tlGroup);
        radFineTune.setToggleGroup(tlGroup);
        radTrainScratch.setToggleGroup(tlGroup);
        radTrainScratch.setSelected(true);
        radFeatureExtractor.setDisable(true);
        radFineTune.setDisable(true);
        radFeatureExtractor.setTooltip(new Tooltip("Not implemented — SimpleCNN trains from scratch only"));
        radFineTune.setTooltip(new Tooltip("Not implemented — SimpleCNN trains from scratch only"));

        imageSizeChoiceBox.setItems(FXCollections.observableArrayList("64x64 (SimpleCNN fixed)"));
        imageSizeChoiceBox.getSelectionModel().select(0);

        if (chkMixedPrecision != null) {
            chkMixedPrecision.setSelected(false);
            chkMixedPrecision.setDisable(true);
            chkMixedPrecision.setTooltip(new Tooltip("FP16 not implemented for SimpleCNN"));
        }

        if (batchSizeField.getText() == null || batchSizeField.getText().isBlank())
            batchSizeField.setText("32");
        if (epochsField.getText() == null || epochsField.getText().isBlank())
            epochsField.setText("20");

        chkUseGpu.setSelected(appState.isUseGpu());
        chkUseGpu.selectedProperty().addListener((obs, o, n) -> appState.setUseGpu(n));

        if (appState.getSelectedDatasetFile() != null && appState.getSelectedDatasetFile().isDirectory()) {
            selectedFolderLabel.setText(appState.getSelectedDatasetFile().getAbsolutePath());
        }
        if (stopTrainingBtn != null)
            stopTrainingBtn.setDisable(true);
    }

    @FXML
    private void handleBrowseFolder() {
        DirectoryChooser dirChooser = new DirectoryChooser();
        dirChooser.setTitle("Open Image Dataset Folder");
        File defaultDir = new File(System.getProperty("user.dir"), "datasets");
        if (defaultDir.exists())
            dirChooser.setInitialDirectory(defaultDir);

        File folder = dirChooser.showDialog(browseFolderButton.getScene().getWindow());
        if (folder != null) {
            selectedFolderLabel.setText(folder.getAbsolutePath());
            appState.setSelectedDatasetFile(folder);
        }
    }

    @FXML
    private void handleRunTraining() {
        if (appState.getSelectedDatasetFile() == null || !appState.getSelectedDatasetFile().isDirectory()) {
            trainingStatusLabel.setText("No image folder selected.");
            return;
        }

        String backbone = cnnBackboneChoiceBox.getValue() != null
                ? cnnBackboneChoiceBox.getValue()
                : "Custom CNN / SimpleCNN (implemented)";
        boolean implemented = backbone.toLowerCase().contains("simple") || backbone.toLowerCase().contains("custom");
        if (!implemented) {
            Alert confirm = new Alert(Alert.AlertType.CONFIRMATION);
            confirm.setTitle("Unimplemented backbone");
            confirm.setHeaderText(backbone + " is not trained today.");
            confirm.setContentText("Continue anyway to train SimpleCNN @ 64×64 (preview mode)?");
            if (confirm.showAndWait().orElse(ButtonType.CANCEL) != ButtonType.OK) {
                return;
            }
        }

        if (!trainingRunning.compareAndSet(false, true)) {
            appendLog("Training already running.\n");
            return;
        }

        int batchSize = parseInt(batchSizeField.getText(), 32);
        int epochs = parseInt(epochsField.getText(), 20);
        String strategy = "SCRATCH";
        activeJobId = UUID.randomUUID().toString();

        runTrainingBtn.setDisable(true);
        stopTrainingBtn.setDisable(false);
        trainingProgress.setProgress(-1);
        trainingStatusLabel.setText("Training SimpleCNN @ 64×64...");
        trainingLogArea.clear();
        appendLog("requested backbone=" + backbone + " → actual=SimpleCNN@64 strategy=" + strategy
                + " flip=" + chkFlip.isSelected() + " rotate=" + chkRotate.isSelected()
                + " crop=" + chkCrop.isSelected()
                + " batch=" + batchSize + " epochs=" + epochs + "\n");
        LogService.info("CNN", "Starting CNN training: SimpleCNN (requested " + backbone + ")");

        Map<String, Object> payload = new HashMap<>();
        payload.put("action", "TRAIN_WORKSPACE");
        payload.put("job_id", activeJobId);
        payload.put("algo", "CNN");
        payload.put("backbone", backbone);
        payload.put("strategy", strategy);
        payload.put("image_size", "64x64");
        payload.put("batch_size", batchSize);
        payload.put("epochs", epochs);
        payload.put("learning_rate", 0.001);
        payload.put("use_gpu", chkUseGpu.isSelected());
        payload.put("fp16", false);
        payload.put("augment_flip", chkFlip.isSelected());
        payload.put("augment_rotate", chkRotate.isSelected());
        payload.put("augment_crop", chkCrop.isSelected());
        payload.put("dataset_path", appState.getSelectedDatasetFile().getAbsolutePath().replace("\\", "/"));
        if (!implemented) {
            payload.put("allow_preview", true);
        }

        new Thread(() -> runWorkspaceJob(payload), "cnn-workspace-train").start();
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
        }, "cnn-cancel").start();
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
