package com.example.automl_prototype_1.controller;

import com.example.automl_prototype_1.gpu.PythonExecutionService;
import com.example.automl_prototype_1.model.Domain;
import com.example.automl_prototype_1.service.AppStateService;
import com.example.automl_prototype_1.service.LogService;
import com.example.automl_prototype_1.service.OllamaService;
import com.fasterxml.jackson.databind.ObjectMapper;
import javafx.application.Platform;
import javafx.collections.FXCollections;
import javafx.fxml.FXML;
import javafx.scene.control.*;
import javafx.stage.FileChooser;

import java.io.File;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.atomic.AtomicBoolean;

public class LlmWorkspaceController implements Cleanable {

    @FXML
    private Button browseButton;
    @FXML
    private Label selectedFileLabel;

    @FXML
    private ChoiceBox<String> llmModelChoiceBox;
    @FXML
    private TextField hfModelIdField;
    @FXML
    private Button downloadHfBtn;

    @FXML
    private ChoiceBox<String> tuningModeChoiceBox;
    @FXML
    private CheckBox chkQuantize8Bit;
    @FXML
    private CheckBox chkQuantize4Bit;

    @FXML
    private TextField chatInputField;
    @FXML
    private Button chatSendBtn;
    @FXML
    private TextArea chatHistoryArea;

    @FXML
    private Button fineTuneBtn, stopFineTuneBtn;
    @FXML
    private ProgressBar fineTuneProgress;
    @FXML
    private Label fineTuneStatusLabel;
    @FXML
    private TextArea fineTuneLogArea;

    private final AppStateService appState = AppStateService.getInstance();
    private final OllamaService ollamaService = OllamaService.getInstance();
    private final ObjectMapper mapper = new ObjectMapper();
    private final AtomicBoolean fineTuneRunning = new AtomicBoolean(false);
    private volatile String activeJobId;

    @FXML
    private void initialize() {
        appState.setCurrentDomain(Domain.LLM);

        List<String> builtInLlms = List.of("deepseek-r1:8b", "llama2", "mistral", "phi3", "Custom HF Model");
        llmModelChoiceBox.setItems(FXCollections.observableArrayList(builtInLlms));
        llmModelChoiceBox.getSelectionModel().select(appState.getOllamaModelName());

        llmModelChoiceBox.getSelectionModel().selectedItemProperty().addListener((obs, oldV, newV) -> {
            boolean isHf = "Custom HF Model".equals(newV);
            hfModelIdField.setDisable(!isHf);
            downloadHfBtn.setDisable(!isHf);
            if (!isHf && newV != null) {
                appState.setOllamaModelName(newV);
            }
        });

        tuningModeChoiceBox.setItems(
                FXCollections.observableArrayList(
                        "Full Fine-Tuning (BERT classifier)",
                        "Zero-Shot (No Tuning)"));
        tuningModeChoiceBox.getSelectionModel().select(0);

        if (chkQuantize8Bit != null) {
            chkQuantize8Bit.setSelected(false);
            chkQuantize8Bit.setDisable(true);
            chkQuantize8Bit.setTooltip(new Tooltip("Not implemented — removed from active path"));
        }
        if (chkQuantize4Bit != null) {
            chkQuantize4Bit.setSelected(false);
            chkQuantize4Bit.setDisable(true);
            chkQuantize4Bit.setTooltip(new Tooltip("Not implemented — removed from active path"));
        }

        if (chatHistoryArea != null && (chatHistoryArea.getText() == null || chatHistoryArea.getText().isBlank())) {
            chatHistoryArea.setText("[Ollama chat — independent of Fine-Tune. Fine-tune builds a BERT classifier, not this chat model.]\n");
        }

        if (stopFineTuneBtn != null)
            stopFineTuneBtn.setDisable(true);
    }

    @FXML
    private void handleBrowseData() {
        FileChooser fileChooser = new FileChooser();
        fileChooser.setTitle("Open Text Corpus");
        File defaultDir = new File(System.getProperty("user.dir"), "datasets");
        if (defaultDir.exists())
            fileChooser.setInitialDirectory(defaultDir);

        fileChooser.getExtensionFilters()
                .add(new FileChooser.ExtensionFilter("Text/JSONL Files", "*.txt", "*.md", "*.jsonl", "*.pdf"));
        File file = fileChooser.showOpenDialog(browseButton.getScene().getWindow());

        if (file != null) {
            selectedFileLabel.setText(file.getName());
            appState.setSelectedDatasetFile(file);
        }
    }

    @FXML
    private void handleLoadHfModel() {
        String hfId = hfModelIdField.getText();
        if (hfId == null || hfId.isBlank())
            return;

        LogService.info("LLMWorkspace", "Registering HF model id: " + hfId);
        appState.registerDynamicModel("CustomLLM", hfId);
        llmModelChoiceBox.getSelectionModel().select("Custom HF Model");

        Alert alert = new Alert(Alert.AlertType.INFORMATION);
        alert.setTitle("HuggingFace Integration");
        alert.setHeaderText("Model registered");
        alert.setContentText("'" + hfId + "' will be used on the next fine-tune / train job.");
        alert.showAndWait();
    }

    @FXML
    private void handleChatSubmit() {
        String input = chatInputField.getText();
        if (input == null || input.isBlank())
            return;

        chatHistoryArea.appendText("User: " + input + "\n");
        chatInputField.clear();
        chatSendBtn.setDisable(true);
        chatInputField.setDisable(true);

        String activeModel = appState.getOllamaModelName();

        new Thread(() -> {
            String response = ollamaService.queryOllama(input, "assistant");
            Platform.runLater(() -> {
                chatHistoryArea.appendText(activeModel + ": " + response + "\n\n");
                chatSendBtn.setDisable(false);
                chatInputField.setDisable(false);
                chatInputField.requestFocus();
            });
        }).start();
    }

    @FXML
    private void handleFineTune() {
        String selected = llmModelChoiceBox.getValue();
        String model = appState.getOllamaModelName();
        String hfId = hfModelIdField.getText() == null ? "" : hfModelIdField.getText().trim();

        if ("Custom HF Model".equals(selected)) {
            if (hfId.isBlank()) {
                fineTuneStatusLabel.setText("Enter a HuggingFace model id first.");
                return;
            }
            model = hfId;
        } else if (model == null || model.isBlank()) {
            model = selected;
        }
        if (model == null) {
            fineTuneStatusLabel.setText("No model selected.");
            return;
        }

        String mode = tuningModeChoiceBox.getValue() != null
                ? tuningModeChoiceBox.getValue()
                : "Full Fine-Tuning (BERT classifier)";
        boolean q8 = false;
        boolean q4 = false;
        String dataPath = appState.getSelectedDatasetFile() != null
                ? appState.getSelectedDatasetFile().getAbsolutePath().replace("\\", "/")
                : null;

        if (mode.toLowerCase().contains("zero-shot")) {
            fineTuneStatusLabel.setText("Zero-shot selected — use Chat (Ollama) instead of Fine-Tune.");
            return;
        }
        if (dataPath == null) {
            fineTuneStatusLabel.setText("Upload a training corpus first.");
            return;
        }

        boolean jsonl = dataPath.toLowerCase().endsWith(".jsonl");
        if (!jsonl) {
            Alert confirm = new Alert(Alert.AlertType.CONFIRMATION);
            confirm.setTitle("Labeled corpus required");
            confirm.setHeaderText("File is not JSONL with labels.");
            confirm.setContentText(
                    "Production fine-tune needs JSONL lines like {\"text\":\"...\",\"label\":0}.\n"
                            + "Continue in preview mode with synthetic labels? (not for real accuracy)");
            if (confirm.showAndWait().orElse(ButtonType.CANCEL) != ButtonType.OK) {
                return;
            }
        }

        if (!fineTuneRunning.compareAndSet(false, true)) {
            fineTuneLogArea.appendText("Fine-tune already running.\n");
            return;
        }

        activeJobId = UUID.randomUUID().toString();
        fineTuneBtn.setDisable(true);
        stopFineTuneBtn.setDisable(false);
        fineTuneProgress.setProgress(-1);
        fineTuneStatusLabel.setText("Fine-tuning BERT classifier...");
        fineTuneLogArea.clear();
        fineTuneLogArea.appendText("model=" + model + " mode=" + mode
                + " (chat remains Ollama — not this model)\n");
        if (!jsonl) {
            fineTuneLogArea.appendText("PREVIEW: synthetic labels enabled.\n");
        }
        LogService.info("LLM", "Starting fine-tune: model=" + model + " mode=" + mode);

        Map<String, Object> payload = new HashMap<>();
        payload.put("action", "TRAIN_WORKSPACE");
        payload.put("job_id", activeJobId);
        payload.put("algo", "LLM_FINETUNE");
        payload.put("model", model);
        payload.put("hf_model_id", hfId.isBlank() ? model : hfId);
        payload.put("tuning_mode", "Full Fine-Tuning");
        payload.put("quantize_8bit", q8);
        payload.put("quantize_4bit", q4);
        payload.put("dataset_path", dataPath);
        payload.put("epochs", 3);
        payload.put("batch_size", 8);
        payload.put("learning_rate", 2e-5);
        if (!jsonl) {
            payload.put("allow_preview", true);
        }

        new Thread(() -> runFineTuneJob(payload), "llm-finetune").start();
    }

    @FXML
    private void handleStopFineTune() {
        if (!fineTuneRunning.get())
            return;
        fineTuneStatusLabel.setText("Stopping...");
        fineTuneLogArea.appendText("Sending CANCEL_TRAINING...\n");
        new Thread(() -> {
            try {
                Map<String, Object> cancel = new HashMap<>();
                cancel.put("action", "CANCEL_TRAINING");
                cancel.put("job_id", activeJobId);
                PythonExecutionService.getInstance().postRaw(mapper.writeValueAsString(cancel));
            } catch (Exception e) {
                Platform.runLater(() -> fineTuneLogArea.appendText("Cancel failed: " + e.getMessage() + "\n"));
            }
        }, "llm-cancel").start();
    }

    private void runFineTuneJob(Map<String, Object> payload) {
        try {
            PythonExecutionService py = PythonExecutionService.getInstance();
            if (!py.isServerRunning()) {
                Platform.runLater(() -> fineTuneLogArea.appendText("Starting GPU server...\n"));
                py.startGpuServer();
            }
            String result = py.postRaw(mapper.writeValueAsString(payload));
            @SuppressWarnings("unchecked")
            Map<String, Object> resp = mapper.readValue(result, Map.class);
            Platform.runLater(() -> {
                if (Boolean.TRUE.equals(resp.get("cancelled"))) {
                    fineTuneStatusLabel.setText("Stopped by user.");
                    fineTuneProgress.setProgress(0);
                } else if (Boolean.TRUE.equals(resp.get("success"))) {
                    fineTuneProgress.setProgress(1.0);
                    fineTuneStatusLabel.setText("Fine-tuning complete.");
                } else {
                    fineTuneStatusLabel.setText("Error: " + resp.getOrDefault("error", "unknown"));
                    fineTuneProgress.setProgress(0);
                }
                fineTuneLogArea.appendText("[Server]\n" + result + "\n");
                resetFineTuneUi();
            });
        } catch (Exception e) {
            Platform.runLater(() -> {
                fineTuneStatusLabel.setText("Error: " + e.getMessage());
                fineTuneLogArea.appendText("ERROR: " + e.getMessage() + "\n");
                resetFineTuneUi();
            });
        }
    }

    private void resetFineTuneUi() {
        fineTuneRunning.set(false);
        fineTuneBtn.setDisable(false);
        stopFineTuneBtn.setDisable(true);
    }

    @Override
    public void cleanup() {
        if (fineTuneRunning.get()) {
            try {
                Map<String, Object> cancel = new HashMap<>();
                cancel.put("action", "CANCEL_TRAINING");
                PythonExecutionService.getInstance().postRaw(mapper.writeValueAsString(cancel));
            } catch (Exception ignored) {
            }
        }
        fineTuneRunning.set(false);
    }
}
