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
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;

import java.io.File;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.atomic.AtomicBoolean;

public class GenAiWorkspaceController implements Cleanable {

    @FXML
    private ChoiceBox<String> taskChoiceBox;
    @FXML
    private ChoiceBox<String> modelChoiceBox;
    @FXML
    private TextField hfModelIdField;
    @FXML
    private ChoiceBox<String> resolutionChoiceBox;
    @FXML
    private TextField stepsField, cfgField;
    @FXML
    private CheckBox chkUseGpu;
    @FXML
    private CheckBox chkAllowPreview;

    @FXML
    private TextField promptInputField;
    @FXML
    private Button generateBtn;
    @FXML
    private TextArea logArea;
    @FXML
    private ImageView previewImage;

    private final AppStateService appState = AppStateService.getInstance();
    private final ObjectMapper mapper = new ObjectMapper();
    private final AtomicBoolean generating = new AtomicBoolean(false);

    @FXML
    private void initialize() {
        appState.setCurrentDomain(Domain.GENAI);

        taskChoiceBox.setItems(FXCollections.observableArrayList("Text-to-Image", "Image-to-Image", "Inpainting"));
        taskChoiceBox.getSelectionModel().select(0);
        taskChoiceBox.getSelectionModel().selectedItemProperty().addListener((obs, o, n) -> {
            if (n != null && !"Text-to-Image".equals(n)) {
                appendLog("Note: " + n + " falls back to Text-to-Image until mask/init-image UI is added.\n");
            }
        });

        modelChoiceBox.setItems(
                FXCollections.observableArrayList("Stable Diffusion v1.5", "Stable Diffusion XL", "StyleGAN2"));
        modelChoiceBox.getSelectionModel().select(0);
        modelChoiceBox.getSelectionModel().selectedItemProperty().addListener((obs, o, n) -> {
            if (n != null && n.contains("StyleGAN") && (hfModelIdField.getText() == null || hfModelIdField.getText().isBlank())) {
                appendLog("StyleGAN2 needs a HuggingFace model id in the Advanced field.\n");
            }
            if (n != null && n.contains("XL") && (hfModelIdField.getText() == null || hfModelIdField.getText().isBlank())) {
                hfModelIdField.setPromptText("stabilityai/stable-diffusion-xl-base-1.0 (optional)");
            }
        });

        resolutionChoiceBox.setItems(FXCollections.observableArrayList("512x512", "768x768", "1024x1024"));
        resolutionChoiceBox.getSelectionModel().select(0);

        if (stepsField.getText() == null || stepsField.getText().isBlank()) {
            stepsField.setText("20");
        }
        if (cfgField.getText() == null || cfgField.getText().isBlank()) {
            cfgField.setText("7.5");
        }

        chkUseGpu.setSelected(appState.isUseGpu());
        chkUseGpu.selectedProperty().addListener((obs, o, n) -> appState.setUseGpu(n));
    }

    @FXML
    private void handleGenerateAction() {
        String prompt = promptInputField.getText();
        if (prompt == null || prompt.isBlank()) {
            appendLog("Enter a prompt first.\n");
            return;
        }
        if (!generating.compareAndSet(false, true)) {
            appendLog("Generation already in progress...\n");
            return;
        }

        generateBtn.setDisable(true);
        appendLog("User requested generation: " + prompt + "\n");
        appendLog("Contacting GPU server (GENERATE_IMAGE)...\n");

        int[] wh = parseResolution(resolutionChoiceBox.getValue());
        int steps = parseInt(stepsField.getText(), 20);
        double cfg = parseDouble(cfgField.getText(), 7.5);

        Map<String, Object> request = new HashMap<>();
        request.put("action", "GENERATE_IMAGE");
        request.put("task", taskChoiceBox.getValue());
        request.put("model", modelChoiceBox.getValue());
        request.put("prompt", prompt);
        request.put("hf_model_id", hfModelIdField.getText() == null ? "" : hfModelIdField.getText().trim());
        request.put("width", wh[0]);
        request.put("height", wh[1]);
        request.put("steps", steps);
        request.put("cfg", cfg);
        request.put("use_gpu", chkUseGpu.isSelected());
        request.put("allow_preview", chkAllowPreview != null && chkAllowPreview.isSelected());

        new Thread(() -> {
            try {
                PythonExecutionService py = PythonExecutionService.getInstance();
                if (!py.isServerRunning()) {
                    Platform.runLater(() -> appendLog("Starting GPU server...\n"));
                    py.startGpuServer();
                }
                String json = mapper.writeValueAsString(request);
                String resp = py.postRaw(json);
                @SuppressWarnings("unchecked")
                Map<String, Object> map = mapper.readValue(resp, Map.class);

                Platform.runLater(() -> {
                    if (Boolean.TRUE.equals(map.get("success"))) {
                        String path = String.valueOf(map.get("path"));
                        String engine = String.valueOf(map.get("engine"));
                        String message = String.valueOf(map.get("message"));
                        boolean fallback = Boolean.TRUE.equals(map.get("fallback"));
                        Object warnings = map.get("warnings");
                        if (fallback) {
                            appendLog("WARNING: procedural fallback preview (full Diffusers not available).\n");
                        }
                        if (warnings instanceof java.util.List<?> list && !list.isEmpty()) {
                            appendLog("Notes: " + list + "\n");
                        }
                        appendLog("OK [" + engine + "]: " + message + "\n");
                        appendLog("Saved: " + path + "\n\n");
                        showPreview(path);
                    } else {
                        appendLog("FAILED: " + map.getOrDefault("error", resp) + "\n\n");
                    }
                });
            } catch (Exception e) {
                LogService.error("GenAI", e.getMessage());
                Platform.runLater(() -> appendLog("Error: " + e.getMessage() + "\n\n"));
            } finally {
                generating.set(false);
                Platform.runLater(() -> generateBtn.setDisable(false));
            }
        }, "genai-worker").start();
    }

    private void showPreview(String path) {
        if (previewImage == null || path == null)
            return;
        try {
            File f = new File(path);
            if (!f.exists())
                return;
            previewImage.setImage(new Image(f.toURI().toString(), true));
        } catch (Exception e) {
            appendLog("Preview failed: " + e.getMessage() + "\n");
        }
    }

    private void appendLog(String msg) {
        if (logArea != null) {
            logArea.appendText(msg);
        }
    }

    private static int[] parseResolution(String value) {
        if (value != null && value.contains("x")) {
            String[] parts = value.toLowerCase().split("x");
            try {
                return new int[] { Integer.parseInt(parts[0].trim()), Integer.parseInt(parts[1].trim()) };
            } catch (Exception ignored) {
            }
        }
        return new int[] { 512, 512 };
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
        generating.set(false);
    }
}
