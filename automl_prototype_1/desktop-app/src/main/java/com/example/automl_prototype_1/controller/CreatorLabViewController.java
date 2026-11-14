package com.example.automl_prototype_1.controller;

import com.example.automl_prototype_1.service.agentic.AgenticArchitectService;
import com.example.automl_prototype_1.service.DynamicCompilerService;

import javafx.fxml.FXML;
import javafx.scene.control.Alert;
import javafx.scene.control.Button;
import javafx.scene.control.TextArea;
import javafx.scene.control.TextField;
import javafx.application.Platform;
import java.io.File;

public class CreatorLabViewController {

    @FXML
    private javafx.scene.control.ComboBox<String> modeSelector;

    @FXML
    private TextField promptField;
    @FXML
    private Button generateBtn;
    @FXML
    private TextArea codeArea;
    @FXML
    private TextField classNameField;
    @FXML
    private Button compileBtn;
    @FXML
    private TextArea logArea;

    private final AgenticArchitectService architectService;
    private final DynamicCompilerService compilerService;

    public CreatorLabViewController() {
        this.architectService = AgenticArchitectService.getInstance();
        this.compilerService = new DynamicCompilerService();
    }

    @FXML
    private Button btnStartStream;
    @FXML
    private Button btnStopStream;
    @FXML
    private Button btnRecordCsv;
    @FXML
    private javafx.scene.control.Label lblStreamStatus;
    @FXML
    private javafx.scene.chart.LineChart<Number, Number> liveDataChart;

    // Real-Time State
    private volatile boolean isStreaming = false;
    private volatile boolean isRecording = false;
    private java.io.PrintWriter csvWriter;

    @FXML
    private javafx.scene.control.ComboBox<String> templateSelector;
    @FXML
    private Button undoBtn;

    // Phase 12: Undo history (max 10 steps)
    private final java.util.Deque<String> codeHistory = new java.util.ArrayDeque<>();
    private static final int MAX_HISTORY = 10;

    // Phase 12: Template presets
    private static final java.util.Map<String, String> TEMPLATES = java.util.Map.of(
            "Classic Ensemble",
            "Create a voting ensemble classifier that combines RandomForest, SVM, and KNN with majority vote. " +
                    "Implement as IClassifier.",
            "GPU Gradient Boost",
            "Create an XGBoost classifier that uses the GpuBatchService for GPU-accelerated training. " +
                    "Send data via Arrow IPC to gpu_server.py. Implement as IClassifier.",
            "Stacked Generalization",
            "Create a stacked classifier with SVM, RandomForest, and KNN as base learners and " +
                    "LogisticRegression as the meta-learner. Implement as IClassifier.",
            "Custom MLP",
            "Create a multilayer perceptron (MLP) with 2 hidden layers (128, 64 neurons), ReLU activation, " +
                    "and dropout 0.2. Implement as IClassifier using PyTorch.");

    public void initialize() {
        if (modeSelector != null) {
            modeSelector.getItems().addAll("RL Environment (IEnvironment)", "ML Classifier (IClassifier)");
            modeSelector.getSelectionModel().select(0);
        }

        // Phase 12: Template chooser
        if (templateSelector != null) {
            templateSelector.getItems().addAll("Classic Ensemble", "GPU Gradient Boost",
                    "Stacked Generalization", "Custom MLP");
            templateSelector.setPromptText("Select a template...");
            templateSelector.setOnAction(e -> {
                String chosen = templateSelector.getValue();
                if (chosen != null && TEMPLATES.containsKey(chosen)) {
                    promptField.setText(TEMPLATES.get(chosen));
                    log("Template loaded: " + chosen);
                }
            });
        }

        generateBtn.setOnAction(e -> handleGenerate());
        compileBtn.setOnAction(e -> handleCompile());

        // Phase 12: Undo button
        if (undoBtn != null) {
            undoBtn.setDisable(true);
            undoBtn.setOnAction(e -> handleUndo());
        }

        // Real-Time Handlers
        if (btnStartStream != null) {
            btnStartStream.setOnAction(e -> startStream());
            btnStopStream.setOnAction(e -> stopStream());
            btnRecordCsv.setOnAction(e -> toggleRecording());
        }

        log("Creator Lab Initialized. Connect to Ollama to start building.");
    }

    /** Phase 12: Push current code to undo stack before generating new code. */
    private void pushHistory(String code) {
        if (code != null && !code.isBlank() && !code.startsWith("// Generating")) {
            if (codeHistory.size() >= MAX_HISTORY) {
                codeHistory.pollLast();
            }
            codeHistory.push(code);
            if (undoBtn != null)
                undoBtn.setDisable(codeHistory.isEmpty());
        }
    }

    @FXML
    private void handleUndo() {
        if (!codeHistory.isEmpty()) {
            codeArea.setText(codeHistory.pop());
            if (undoBtn != null)
                undoBtn.setDisable(codeHistory.isEmpty());
            log("Undo: Restored previous code version.");
        }
    }

    private void startStream() {
        if (isStreaming)
            return;
        isStreaming = true;
        btnStartStream.setDisable(true);
        btnStopStream.setDisable(false);
        lblStreamStatus.setText("Status: Streaming...");

        // Clear old data
        liveDataChart.getData().clear();
        javafx.scene.chart.XYChart.Series<Number, Number> series = new javafx.scene.chart.XYChart.Series<>();
        series.setName("Sensor 1");
        liveDataChart.getData().add(series);

        // Subscribe to Service
        com.example.automl_prototype_1.service.RealTimeDataService.getInstance().addListener(point -> {
            if (!isStreaming)
                return;

            Platform.runLater(() -> {
                // Update Chart (Max 100 points visible)
                double t = (point.timestamp() % 100000) / 1000.0; // Wrap time for demo
                double val = point.values()[0]; // Visualize first dimension only for now

                series.getData().add(new javafx.scene.chart.XYChart.Data<>(t, val));
                if (series.getData().size() > 100) {
                    series.getData().remove(0);
                }
            });

            // Record to CSV if enabled
            if (isRecording && csvWriter != null) {
                StringBuilder sb = new StringBuilder();
                sb.append(point.timestamp());
                for (double v : point.values()) {
                    sb.append(",").append(v);
                }
                csvWriter.println(sb.toString());
                csvWriter.flush();
            }
        });

        // Simulate Data if none exists (for demo)
        new Thread(() -> {
            while (isStreaming) {
                try {
                    double noise = Math.random();
                    double val = Math.sin(System.currentTimeMillis() / 1000.0) + noise;
                    com.example.automl_prototype_1.service.RealTimeDataService.getInstance()
                            .pushData(new double[] { val });
                    Thread.sleep(100);
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                }
            }
        }).start();
    }

    private void stopStream() {
        isStreaming = false;
        btnStartStream.setDisable(false);
        btnStopStream.setDisable(true);
        lblStreamStatus.setText("Status: Idle");
        closeRecording();
    }

    private void toggleRecording() {
        if (isRecording) {
            closeRecording();
            btnRecordCsv.setText("Record to CSV");
        } else {
            try {
                String filename = "stream_" + System.currentTimeMillis() + ".csv";
                csvWriter = new java.io.PrintWriter(new java.io.FileWriter(filename));
                csvWriter.println("Timestamp,Value"); // Header
                isRecording = true;
                btnRecordCsv.setText("Stop Recording");
                lblStreamStatus.setText("Status: Recording to " + filename);
            } catch (java.io.IOException e) {
                log("Error starting recording: " + e.getMessage());
            }
        }
    }

    private void closeRecording() {
        if (isRecording) {
            isRecording = false;
            if (csvWriter != null) {
                csvWriter.close();
                csvWriter = null;
            }
            log("Recording saved.");
        }
    }

    // ... existing methods ...

    private void handleGenerate() {
        String prompt = promptField.getText();
        if (prompt == null || prompt.isBlank()) {
            log("Error: Prompt cannot be empty.");
            return;
        }

        if (classNameField.getText().isEmpty()) {
            classNameField.setText("CustomComponent");
        }

        String className = classNameField.getText();
        boolean isClassifier = modeSelector.getSelectionModel().getSelectedIndex() == 1;

        log("Sending request to Architect AI (" + (isClassifier ? "Classifier" : "Environment") + ")...");
        generateBtn.setDisable(true);
        codeArea.setText("// Generating...");

        new Thread(() -> {
            String code;
            // Phase 12: save previous code to undo history before overwriting
            Platform.runLater(() -> pushHistory(codeArea.getText()));

            if (isClassifier) {
                code = architectService.generateClassifierCode(prompt, className);
            } else {
                code = architectService.generateEnvironmentCode(prompt, className);
            }

            Platform.runLater(() -> {
                codeArea.setText(code);
                generateBtn.setDisable(false);
                compileBtn.setDisable(false);
                log("Code generated! Review it before compiling.");
            });
        }).start();
    }

    private void handleCompile() {
        String code = codeArea.getText();
        String className = classNameField.getText();

        if (code.isEmpty() || className.isEmpty()) {
            log("Error: Code or Class Name missing.");
            compileBtn.setDisable(true);
            return;
        }

        log("Auditing generated code...");
        com.example.automl_prototype_1.service.agentic.ConstitutionalService.Verdict verdict =
                com.example.automl_prototype_1.service.agentic.ConstitutionalService.getInstance().auditCode(code);
        if (!verdict.allowed) {
            log("BLOCKED by ConstitutionalService: " + verdict.reason);
            Alert alert = new Alert(Alert.AlertType.ERROR);
            alert.setTitle("Code audit failed");
            alert.setHeaderText("Compile aborted");
            alert.setContentText(verdict.reason);
            alert.showAndWait();
            return;
        }
        log("Code audit passed.");

        log("Saving source file...");
        boolean saved = architectService.saveClassFile(code, className);
        if (!saved) {
            log("Error: Failed to save source file.");
            return;
        }

        log("Compiling...");
        // Locate the file we just saved
        // Hardcoded path match with Service
        String projectRoot = System.getProperty("user.dir");
        File sourceFile = new File(projectRoot,
                "../automl_core/src/main/java/com/example/automl_prototype_1/rl/extensions/" + className + ".java");

        if (!sourceFile.exists()) {
            // Try relative path from desktop app execution context
            sourceFile = new File(
                    "../automl_core/src/main/java/com/example/automl_prototype_1/rl/extensions/" + className + ".java");
        }

        if (!sourceFile.exists()) {
            log("Error: Could not locate saved source file at: " + sourceFile.getAbsolutePath());
            return;
        }

        // Run Dynamic Compilation
        // Note: For this to work fully in a running app, we need a custom ClassLoader.
        // For prototype, we verify compilation success.

        DynamicCompilerService.CompilationResult result = compilerService.compile(sourceFile);

        if (result.success) {
            log("SUCCESS: Compilation passed!");
            log("New component '" + className + "' created.");

            // Auto-Register if it's a classifier
            boolean isClassifier = modeSelector.getSelectionModel().getSelectedIndex() == 1;
            if (isClassifier) {
                com.example.automl_prototype_1.service.AppStateService.getInstance()
                        .registerDynamicModel(className, "CLASS:" + className);
                log("Registered '" + className + "' — it now appears in Algorithm Selection.");
            } else {
                try {
                    com.example.automl_prototype_1.registry.EnvironmentRegistry.getInstance().refresh();
                    log("RL environment compiled — EnvironmentRegistry refreshed.");
                } catch (Exception refreshErr) {
                    log("Compiled, but env refresh failed: " + refreshErr.getMessage());
                }
            }

            log("Compilation OK. Dynamic class bytes were written to target/classes.");
        } else {
            log("FAILURE: " + result.messages);
        }
    }

    private void log(String msg) {
        Platform.runLater(() -> logArea.appendText(msg + "\n"));
    }
}
