package com.example.automl_prototype_1.controller;

import com.example.automl_prototype_1.service.agentic.AgenticArchitectService;
import com.example.automl_prototype_1.service.DynamicCompilerService;

import javafx.fxml.FXML;
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
        this.architectService = new AgenticArchitectService();
        this.compilerService = new DynamicCompilerService();
    }

    @FXML
    public void initialize() {
        if (modeSelector != null) {
            modeSelector.getItems().addAll("RL Environment (IEnvironment)", "ML Classifier (IClassifier)");
            modeSelector.getSelectionModel().select(0);
        }

        generateBtn.setOnAction(e -> handleGenerate());
        compileBtn.setOnAction(e -> handleCompile());

        log("Creator Lab Initialized. Connect to Ollama to start building.");
    }

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
                // We register it as a "Custom Class" type
                // AppStateService needs a way to store this.
                // For now, we can piggyback on Dynamic Models if we treat ID as
                // "CLASS:ClassName"
                com.example.automl_prototype_1.service.AppStateService.getInstance()
                        .registerDynamicModel(className, "CLASS:" + className);
                log("Registered '" + className + "' to Algorithm List.");
            }

            log("NOTE: A restart may be required to fully load the new class bytes in this prototype.");
        } else {
            log("FAILURE: " + result.messages);
        }
    }

    private void log(String msg) {
        Platform.runLater(() -> logArea.appendText(msg + "\n"));
    }
}
