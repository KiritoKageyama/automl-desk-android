package com.example.automl_prototype_1.controller;

import com.example.automl_prototype_1.service.OllamaService;
import com.example.automl_prototype_1.service.agentic.ConstitutionalService;
import javafx.application.Platform;
import javafx.fxml.FXML;
import javafx.scene.control.Button;
import javafx.scene.control.Label;
import javafx.scene.control.ScrollPane;
import javafx.scene.control.TextArea;
import javafx.scene.layout.VBox;
import javafx.scene.text.Text;
import javafx.scene.text.TextFlow;
// REMOVED: import com.example.automl_prototype_1.service.agentic.LocalLlmService;

public class ChatViewController {

    @FXML
    private Label statusLabel;
    @FXML
    private ScrollPane scrollPane;
    @FXML
    private VBox messageContainer;
    @FXML
    private TextArea inputField;
    @FXML
    private Button sendButton;

    @FXML
    private final OllamaService llmService = OllamaService.getInstance();
    private com.example.automl_prototype_1.service.agentic.ErrorMonitorService errorMonitor;

    // Callback for closing the panel
    private Runnable onCloseAction;

    public void setOnCloseAction(Runnable action) {
        this.onCloseAction = action;
    }

    @FXML
    private void initialize() {
        // Auto-scroll to bottom
        messageContainer.heightProperty().addListener((obs, oldVal, newVal) -> {
            scrollPane.setVvalue(1.0);
        });

        // Enter key to send (Shift+Enter for newline)
        inputField.setOnKeyPressed(event -> {
            if (event.getCode() == javafx.scene.input.KeyCode.ENTER && !event.isShiftDown()) {
                event.consume();
                handleSend();
            }
        });

        // Initial Greeting
        addBotMessage(
                "Hello! I'm your AutoML Assistant. I can help you analyze your dataset, choose algorithms, or explain features. What would you like to know?");

        // Initialize Error Monitor
        this.errorMonitor = new com.example.automl_prototype_1.service.agentic.ErrorMonitorService();
        this.errorMonitor.setAlertCallback(diagnosis -> {
            Platform.runLater(() -> addSystemAlert(diagnosis));
        });
    }

    @FXML
    private void handleSend() {
        String input = inputField.getText().trim();
        if (input.isEmpty())
            return;

        inputField.clear();
        addUserMessage(input);

        statusLabel.setText("Thinking...");
        sendButton.setDisable(true);

        OllamaService.getInstance().chat(input).thenAccept(response -> {
            Platform.runLater(() -> {
                addBotMessage(response);
                statusLabel.setText("Ready");
                sendButton.setDisable(false);
            });
        }).exceptionally(e -> {
            Platform.runLater(() -> {
                addBotMessage("Error: " + e.getMessage());
                statusLabel.setText("Error");
                sendButton.setDisable(false);
            });
            return null;
        });
    }

    @FXML
    private void handleClose() {
        if (onCloseAction != null) {
            onCloseAction.run();
        }
    }

    private void addUserMessage(String text) {
        // User Message (Right Aligned)
        Text textNode = new Text(text);
        textNode.wrappingWidthProperty().bind(messageContainer.widthProperty().multiply(0.8));
        // Use standard text color (black/white handles by theme if Text class is
        // styled)
        // Hardcoding standard black for bubbles to ensure readability against light
        // blue background
        textNode.setStyle("-fx-fill: #1f2937;");

        TextFlow flow = new TextFlow(textNode);
        flow.setStyle("-fx-background-color: #dbeafe; -fx-background-radius: 15; -fx-padding: 10;");

        javafx.scene.layout.HBox container = new javafx.scene.layout.HBox(flow);
        container.setAlignment(javafx.geometry.Pos.CENTER_RIGHT);

        messageContainer.getChildren().add(container);
    }

    private void addBotMessage(String text) {
        // Check for Code Edit Request
        if (text.contains("<edit_request>")) {
            handleCodeEditRequest(text);
            return;
        }
        // Check for Create File Request
        if (text.contains("<create_file>")) {
            handleCreateFileRequest(text);
            return;
        }
        // Check for Constitution Update Request
        if (text.contains("<update_constitution>")) {
            handleUpdateConstitutionRequest(text);
            return;
        }
        // Check for Strict Mode Request
        if (text.contains("<set_strict_mode>")) {
            handleStrictModeRequest(text);
            return;
        }

        // Bot Message (Left Aligned)
        Text textNode = new Text(text);
        textNode.wrappingWidthProperty().bind(messageContainer.widthProperty().multiply(0.8));
        textNode.getStyleClass().add("text-normal");

        TextFlow flow = new TextFlow(textNode);
        flow.getStyleClass().add("chat-bubble-received");
        flow.setStyle("-fx-background-color: #374151; -fx-background-radius: 15; -fx-padding: 10;");
        textNode.setStyle("-fx-fill: #f3f4f6;");

        javafx.scene.layout.HBox container = new javafx.scene.layout.HBox(flow);
        container.setAlignment(javafx.geometry.Pos.CENTER_LEFT);

        messageContainer.getChildren().add(container);
    }

    // Reuse handleCodeEditRequest logic but adapted for path tag
    private void handleCreateFileRequest(String xml) {
        String path = extractTag(xml, "path");
        String content = extractTag(xml, "content");

        if (path == null || content == null) {
            addBotMessage("Error: AI proposed an invalid file creation format.");
            return;
        }

        VBox editBox = new VBox(5);
        editBox.setStyle(
                "-fx-background-color: #1f2937; -fx-padding: 10; -fx-background-radius: 5; -fx-border-color: #10b981; -fx-border-width: 1;"); // Green
                                                                                                                                              // border
                                                                                                                                              // for
                                                                                                                                              // Create
        editBox.setMaxWidth(400);

        Label title = new Label("Proposed New File: " + path);
        title.setStyle("-fx-text-fill: #34d399; -fx-font-weight: bold;"); // Green Text

        TextArea codePreview = new TextArea(content);
        codePreview.setEditable(false);
        codePreview.setPrefRowCount(10);
        codePreview.setStyle(
                "-fx-font-family: 'Consolas', monospace; -fx-control-inner-background: #111827; -fx-text-fill: #d1d5db;");

        Button applyBtn = new Button("Create File");
        applyBtn.setStyle("-fx-background-color: #059669; -fx-text-fill: white; -fx-cursor: hand;");

        Label status = new Label("");
        status.setStyle("-fx-text-fill: #9ca3af;");

        applyBtn.setOnAction(e -> {
            applyBtn.setDisable(true);
            status.setText("Creating...");
            new Thread(() -> {
                try {
                    com.example.automl_prototype_1.service.agentic.CodeModificationService cms = new com.example.automl_prototype_1.service.agentic.CodeModificationService();
                    cms.writeFile(path, content);
                    Platform.runLater(() -> {
                        status.setText("Success! File created.");
                        status.setStyle("-fx-text-fill: #34d399;");
                    });
                } catch (Exception ex) {
                    Platform.runLater(() -> {
                        status.setText("Error: " + ex.getMessage());
                        status.setStyle("-fx-text-fill: #ef4444;");
                        applyBtn.setDisable(false);
                    });
                }
            }).start();
        });

        editBox.getChildren().addAll(title, codePreview, applyBtn, status);

        javafx.scene.layout.HBox container = new javafx.scene.layout.HBox(editBox);
        container.setAlignment(javafx.geometry.Pos.CENTER_LEFT);
        messageContainer.getChildren().add(container);
    }

    private void handleUpdateConstitutionRequest(String xml) {
        String id = extractTag(xml, "id");
        String condition = extractTag(xml, "condition");
        String verdict = extractTag(xml, "verdict");
        String reason = extractTag(xml, "reason");

        if (id == null || condition == null || verdict == null) {
            addBotMessage("Error: AI proposed an invalid constitution update format.");
            return;
        }

        VBox editBox = new VBox(5);
        editBox.setStyle(
                "-fx-background-color: #1f2937; -fx-padding: 10; -fx-background-radius: 5; -fx-border-color: #f59e0b; -fx-border-width: 1;"); // Amber
                                                                                                                                              // for
                                                                                                                                              // Constitution
        editBox.setMaxWidth(400);

        Label title = new Label("Suggest Rule Update: " + id);
        title.setStyle("-fx-text-fill: #fbbf24; -fx-font-weight: bold;"); // Amber Text

        TextArea preview = new TextArea("Condition: " + condition + "\nVerdict: " + verdict + "\nReason: " + reason);
        preview.setEditable(false);
        preview.setPrefRowCount(4);
        preview.setStyle(
                "-fx-font-family: 'Consolas', monospace; -fx-control-inner-background: #111827; -fx-text-fill: #d1d5db;");

        Button applyBtn = new Button("Update Rule");
        applyBtn.setStyle("-fx-background-color: #d97706; -fx-text-fill: white; -fx-cursor: hand;");

        Label status = new Label("");
        status.setStyle("-fx-text-fill: #9ca3af;");

        applyBtn.setOnAction(e -> {
            applyBtn.setDisable(true);
            status.setText("Updating...");
            new Thread(() -> {
                try {
                    ConstitutionalService cs = ConstitutionalService.getInstance();
                    cs.addRule(id, condition, verdict, reason);
                    Platform.runLater(() -> {
                        status.setText("Success! Constitution Amended.");
                        status.setStyle("-fx-text-fill: #34d399;");
                    });
                } catch (Exception ex) {
                    Platform.runLater(() -> {
                        status.setText("Error: " + ex.getMessage());
                        status.setStyle("-fx-text-fill: #ef4444;");
                        applyBtn.setDisable(false);
                    });
                }
            }).start();
        });

        editBox.getChildren().addAll(title, preview, applyBtn, status);

        javafx.scene.layout.HBox container = new javafx.scene.layout.HBox(editBox);
        container.setAlignment(javafx.geometry.Pos.CENTER_LEFT);
        messageContainer.getChildren().add(container);
    }

    private void addSystemAlert(String message) {
        VBox alertBox = new VBox(5);
        alertBox.setStyle(
                "-fx-background-color: #450a0a; -fx-padding: 10; -fx-background-radius: 5; -fx-border-color: #ef4444; -fx-border-width: 1;");
        alertBox.setMaxWidth(400);

        Label title = new Label("SYSTEM ALERT: Critical Error Detected");
        title.setStyle("-fx-text-fill: #ef4444; -fx-font-weight: bold;");

        Label content = new Label(message);
        content.setWrapText(true);
        content.setStyle("-fx-text-fill: #fecaca;");

        alertBox.getChildren().addAll(title, content);

        javafx.scene.layout.HBox container = new javafx.scene.layout.HBox(alertBox);
        container.setAlignment(javafx.geometry.Pos.CENTER_LEFT);
        messageContainer.getChildren().add(container);

        // Scroll to bottom
        scrollPane.setVvalue(1.0);
    }

    private void handleStrictModeRequest(String xml) {
        String val = extractTag(xml, "set_strict_mode");
        boolean enable = Boolean.parseBoolean(val);

        VBox box = new VBox(5);
        box.setStyle(
                "-fx-background-color: #1f2937; -fx-padding: 10; -fx-background-radius: 5; -fx-border-color: #dc2626; -fx-border-width: 1;");
        box.setMaxWidth(400);

        Label title = new Label(
                enable ? "System Alert: ENABLE Absolute Strict Mode?" : "System Alert: DISABLE Strict Mode?");
        title.setStyle("-fx-text-fill: #f87171; -fx-font-weight: bold;");

        Label desc = new Label(enable
                ? "The AI will become an ABSOLUTE Rule Abider. It will VETO any algorithm that violates resource constraints (RAM/CPU) without exception."
                : "The AI will relax constraints. Warnings will remain warnings.");
        desc.setWrapText(true);
        desc.setStyle("-fx-text-fill: #d1d5db;");

        Button btn = new Button(enable ? "Confirm Strict Mode" : "Disable Strict Mode");
        btn.setStyle(enable ? "-fx-background-color: #dc2626; -fx-text-fill: white;"
                : "-fx-background-color: #4b5563; -fx-text-fill: white;");

        Label status = new Label("");
        status.setStyle("-fx-text-fill: #9ca3af;");

        btn.setOnAction(e -> {
            btn.setDisable(true);
            new Thread(() -> {
                ConstitutionalService cs = ConstitutionalService.getInstance();
                cs.setStrictMode(enable);
                Platform.runLater(() -> {
                    status.setText("Success! Strict Mode: " + (enable ? "ON" : "OFF"));
                    status.setStyle("-fx-text-fill: #34d399;");
                });
            }).start();
        });

        box.getChildren().addAll(title, desc, btn, status);

        javafx.scene.layout.HBox container = new javafx.scene.layout.HBox(box);
        container.setAlignment(javafx.geometry.Pos.CENTER_LEFT);
        messageContainer.getChildren().add(container);
    }

    private void handleCodeEditRequest(String xml) {
        // Parse XML (Simple String extraction for robustness)
        String file = extractTag(xml, "file");
        String content = extractTag(xml, "content");

        if (file == null || content == null) {
            addBotMessage("Error: AI proposed an invalid code edit format.");
            return;
        }

        VBox editBox = new VBox(5);
        editBox.setStyle(
                "-fx-background-color: #1f2937; -fx-padding: 10; -fx-background-radius: 5; -fx-border-color: #4b5563; -fx-border-width: 1;");
        editBox.setMaxWidth(400);

        Label title = new Label("Proposed Edit: " + file);
        title.setStyle("-fx-text-fill: #60a5fa; -fx-font-weight: bold;");

        TextArea codePreview = new TextArea(content);
        codePreview.setEditable(false);
        codePreview.setPrefRowCount(10);
        codePreview.setStyle(
                "-fx-font-family: 'Consolas', monospace; -fx-control-inner-background: #111827; -fx-text-fill: #d1d5db;");

        Button applyBtn = new Button("Apply Edit");
        applyBtn.setStyle("-fx-background-color: #2563eb; -fx-text-fill: white; -fx-cursor: hand;");

        Label status = new Label("");
        status.setStyle("-fx-text-fill: #9ca3af;");

        applyBtn.setOnAction(e -> {
            applyBtn.setDisable(true);
            status.setText("Applying...");
            new Thread(() -> {
                try {
                    com.example.automl_prototype_1.service.agentic.CodeModificationService cms = new com.example.automl_prototype_1.service.agentic.CodeModificationService();
                    cms.writeFile(file, content);
                    Platform.runLater(() -> {
                        status.setText("Success! Backup created.");
                        status.setStyle("-fx-text-fill: #34d399;");
                    });
                } catch (Exception ex) {
                    Platform.runLater(() -> {
                        status.setText("Error: " + ex.getMessage());
                        status.setStyle("-fx-text-fill: #ef4444;");
                        applyBtn.setDisable(false);
                    });
                }
            }).start();
        });

        editBox.getChildren().addAll(title, codePreview, applyBtn, status);

        javafx.scene.layout.HBox container = new javafx.scene.layout.HBox(editBox);
        container.setAlignment(javafx.geometry.Pos.CENTER_LEFT);
        messageContainer.getChildren().add(container);
    }

    private String extractTag(String xml, String tag) {
        String startTag = "<" + tag + ">";
        String endTag = "</" + tag + ">";
        int start = xml.indexOf(startTag);
        int end = xml.indexOf(endTag);
        if (start == -1 || end == -1)
            return null;
        return xml.substring(start + startTag.length(), end).trim();
    }
}
