package com.example.automl_prototype_1.controller;

import com.example.automl_prototype_1.lca.LcaService;
import com.example.automl_prototype_1.service.DesktopAutomationGate;
import javafx.application.Platform;
import javafx.fxml.FXML;
import javafx.scene.control.Alert;
import javafx.scene.control.Button;
import javafx.scene.control.ButtonType;
import javafx.scene.control.TextArea;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;

import java.io.ByteArrayInputStream;
import java.util.Base64;

public class AgentDebugViewController {

    @FXML
    private ImageView screenView;

    @FXML
    private TextArea logicLog;

    @FXML
    private Button btnStartAgent;

    @FXML
    private Button btnStopAgent;

    @FXML
    public void initialize() {
        btnStartAgent.setOnAction(e -> {
            if (!DesktopAutomationGate.getInstance().isEnabled()) {
                Alert confirm = new Alert(Alert.AlertType.CONFIRMATION);
                confirm.setTitle("Desktop automation");
                confirm.setHeaderText("LCA Agent can control mouse/keyboard.");
                confirm.setContentText(
                        "Enable Robot automation for this session? You can stop the agent anytime.");
                if (confirm.showAndWait().orElse(ButtonType.CANCEL) != ButtonType.OK) {
                    logicLog.appendText("Start cancelled — Robot gate remains OFF.\n");
                    return;
                }
                DesktopAutomationGate.getInstance().setEnabled(true);
            }
            LcaService.getInstance().startAgent();
            logicLog.appendText("Agent Started (Robot gate ON).\n");
        });

        btnStopAgent.setOnAction(e -> {
            LcaService.getInstance().stopAgent();
            DesktopAutomationGate.getInstance().setEnabled(false);
            logicLog.appendText("Agent Stopped — Robot gate OFF.\n");
        });

        LcaService.getInstance().setUiCallback(this::updateUi);
    }

    private void updateUi(LcaService.AgentState state) {
        Platform.runLater(() -> {
            try {
                if (state.screenshotBase64 != null) {
                    byte[] imageBytes = Base64.getDecoder().decode(state.screenshotBase64);
                    Image img = new Image(new ByteArrayInputStream(imageBytes));
                    screenView.setImage(img);
                }

                logicLog.appendText("T=" + state.timestamp + ": " + state.thoughtProcess + "\n");
                logicLog.setScrollTop(Double.MAX_VALUE);

            } catch (Exception e) {
                e.printStackTrace();
            }
        });
    }
}
