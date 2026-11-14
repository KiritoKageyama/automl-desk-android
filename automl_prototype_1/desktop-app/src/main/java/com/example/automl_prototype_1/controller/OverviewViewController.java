package com.example.automl_prototype_1.controller;

import com.example.automl_prototype_1.model.progress.OrchestratorState;
import com.example.automl_prototype_1.service.AppStateService;
import com.example.automl_prototype_1.service.RemoteStatusServer;
import com.example.automl_prototype_1.service.progress.ProgressTrackingService;
import com.example.automl_prototype_1.ui.UiMotion;
import javafx.fxml.FXML;
import javafx.scene.Node;
import javafx.scene.control.Button;
import javafx.scene.control.Label;
import javafx.scene.layout.GridPane;
import javafx.scene.layout.Region;
import javafx.scene.layout.StackPane;
import javafx.scene.layout.VBox;

import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Consumer;

/**
 * Landing page for the unified AutoML Studio.
 * Brand-first composition with staggered entrance motion.
 */
public class OverviewViewController {

    @FXML
    private Label datasetValue;
    @FXML
    private Label domainValue;
    @FXML
    private Label experimentValue;
    @FXML
    private Label remoteValue;
    @FXML
    private Label bestValue;

    @FXML
    private StackPane heroPane;
    @FXML
    private Region heroGlow;
    @FXML
    private GridPane metricStrip;
    @FXML
    private VBox domainsSection;
    @FXML
    private VBox actionsSection;

    @FXML
    private Button heroPrimary;
    @FXML
    private Button heroSecondary;
    @FXML
    private Button mlCard;
    @FXML
    private Button dlCard;
    @FXML
    private Button visionCard;
    @FXML
    private Button llmCard;
    @FXML
    private Button genAiCard;
    @FXML
    private Button rlCard;
    @FXML
    private Button uploadAction;
    @FXML
    private Button pipelineAction;
    @FXML
    private Button modelHubAction;
    @FXML
    private Button runAction;

    private final Consumer<String> navigator;
    private final Runnable runExperiment;

    public OverviewViewController(Consumer<String> navigator, Runnable runExperiment) {
        this.navigator = navigator;
        this.runExperiment = runExperiment;
    }

    @FXML
    private void initialize() {
        bindRoute(mlCard, "ml_workspace.fxml");
        bindRoute(dlCard, "dl_workspace.fxml");
        bindRoute(visionCard, "cnn_workspace.fxml");
        bindRoute(llmCard, "llm_workspace.fxml");
        bindRoute(genAiCard, "genai_workspace.fxml");
        bindRoute(rlCard, "rl_lab_view.fxml");
        bindRoute(uploadAction, "upload_view.fxml");
        bindRoute(pipelineAction, "pipeline_simulator.fxml");
        bindRoute(modelHubAction, "model_hub_view.fxml");
        bindRoute(heroPrimary, "upload_view.fxml");
        bindRoute(heroSecondary, "pipeline_simulator.fxml");
        runAction.setOnAction(event -> {
            UiMotion.pressScale(runAction);
            runExperiment.run();
        });
        refreshSummary();
        playEntrance();
    }

    private void playEntrance() {
        List<Node> sequence = new ArrayList<>();
        if (heroPane != null) {
            sequence.add(heroPane);
        }
        if (metricStrip != null) {
            sequence.add(metricStrip);
        }
        if (domainsSection != null) {
            sequence.add(domainsSection);
        }
        if (actionsSection != null) {
            sequence.add(actionsSection);
        }
        UiMotion.staggerIn(sequence, 40, 90);
        if (heroGlow != null) {
            UiMotion.pulseOpacity(heroGlow, 0.35, 0.75, 3200);
            UiMotion.drift(heroGlow, 18, -10, 7000);
        }
    }

    private void bindRoute(Button button, String route) {
        if (button == null) {
            return;
        }
        button.setOnAction(event -> {
            UiMotion.pressScale(button);
            navigator.accept(route);
        });
    }

    private void refreshSummary() {
        AppStateService app = AppStateService.getInstance();
        File file = app.getSelectedDatasetFile();
        datasetValue.setText(file != null ? file.getName() : "No dataset");
        domainValue.setText(app.getCurrentDomain() != null
                ? app.getCurrentDomain().getDisplayName()
                : "Machine Learning");

        OrchestratorState progress = ProgressTrackingService.getInstance().snapshot();
        if (progress.currentRun() != null) {
            experimentValue.setText(progress.currentRun().algoId() + " · "
                    + progress.currentRun().currentIter() + "/" + progress.currentRun().totalIters());
        } else {
            experimentValue.setText("Idle");
        }

        if (progress.globalBestScore() > -1e100 && Double.isFinite(progress.globalBestScore())) {
            bestValue.setText(progress.globalBestAlgo() + " · "
                    + String.format("%.4f", progress.globalBestScore()));
        } else {
            bestValue.setText("No completed run");
        }

        RemoteStatusServer remote = RemoteStatusServer.getInstance();
        remoteValue.setText(remote.isRunning() ? "Online · :" + remote.getPort() : "Offline");
        remoteValue.setStyle(remote.isRunning()
                ? "-fx-text-fill: #5cba7d; -fx-font-weight: bold;"
                : "-fx-text-fill: #d35a4a; -fx-font-weight: bold;");
    }
}
