package com.example.automl_prototype_1.controller;

import com.example.automl_prototype_1.service.LibraryManagerService;
import com.example.automl_prototype_1.classifier.GenericHuggingFaceClassifier;
import com.example.automl_prototype_1.viewmodel.AlgoSelectViewModel;
import javafx.application.Platform;
import javafx.fxml.FXML;
import javafx.scene.control.Button;
import javafx.scene.control.TextArea;
import javafx.scene.control.TextField;
import java.util.concurrent.Executors;

public class ModelHubViewController {

    @FXML
    private TextField txtLibraryName;
    @FXML
    private Button btnInstallLib;
    @FXML
    private TextArea txtConsole;

    @FXML
    private TextField txtModelId; // e.g. facebook/bart-large
    @FXML
    private TextField txtModelName; // e.g. My BART
    @FXML
    private Button btnAddModel;

    // We need access to AlgoSelectViewModel to register the model?
    // Or we assume a global Registry service.
    // AlgoSelectViewModel pulls from a hardcoded list mostly, we need to update
    // that list?
    // AlgoSelectViewModel calls 'getAvailableClassifiers'. We need to make that
    // dynamic.

    private AlgoSelectViewModel sharedViewModel; // Injected

    public void setSharedViewModel(AlgoSelectViewModel vm) {
        this.sharedViewModel = vm;
    }

    @FXML
    private void initialize() {
        btnInstallLib.setOnAction(e -> handleInstall());
        btnAddModel.setOnAction(e -> handleAddModel());
    }

    private void handleInstall() {
        String lib = txtLibraryName.getText().trim();
        if (lib.isEmpty())
            return;

        log("Installing " + lib + "...");
        btnInstallLib.setDisable(true);

        LibraryManagerService.getInstance().installLibrary(lib)
                .thenAccept(output -> Platform.runLater(() -> {
                    log(output);
                    log("Done.");
                    btnInstallLib.setDisable(false);
                }));
    }

    private void handleAddModel() {
        String id = txtModelId.getText().trim();
        String name = txtModelName.getText().trim();

        if (id.isEmpty())
            return;
        if (name.isEmpty())
            name = id;

        log("Registering Model: " + name + " (" + id + ")...");

        // Add to Registry
        // We need a mechanism to persist this. For now, we add to AppState dynamic
        // list?
        // Let's add 'DynamicModels' to AppStateService

        com.example.automl_prototype_1.service.AppStateService.getInstance()
                .registerDynamicModel(name, id);

        log("Model Added! Go to Algorithm Selection to see it.");
        txtModelId.clear();
        txtModelName.clear();
    }

    private void log(String msg) {
        txtConsole.appendText(msg + "\n");
    }
}
