package com.example.automl_prototype_1.controller;

import com.example.automl_prototype_1.service.AppStateService;
import com.example.automl_prototype_1.service.LogService;
import javafx.fxml.FXML;
import javafx.scene.control.CheckBox;
import javafx.scene.control.Button;
import javafx.scene.layout.VBox;
import javafx.stage.Stage;
import java.util.ArrayList;
import java.util.List;

public class FeatureSelectionDialogController {

    @FXML
    private VBox featuresContainer;
    @FXML
    private Button confirmButton;
    @FXML
    private Button cancelButton;
    @FXML
    private CheckBox selectAllCheckBox;

    private List<CheckBox> featureCheckBoxes = new ArrayList<>();
    private Stage dialogStage;
    private boolean confirmed = false;

    @FXML
    private void initialize() {
        loadFeatures();

        if (selectAllCheckBox != null) {
            selectAllCheckBox.setOnAction(e -> {
                boolean selected = selectAllCheckBox.isSelected();
                for (CheckBox cb : featureCheckBoxes) {
                    cb.setSelected(selected);
                }
            });
        }
    }

    public void setDialogStage(Stage dialogStage) {
        this.dialogStage = dialogStage;
    }

    public boolean isConfirmed() {
        return confirmed;
    }

    private void loadFeatures() {
        List<String> headers = AppStateService.getInstance().getDatasetHeaders();
        String target = AppStateService.getInstance().getTargetColumnName();

        if (headers == null || headers.isEmpty()) {
            LogService.warn("FeatureSelectionDialog", "No headers found in AppState.");
            return;
        }

        // Generate checkboxes
        int index = 0;
        for (String header : headers) {
            // Skip target column if possible, but indices must align with data
            // Usually, target is the last one or user specified.
            if (header.equals(target)) {
                index++;
                continue;
            }

            CheckBox cb = new CheckBox(header + " (Index " + index + ")");
            cb.setUserData(index); // Store original index
            cb.setSelected(true); // Default select all

            featuresContainer.getChildren().add(cb);
            featureCheckBoxes.add(cb);
            index++;
        }
    }

    @FXML
    private void handleConfirm() {
        List<Integer> selectedIndices = new ArrayList<>();
        for (CheckBox cb : featureCheckBoxes) {
            if (cb.isSelected()) {
                selectedIndices.add((Integer) cb.getUserData());
            }
        }

        if (selectedIndices.isEmpty()) {
            // Show alert or just warn?
            LogService.warn("FeatureSelectionDialog", "No features selected!");
            // return; // Allow empty? Probably not.
        }

        AppStateService.getInstance().setManualFeatureIndices(selectedIndices);
        LogService.info("FeatureSelectionDialog", "Saved " + selectedIndices.size() + " manually selected features.");

        confirmed = true;
        if (dialogStage != null)
            dialogStage.close();
    }

    @FXML
    private void handleCancel() {
        confirmed = false;
        if (dialogStage != null)
            dialogStage.close();
    }
}
