package com.example.automl_prototype_1.controller;

import com.example.automl_prototype_1.viewmodel.AlgoSelectViewModel;
import javafx.collections.FXCollections;
import javafx.fxml.FXML;
import javafx.scene.control.*;

public class AlgoSelectViewController {
    @FXML
    private TextArea adviceTextArea;
    @FXML
    private CheckBox checkGA, checkIGPSO, checkWWO, checkBPSO, checkASO, checkPLS, checkSA, checkGWO, checkDE;
    @FXML
    private ChoiceBox<String> baseModelChoiceBox;

    @FXML
    private ChoiceBox<String> scalingStrategyChoiceBox;
    @FXML
    private TextField kFoldsTextField;
    @FXML
    private CheckBox enableTuningCheckBox;
    @FXML
    private Button recommendButton;

    private final AlgoSelectViewModel viewModel;

    public AlgoSelectViewController() {
        this.viewModel = new AlgoSelectViewModel();
    }

    @FXML
    private void initialize() {
        // Bind UI elements to ViewModel properties

        // Advice Text
        adviceTextArea.textProperty().bind(viewModel.adviceTextProperty());

        // Tuning Checkbox
        enableTuningCheckBox.selectedProperty().bindBidirectional(viewModel.tuningEnabledProperty());

        // K-Folds TextField
        kFoldsTextField.textProperty().bindBidirectional(viewModel.kFoldsProperty());

        // Scaling Strategy ChoiceBox
        scalingStrategyChoiceBox
                .setItems(FXCollections.observableArrayList("Standardization", "Normalization", "None"));
        scalingStrategyChoiceBox.valueProperty().bindBidirectional(viewModel.scalingStrategyProperty());

        // Base Model ChoiceBox
        baseModelChoiceBox.setItems(FXCollections.observableArrayList(viewModel.getAvailableClassifiers()));
        baseModelChoiceBox.valueProperty().bindBidirectional(viewModel.selectedBaseModelProperty());

        // Algorithm Checkboxes
        checkGA.selectedProperty().bindBidirectional(viewModel.gaSelectedProperty());
        checkIGPSO.selectedProperty().bindBidirectional(viewModel.igpsoSelectedProperty());
        checkWWO.selectedProperty().bindBidirectional(viewModel.wwoSelectedProperty());
        checkBPSO.selectedProperty().bindBidirectional(viewModel.bpsoSelectedProperty());
        checkASO.selectedProperty().bindBidirectional(viewModel.asoSelectedProperty());
        checkPLS.selectedProperty().bindBidirectional(viewModel.plsSelectedProperty());
        checkSA.selectedProperty().bindBidirectional(viewModel.saSelectedProperty());
        checkGWO.selectedProperty().bindBidirectional(viewModel.gwoSelectedProperty());
        checkDE.selectedProperty().bindBidirectional(viewModel.deSelectedProperty());

        // Button Action
        recommendButton.setText("Get Advice");
        recommendButton.setOnAction(event -> viewModel.generateAdvice());
    }
}