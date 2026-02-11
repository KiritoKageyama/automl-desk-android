package com.example.automl_prototype_1.controller;

import com.example.automl_prototype_1.model.Dataset;
import com.example.automl_prototype_1.service.AppStateService;
import com.example.automl_prototype_1.dataprovider.CsvDataProvider;
import com.example.automl_prototype_1.dataprovider.SyntheticDataGenerator;
import javafx.collections.FXCollections;
import javafx.fxml.FXML;
import javafx.scene.control.*;
import javafx.stage.FileChooser;
import javafx.stage.DirectoryChooser;
import javafx.concurrent.Task;

import java.io.File;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class UploadViewController {
    @FXML
    private TextArea analysisTextArea;
    @FXML
    private Button browseButton;
    @FXML
    private Label selectedFileLabel;
    @FXML
    private ChoiceBox<String> targetColumnChoiceBox;
    @FXML
    private Button analyzeButton;

    // --- Phase 2: 3-Tier UI Components ---
    @FXML
    private javafx.scene.control.ComboBox<AppStateService.DataType> dataTypeComboBox;
    @FXML
    private javafx.scene.control.ComboBox<AppStateService.DataQuality> dataQualityComboBox;
    @FXML
    private javafx.scene.control.CheckBox enableQualityCheck;
    @FXML
    private javafx.scene.control.ComboBox<AppStateService.TrainingMode> trainingModeComboBox;
    @FXML
    private javafx.scene.layout.VBox dynamicOptionsBox; // Container for these new options if needed

    private final AppStateService appStateService = AppStateService.getInstance();

    public UploadViewController() {
        // No-arg constructor for FXML
    }

    @FXML
    private void initialize() {
        browseButton.setOnAction(event -> handleBrowse());

        // Listener for when a new target column is selected
        targetColumnChoiceBox.getSelectionModel().selectedItemProperty().addListener(
                (obs, oldVal, newVal) -> {
                    if (newVal != null) {
                        appStateService.setTargetColumnName(newVal);
                    }
                });

        // --- Phase 2: 3-Tier UI Initialization ---
        init3TierSelection();

        // Initialize view based on current state
        updateViewFromState();
        analyzeButton.setOnAction(event -> handleAnalyze());
        updateViewFromState(); // Call this to set initial button state
    }

    private void init3TierSelection() {
        // 1. Top Level (Data Type)
        if (dataTypeComboBox != null) {
            dataTypeComboBox.setItems(FXCollections.observableArrayList(AppStateService.DataType.values()));
            dataTypeComboBox.setValue(AppStateService.DataType.NUMERICAL); // Default
            dataTypeComboBox.setOnAction(e -> {
                appStateService.setSelectedDataType(dataTypeComboBox.getValue());
                updateTrainingModes(); // Logic to filter bottom level
            });
        }

        // 2. Middle Level (Data Quality)
        if (dataQualityComboBox != null) {
            dataQualityComboBox.setItems(FXCollections.observableArrayList(AppStateService.DataQuality.values()));
            dataQualityComboBox.setValue(AppStateService.DataQuality.STRUCTURED);
            dataQualityComboBox.setDisable(true); // Default disabled
            dataQualityComboBox
                    .setOnAction(e -> appStateService.setSelectedDataQuality(dataQualityComboBox.getValue()));
        }

        if (enableQualityCheck != null) {
            enableQualityCheck.setSelected(false);
            enableQualityCheck.setOnAction(e -> {
                boolean enabled = enableQualityCheck.isSelected();
                appStateService.setEnableQualityCheck(enabled);
                if (dataQualityComboBox != null)
                    dataQualityComboBox.setDisable(!enabled);
            });
        }

        // 3. Bottom Level (Training Mode)
        if (trainingModeComboBox != null) {
            updateTrainingModes(); // Initial population
            trainingModeComboBox.setOnAction(e -> {
                if (trainingModeComboBox.getValue() != null)
                    appStateService.setSelectedTrainingMode(trainingModeComboBox.getValue());
            });
        }
    }

    private void updateTrainingModes() {
        if (dataTypeComboBox == null || trainingModeComboBox == null)
            return;

        AppStateService.DataType selectedType = dataTypeComboBox.getValue();
        List<AppStateService.TrainingMode> relevantModes = new java.util.ArrayList<>();

        // Dynamic Filtering Logic
        switch (selectedType) {
            case NUMERICAL:
                relevantModes.add(AppStateService.TrainingMode.STATISTICAL);
                relevantModes.add(AppStateService.TrainingMode.ML_ALGORITHMS);
                relevantModes.add(AppStateService.TrainingMode.NEURAL_NETWORK_MLP);
                break;
            case IMAGE:
                relevantModes.add(AppStateService.TrainingMode.DEEP_LEARNING);
                relevantModes.add(AppStateService.TrainingMode.CNN);
                relevantModes.add(AppStateService.TrainingMode.VISION_TRANSFORMERS);
                break;
            case TEXT:
                relevantModes.add(AppStateService.TrainingMode.GEN_AI);
                relevantModes.add(AppStateService.TrainingMode.ATTENTION_NETWORKS);
                relevantModes.add(AppStateService.TrainingMode.RNN_LSTM);
                break;
            case USER_PATTERN:
                relevantModes.add(AppStateService.TrainingMode.AGENTIC_TRAINING);
                relevantModes.add(AppStateService.TrainingMode.REINFORCEMENT_LEARNING);
                break;
        }

        trainingModeComboBox.setItems(FXCollections.observableArrayList(relevantModes));
        if (!relevantModes.isEmpty()) {
            trainingModeComboBox.setValue(relevantModes.get(0));
            appStateService.setSelectedTrainingMode(relevantModes.get(0));
        }
    }

    // Old handleAnalyze removed.

    @FXML
    private void handleAdvancedSettings() {
        // Create a custom dialog for preprocessing settings
        javafx.scene.control.Dialog<com.example.automl_prototype_1.model.PreprocessingConfig> dialog = new javafx.scene.control.Dialog<>();
        dialog.setTitle("Advanced Preprocessing Settings");
        dialog.setHeaderText("Configure Data Cleaning & Scaling");

        // Set the button types
        javafx.scene.control.ButtonType saveButtonType = new javafx.scene.control.ButtonType("Save",
                javafx.scene.control.ButtonBar.ButtonData.OK_DONE);
        dialog.getDialogPane().getButtonTypes().addAll(saveButtonType, javafx.scene.control.ButtonType.CANCEL);

        // Create the layout
        javafx.scene.layout.GridPane grid = new javafx.scene.layout.GridPane();
        grid.setHgap(10);
        grid.setVgap(10);
        grid.setPadding(new javafx.geometry.Insets(20, 150, 10, 10));

        // Get current config
        com.example.automl_prototype_1.model.PreprocessingConfig currentConfig = appStateService
                .getPreprocessingConfig();

        // Add controls
        javafx.scene.control.CheckBox removeIds = new javafx.scene.control.CheckBox("Auto-remove ID Columns");
        removeIds.setSelected(currentConfig.isAutoRemoveIdColumns());

        javafx.scene.control.CheckBox impute = new javafx.scene.control.CheckBox("Impute Missing Values (Mean/Mode)");
        impute.setSelected(currentConfig.isImputeMissingValues());

        javafx.scene.control.CheckBox outliers = new javafx.scene.control.CheckBox("Remove Outliers (IQR Method)");
        outliers.setSelected(currentConfig.isRemoveOutliers());

        javafx.scene.control.ChoiceBox<String> scaling = new javafx.scene.control.ChoiceBox<>();
        scaling.getItems().addAll("Standardization", "Normalization", "None");
        scaling.setValue(currentConfig.getScalingStrategy());

        grid.add(new javafx.scene.control.Label("Scaling Strategy:"), 0, 0);
        grid.add(scaling, 1, 0);
        grid.add(removeIds, 0, 1);
        grid.add(impute, 0, 2);
        grid.add(outliers, 0, 3);

        dialog.getDialogPane().setContent(grid);

        // Convert the result to a config object when the login button is clicked.
        dialog.setResultConverter(dialogButton -> {
            if (dialogButton == saveButtonType) {
                return new com.example.automl_prototype_1.model.PreprocessingConfig(
                        removeIds.isSelected(),
                        impute.isSelected(),
                        outliers.isSelected(),
                        scaling.getValue());
            }
            return null;
        });

        java.util.Optional<com.example.automl_prototype_1.model.PreprocessingConfig> result = dialog.showAndWait();

        result.ifPresent(newConfig -> {
            appStateService.setPreprocessingConfig(newConfig);
            analysisTextArea.setText(analysisTextArea.getText() + "\n\n[Settings Updated]: " + newConfig.toString());
        });
    }

    @FXML
    private void handleBrowse() {
        AppStateService.DataType selectedType = appStateService.getSelectedDataType();

        if (selectedType == AppStateService.DataType.IMAGE || selectedType == AppStateService.DataType.TEXT) {
            DirectoryChooser directoryChooser = new DirectoryChooser();
            directoryChooser.setTitle("Select Dataset Folder");
            File selectedDirectory = directoryChooser.showDialog(browseButton.getScene().getWindow());

            if (selectedDirectory != null) {
                selectedFileLabel.setText(selectedDirectory.getAbsolutePath()); // Changed from datasetPathField
                appStateService.setSelectedDatasetFile(selectedDirectory);
                analyzeButton.setDisable(false);
            }
        } else {
            // Default CSV File Chooser
            FileChooser fileChooser = new FileChooser();
            fileChooser.setTitle("Select Dataset File");
            fileChooser.getExtensionFilters().add(new FileChooser.ExtensionFilter("CSV Files", "*.csv"));
            File selectedFile = fileChooser.showOpenDialog(browseButton.getScene().getWindow());

            if (selectedFile != null) {
                selectedFileLabel.setText(selectedFile.getAbsolutePath()); // Changed from datasetPathField
                appStateService.setSelectedDatasetFile(selectedFile);
                analyzeButton.setDisable(false);
            }
        }
    }

    @FXML
    private void handleAnalyze() {
        if (appStateService.getSelectedDatasetFile() == null) {
            analysisTextArea.appendText("Error: No dataset selected.\n");
            return;
        }

        analysisTextArea.appendText("Analyzing dataset...\n");

        Task<Dataset> loadTask = new Task<>() {
            @Override
            protected Dataset call() throws Exception {
                AppStateService.DataType dtype = appStateService.getSelectedDataType();
                File source = appStateService.getSelectedDatasetFile();

                if (dtype == AppStateService.DataType.IMAGE) {
                    return loadImagesFromFolder(source);
                } else if (dtype == AppStateService.DataType.TEXT) {
                    return loadImagesFromFolder(source);
                } else {
                    // CSV Loading
                    try (FileReader reader = new FileReader(source)) {
                        return new CsvDataProvider().loadDataset(reader);
                    }
                }
            }
        };

        loadTask.setOnSucceeded(e -> {
            Dataset dataset = loadTask.getValue();
            appStateService.setCurrentDataset(dataset);
            appStateService.setDatasetHeaders(dataset.getHeaders()); // Ensure headers are set

            StringBuilder stats = new StringBuilder();
            stats.append("Dataset Loaded Successfully!\n");
            stats.append("Type: ").append(dataset.getDataType()).append("\n");

            if (dataset.getDataType() == Dataset.DataType.IMAGE) {
                stats.append("Total Images: ").append(dataset.getFilePaths().size()).append("\n");
                // Infer classes from folders?
                if (dataset.getLabelVector() != null) {
                    long uniqueLabels = Arrays.stream(dataset.getLabelVector()).distinct().count();
                    stats.append("Classes Found: ").append(uniqueLabels).append("\n");
                }
            } else {
                stats.append("Rows: ").append(dataset.getFeatures().length).append("\n");
                stats.append("Columns: ").append(dataset.getFeatures().length > 0 ? dataset.getFeatures()[0].length : 0)
                        .append("\n");
            }

            analysisTextArea.setText(stats.toString());
            populateColumnList(dataset);
        });

        loadTask.setOnFailed(e -> {
            analysisTextArea.appendText("Error loading dataset: " + loadTask.getException().getMessage() + "\n");
            loadTask.getException().printStackTrace();
        });

        new Thread(loadTask).start();
    }

    private void populateColumnList(Dataset dataset) {
        if (dataset.getHeaders() != null && !dataset.getHeaders().isEmpty()) {
            targetColumnChoiceBox.setItems(FXCollections.observableArrayList(dataset.getHeaders()));
            targetColumnChoiceBox.setDisable(false);
            // Default to last
            targetColumnChoiceBox.setValue(dataset.getHeaders().get(dataset.getHeaders().size() - 1));
        } else {
            targetColumnChoiceBox.setDisable(true);
        }
    }

    private Dataset loadImagesFromFolder(File folder) {
        List<String> filePaths = new ArrayList<>();
        List<Integer> labels = new ArrayList<>();
        Map<String, Integer> labelMap = new HashMap<>();
        int labelCounter = 0;

        // Recursive scan (simplified for now: assume Class/Image structure or just
        // flat)
        // Heuristic: If subfolders exist, use them as labels.
        File[] children = folder.listFiles(File::isDirectory);

        if (children != null && children.length > 0) {
            // Labeled folders
            for (File classFolder : children) {
                String className = classFolder.getName();
                labelMap.putIfAbsent(className, labelCounter++);
                int currentLabel = labelMap.get(className);

                File[] images = classFolder.listFiles(f -> isImageFile(f));
                if (images != null) {
                    for (File img : images) {
                        filePaths.add(img.getAbsolutePath());
                        labels.add(currentLabel);
                    }
                }
            }
        } else {
            // Flat folder (Unsupervised or inference only?) -> Label 0
            File[] images = folder.listFiles(f -> isImageFile(f));
            if (images != null) {
                for (File img : images) {
                    filePaths.add(img.getAbsolutePath());
                    labels.add(0);
                }
            }
        }

        int[] labelArray = labels.stream().mapToInt(i -> i).toArray();
        // Map AppStateService.DataType to Dataset.DataType
        Dataset.DataType dsType = Dataset.DataType.TABULAR;
        if (appStateService.getSelectedDataType() == AppStateService.DataType.IMAGE) {
            dsType = Dataset.DataType.IMAGE;
        } else if (appStateService.getSelectedDataType() == AppStateService.DataType.TEXT) {
            dsType = Dataset.DataType.TEXT;
        }

        // Return dataset with EMPTY features but Valid Paths
        return new Dataset(folder, new double[0][0], labelArray, filePaths, dsType);
    }

    private boolean isImageFile(File f) {
        String name = f.getName().toLowerCase();
        return name.endsWith(".jpg") || name.endsWith(".jpeg") || name.endsWith(".png") || name.endsWith(".bmp");
    }

    // A helper to sync the UI with the AppStateService
    private void updateViewFromState() {
        File currentFile = appStateService.getSelectedDatasetFile();
        List<String> headers = appStateService.getDatasetHeaders();

        if (currentFile != null) {
            if (headers != null && !headers.isEmpty()) {
                selectedFileLabel.setText("Selected: " + currentFile.getName());
                selectedFileLabel.setStyle(""); // Reset style
                analyzeButton.setDisable(false);

                targetColumnChoiceBox.setItems(FXCollections.observableArrayList(headers));
                targetColumnChoiceBox.setDisable(false);

                // Set selected item if it's already in the list
                String currentTarget = appStateService.getTargetColumnName();
                if (currentTarget != null && headers.contains(currentTarget)) {
                    targetColumnChoiceBox.setValue(currentTarget);
                } else {
                    // Default to the last column
                    targetColumnChoiceBox.setValue(headers.get(headers.size() - 1));
                }
            } else {
                selectedFileLabel.setText("Invalid File: " + currentFile.getName());
                selectedFileLabel.setStyle("-fx-text-fill: red;");
                analyzeButton.setDisable(true);
                targetColumnChoiceBox.getItems().clear();
                targetColumnChoiceBox.setDisable(true);
            }
        } else {
            selectedFileLabel.setText("No file selected.");
            selectedFileLabel.setStyle(""); // Reset style
            analyzeButton.setDisable(true);
            targetColumnChoiceBox.getItems().clear();
            targetColumnChoiceBox.setDisable(true);
        }
    }

    @FXML
    private void handleGenerateSynthetic() {
        try {
            // Generate synthetic dataset: 500 samples, 20 features, 10 informative
            Dataset syntheticDataset = SyntheticDataGenerator.generate(500, 20, 10, false);

            // Store in app state
            appStateService.setCurrentDataset(syntheticDataset);
            appStateService.setDatasetHeaders(syntheticDataset.getHeaders());
            appStateService.setTargetColumnName("Class");

            // Update UI
            selectedFileLabel.setText("Synthetic Dataset (500 x 20)");
            selectedFileLabel.setStyle("-fx-text-fill: #4CAF50;");

            targetColumnChoiceBox.setItems(FXCollections.observableArrayList(syntheticDataset.getHeaders()));
            targetColumnChoiceBox.setValue("Class");
            targetColumnChoiceBox.setDisable(false);
            analyzeButton.setDisable(false);

            analysisTextArea.setText("✅ Synthetic Dataset Generated Successfully!\n\n" +
                    "Samples: 500\n" +
                    "Features: 20 (10 informative, 10 noise)\n" +
                    "Target: Class (binary: 0 or 1)\n\n" +
                    "This dataset is ideal for testing the pipeline quickly.\n" +
                    "You can proceed to Algorithm Selection.");

            com.example.automl_prototype_1.service.LogService.info("UploadView",
                    "Generated synthetic dataset: 500 samples, 20 features");

        } catch (Exception e) {
            showError("Failed to generate synthetic data: " + e.getMessage());
            com.example.automl_prototype_1.service.LogService.error("UploadView",
                    "Synthetic generation failed: " + e.getMessage());
        }
    }

    private void showError(String message) {
        Alert alert = new Alert(Alert.AlertType.ERROR);
        alert.setTitle("Error");
        alert.setHeaderText(null);
        alert.setContentText(message);
        alert.showAndWait();
    }
}
