package com.example.automl_prototype_1.controller;

import javafx.fxml.FXML;
import javafx.scene.control.*;
import javafx.scene.image.ImageView;
import javafx.scene.image.WritableImage;
import javafx.scene.canvas.Canvas;
import javafx.scene.canvas.GraphicsContext;
import javafx.scene.layout.StackPane;
import javafx.scene.layout.VBox;
import javafx.scene.layout.HBox;
import javafx.scene.paint.Color;
import javafx.stage.FileChooser;
import java.awt.image.BufferedImage;
import javafx.embed.swing.SwingFXUtils;
import java.io.File;
import java.util.List;
import java.util.ArrayList;
import java.util.Map;
import java.util.stream.Collectors;
import java.util.Set;
import java.util.HashSet;
import com.example.automl_prototype_1.service.agentic.KnowledgeBaseService;
import com.example.automl_prototype_1.model.KnowledgeNode;
import com.example.automl_prototype_1.model.KnowledgeEdge;
import com.example.automl_prototype_1.model.KnowledgeNode.NodeType;
import com.example.automl_prototype_1.service.agentic.ScreenWatcherService;
import com.example.automl_prototype_1.gpu.PythonExecutionService;
import com.example.automl_prototype_1.service.LogService;
import javafx.animation.AnimationTimer;
import javafx.application.Platform;

public class NeuralForgeViewController {

    // --- Tab 1: Vision ---
    @FXML
    private ImageView visionView;
    @FXML
    private Canvas attentionOverlay;
    @FXML
    private Button btnStartVision;
    @FXML
    private Button btnStopVision;
    @FXML
    private CheckBox chkShowAttention;
    @FXML
    private TextArea visionLog;

    // --- Tab 2: Reasoning ---
    @FXML
    private TextArea txtChatHistory;
    @FXML
    private TextField txtChatInput;
    @FXML
    private Button btnSendChat;

    // --- Tab 3: Knowledge Graph ---
    @FXML
    private StackPane graphContainer;
    @FXML
    private VBox detailPanel;
    @FXML
    private Button btnRefreshGraph;
    @FXML
    private Button btnAutoLayout;
    @FXML
    private Button btnCenterView;
    @FXML
    private Button btnAddNode;
    @FXML
    private Button btnExportGraph;
    @FXML
    private Button btnImportGraph;
    @FXML
    private CheckBox chkLiveMode;
    @FXML
    private CheckBox chkShowPhysics;
    @FXML
    private ComboBox<String> filterType;
    @FXML
    private TextField graphSearchField;
    @FXML
    private Label lblGraphStatus;
    @FXML
    private Slider sldLayoutSpeed;
    // Node type filter toggles
    @FXML
    private ToggleButton filterML;
    @FXML
    private ToggleButton filterRL;
    @FXML
    private ToggleButton filterLLM;
    @FXML
    private ToggleButton filterLCA;
    @FXML
    private ToggleButton filterDL;
    @FXML
    private ToggleButton filterGenAI;
    @FXML
    private ToggleButton filterData;
    // Node detail panel
    @FXML
    private Label detailTitle;
    @FXML
    private Label detailType;
    @FXML
    private TextArea detailDescription;
    @FXML
    private ListView<String> detailConnections;
    @FXML
    private TextArea detailMetadata;
    @FXML
    private TextArea detailPathTrace;
    // Node edit sub-panel (visible on node click)
    @FXML
    private VBox nodeEditPanel;
    @FXML
    private TextField nodeLabelField;
    @FXML
    private TextArea nodeDescField;
    @FXML
    private ComboBox<String> nodeTypeCombo;
    @FXML
    private Button btnSaveNodeEdit;
    @FXML
    private Button btnDeleteNode;
    @FXML
    private Button btnPinNode;

    // --- Tab 4: Memory ---
    @FXML
    private TextArea txtMemoryInput;
    @FXML
    private Button btnRemember;
    @FXML
    private TextField txtMemoryQuery;
    @FXML
    private Button btnRecall;
    @FXML
    private ListView<String> listMemoryResults;

    private volatile boolean isRunning = false;
    private final ScreenWatcherService screenWatcher = new ScreenWatcherService();

    // Knowledge Graph
    private KnowledgeGraphCanvas graphCanvas;
    private AnimationTimer layoutTimer;
    private AnimationTimer liveTimer;
    private int layoutIterations = 0;

    @FXML
    public void initialize() {
        // Vision tab
        btnStartVision.setOnAction(e -> startVisionLoop());
        btnStopVision.setOnAction(e -> stopVisionLoop());

        // Memory tab
        btnRemember.setOnAction(e -> handleRemember());
        btnRecall.setOnAction(e -> handleRecall());

        // Knowledge Graph tab
        initKnowledgeGraph();
    }

    // =========================================================================
    // Knowledge Graph (Phase 10)
    // =========================================================================

    private void initKnowledgeGraph() {
        graphCanvas = new KnowledgeGraphCanvas();
        graphContainer.getChildren().add(graphCanvas);

        // Bind canvas size to container
        graphCanvas.widthProperty().bind(graphContainer.widthProperty());
        graphCanvas.heightProperty().bind(graphContainer.heightProperty());

        // Populate filter combo (legacy dropdown)
        filterType.getItems().addAll(
                "All Types",
                "ML Model", "RL State", "LLM Layer", "SLM Component",
                "GenAI Module", "Knowledge Insight", "Data Source",
                "Pipeline Stage", "ONNX Export", "Constitutional Rule",
                "LCA Layer", "DL Backbone", "Fine-Tune Job");
        filterType.setValue("All Types");
        filterType.setOnAction(e -> handleFilterChange());

        // Populate nodeType ComboBox for editing
        if (nodeTypeCombo != null) {
            for (NodeType t : NodeType.values())
                nodeTypeCombo.getItems().add(t.name());
        }

        // Wire toolbar buttons
        btnRefreshGraph.setOnAction(e -> handleRefreshGraph());
        btnAutoLayout.setOnAction(e -> handleAutoLayout());
        if (btnCenterView != null)
            btnCenterView.setOnAction(e -> graphCanvas.centerView());
        if (btnAddNode != null)
            btnAddNode.setOnAction(e -> handleAddNode());
        if (btnExportGraph != null)
            btnExportGraph.setOnAction(e -> handleExportGraph());
        if (btnImportGraph != null)
            btnImportGraph.setOnAction(e -> handleImportGraph());

        // Node edit sub-panel buttons
        if (btnSaveNodeEdit != null)
            btnSaveNodeEdit.setOnAction(e -> handleSaveNodeEdit());
        if (btnDeleteNode != null)
            btnDeleteNode.setOnAction(e -> handleDeleteNode());
        if (btnPinNode != null)
            btnPinNode.setOnAction(e -> handlePinNode());
        if (nodeEditPanel != null)
            nodeEditPanel.setVisible(false);

        // Search field — pan + select on Enter
        if (graphSearchField != null) {
            graphSearchField.setOnAction(e -> handleGraphSearch());
            graphSearchField.textProperty().addListener((obs, old, nw) -> {
                if (nw.length() >= 2)
                    handleGraphSearch();
            });
        }

        // Type filter toggle buttons
        ToggleButton[] typeToggles = { filterML, filterRL, filterLLM, filterLCA, filterDL, filterGenAI, filterData };
        for (ToggleButton tb : typeToggles)
            if (tb != null)
                tb.setOnAction(e -> handleToggleFilter());

        // Layout speed slider
        if (sldLayoutSpeed != null) {
            sldLayoutSpeed.setMin(0.25);
            sldLayoutSpeed.setMax(4.0);
            sldLayoutSpeed.setValue(1.0);
        }

        // Node click callback
        graphCanvas.setOnNodeSelected((node, edges) -> Platform.runLater(() -> {
            showNodeDetail(node, edges);
            populateEditPanel(node);
        }));

        // Edge click callback
        graphCanvas.setOnEdgeSelected((edge, endpoints) -> Platform.runLater(() -> showEdgeDetail(edge, endpoints)));

        // Live mode timer (10 FPS)
        liveTimer = new AnimationTimer() {
            private long lastUpdate = 0;

            @Override
            public void handle(long now) {
                if (now - lastUpdate > 100_000_000L) {
                    lastUpdate = now;
                    graphCanvas.render();
                }
            }
        };

        chkLiveMode.selectedProperty().addListener((obs, old, val) -> {
            if (val) {
                liveTimer.start();
                lblGraphStatus.setText("Live mode ON — nodes glow during model activity");
            } else {
                liveTimer.stop();
                lblGraphStatus.setText("Live mode OFF");
                graphCanvas.render();
            }
        });
    }

    // =========================================================================
    // Graph Edit Handlers
    // =========================================================================

    /** Populate the inline edit panel when a node is selected. */
    private void populateEditPanel(KnowledgeNode node) {
        if (nodeEditPanel == null)
            return;
        nodeEditPanel.setVisible(true);
        nodeLabelField.setText(node.getLabel());
        nodeDescField.setText(node.getDescription());
        if (nodeTypeCombo != null)
            nodeTypeCombo.setValue(node.getType().name());
        btnPinNode.setText(node.isPinned() ? "Unpin" : "Pin");
    }

    /** Save edited node label/description/type back to the graph. */
    @FXML
    public void handleSaveNodeEdit() {
        KnowledgeNode node = graphCanvas.getSelectedNode();
        if (node == null)
            return;
        node.setLabel(nodeLabelField.getText().trim());
        node.setDescription(nodeDescField.getText().trim());
        if (nodeTypeCombo != null && nodeTypeCombo.getValue() != null) {
            try {
                node.setType(NodeType.valueOf(nodeTypeCombo.getValue()));
            } catch (Exception ignored) {
            }
        }
        graphCanvas.render();
        lblGraphStatus.setText("Node updated: " + node.getLabel());
        LogService.info("NeuralForge", "Node edited: " + node.getLabel());
    }

    /** Delete selected node and all its edges. */
    @FXML
    public void handleDeleteNode() {
        KnowledgeNode node = graphCanvas.getSelectedNode();
        if (node == null)
            return;
        int removed = 0;
        try {
            KnowledgeBaseService kb = KnowledgeBaseService.getInstance();
            kb.removeNode(node.getId());
            // Re-render after removal
            List<KnowledgeNode> nodes = kb.getAllNodes();
            List<KnowledgeEdge> edges = kb.getAllEdges();
            graphCanvas.setData(nodes, edges);
            removed = 1;
        } catch (Exception e) {
            LogService.error("NeuralForge", "Delete node failed: " + e.getMessage());
        }
        if (nodeEditPanel != null)
            nodeEditPanel.setVisible(false);
        graphCanvas.render();
        lblGraphStatus.setText(removed > 0 ? "Node deleted." : "Nothing deleted.");
    }

    /** Toggle pin/unpin on the selected node. */
    @FXML
    public void handlePinNode() {
        KnowledgeNode node = graphCanvas.getSelectedNode();
        if (node == null)
            return;
        node.setPinned(!node.isPinned());
        if (btnPinNode != null)
            btnPinNode.setText(node.isPinned() ? "Unpin" : "Pin");
        graphCanvas.render();
    }

    /** Add a blank KNOWLEDGE_INSIGHT node at the canvas center. */
    @FXML
    public void handleAddNode() {
        KnowledgeNode newNode = new KnowledgeNode(
                "New Node", NodeType.KNOWLEDGE_INSIGHT, "User-created node")
                .at(graphContainer.getWidth() / 2, graphContainer.getHeight() / 2);
        KnowledgeBaseService.getInstance().addNode(newNode);
        graphCanvas.setData(
                KnowledgeBaseService.getInstance().getAllNodes(),
                KnowledgeBaseService.getInstance().getAllEdges());
        graphCanvas.selectNode(newNode);
        populateEditPanel(newNode);
        lblGraphStatus.setText("New node added — edit label in the panel.");
    }

    /** Export graph to a user-chosen JSON file. */
    @FXML
    public void handleExportGraph() {
        FileChooser fc = new FileChooser();
        fc.setTitle("Export Knowledge Graph");
        fc.getExtensionFilters().add(new FileChooser.ExtensionFilter("JSON Files", "*.json"));
        fc.setInitialFileName("automl_knowledge_graph.json");
        File f = fc.showSaveDialog(graphContainer.getScene().getWindow());
        if (f != null) {
            KnowledgeBaseService.getInstance().exportToJson(f);
            lblGraphStatus.setText("Graph exported → " + f.getName());
        }
    }

    /** Import graph from a user-chosen JSON file. */
    @FXML
    public void handleImportGraph() {
        FileChooser fc = new FileChooser();
        fc.setTitle("Import Knowledge Graph");
        fc.getExtensionFilters().add(new FileChooser.ExtensionFilter("JSON Files", "*.json"));
        File f = fc.showOpenDialog(graphContainer.getScene().getWindow());
        if (f != null) {
            KnowledgeBaseService.getInstance().importFromJson(f);
            graphCanvas.setData(
                    KnowledgeBaseService.getInstance().getAllNodes(),
                    KnowledgeBaseService.getInstance().getAllEdges());
            handleAutoLayout();
            lblGraphStatus.setText("Graph imported from " + f.getName());
        }
    }

    /** Search nodes by label prefix and pan+select the first match. */
    @FXML
    public void handleGraphSearch() {
        if (graphSearchField == null)
            return;
        String query = graphSearchField.getText().trim().toLowerCase();
        if (query.isEmpty())
            return;
        KnowledgeBaseService.getInstance().getAllNodes().stream()
                .filter(n -> n.getLabel().toLowerCase().contains(query))
                .findFirst()
                .ifPresentOrElse(found -> {
                    graphCanvas.panToNode(found);
                    graphCanvas.selectNode(found);
                    lblGraphStatus.setText("Found: " + found.getLabel());
                }, () -> lblGraphStatus.setText("No match: '" + query + "'"));
    }

    /** Handle node type filter toggle buttons. */
    @FXML
    public void handleToggleFilter() {
        Set<NodeType> activeTypes = new HashSet<>();
        if (filterML != null && filterML.isSelected())
            activeTypes.add(NodeType.ML_MODEL);
        if (filterRL != null && filterRL.isSelected())
            activeTypes.add(NodeType.RL_STATE);
        if (filterLLM != null && filterLLM.isSelected()) {
            activeTypes.add(NodeType.LLM_LAYER);
            activeTypes.add(NodeType.SLM_COMPONENT);
        }
        if (filterLCA != null && filterLCA.isSelected())
            activeTypes.add(NodeType.LCA_LAYER);
        if (filterDL != null && filterDL.isSelected())
            activeTypes.add(NodeType.DL_BACKBONE);
        if (filterGenAI != null && filterGenAI.isSelected()) {
            activeTypes.add(NodeType.GENAI_MODULE);
            activeTypes.add(NodeType.FINE_TUNE_JOB);
        }
        if (filterData != null && filterData.isSelected())
            activeTypes.add(NodeType.DATA_SOURCE);

        List<KnowledgeNode> all = KnowledgeBaseService.getInstance().getAllNodes();
        List<KnowledgeEdge> allEdges = KnowledgeBaseService.getInstance().getAllEdges();

        if (activeTypes.isEmpty()) {
            // no filter selected — show all
            graphCanvas.setData(all, allEdges);
        } else {
            List<KnowledgeNode> filtered = all.stream()
                    .filter(n -> activeTypes.contains(n.getType())).toList();
            Set<String> filteredIds = filtered.stream().map(KnowledgeNode::getId).collect(Collectors.toSet());
            List<KnowledgeEdge> filteredEdges = allEdges.stream()
                    .filter(e -> filteredIds.contains(e.getSourceId()) || filteredIds.contains(e.getTargetId()))
                    .toList();
            graphCanvas.setData(filtered, filteredEdges);
        }
        graphCanvas.render();
        lblGraphStatus
                .setText(activeTypes.isEmpty() ? "All types shown" : "Filtered: " + activeTypes.size() + " type(s)");
    }

    @FXML
    private void handleRefreshGraph() {
        KnowledgeBaseService kbs = KnowledgeBaseService.getInstance();
        kbs.buildFromCurrentState();

        List<KnowledgeNode> nodes = kbs.getAllNodes();
        List<KnowledgeEdge> edges = kbs.getAllEdges();

        graphCanvas.setData(nodes, edges);
        lblGraphStatus.setText("Graph: " + nodes.size() + " nodes, " + edges.size() + " edges");

        // Run auto-layout
        handleAutoLayout();
    }

    @FXML
    private void handleAutoLayout() {
        if (layoutTimer != null)
            layoutTimer.stop();

        layoutIterations = 0;
        layoutTimer = new AnimationTimer() {
            @Override
            public void handle(long now) {
                graphCanvas.layoutStep();
                graphCanvas.render();
                layoutIterations++;
                if (layoutIterations > 200) { // 200 iterations ≈ 3-4 seconds
                    this.stop();
                    lblGraphStatus.setText("Layout settled. " +
                            KnowledgeBaseService.getInstance().getAllNodes().size() + " nodes");
                }
            }
        };
        layoutTimer.start();
    }

    private void handleFilterChange() {
        String filter = filterType.getValue();
        KnowledgeBaseService kbs = KnowledgeBaseService.getInstance();

        List<KnowledgeNode> allNodes = kbs.getAllNodes();
        List<KnowledgeEdge> allEdges = kbs.getAllEdges();

        if (filter == null || filter.equals("All Types")) {
            graphCanvas.setData(allNodes, allEdges);
        } else {
            KnowledgeNode.NodeType targetType = mapFilterToType(filter);
            List<KnowledgeNode> filtered = allNodes.stream()
                    .filter(n -> n.getType() == targetType)
                    .toList();
            List<String> filteredIds = filtered.stream()
                    .map(KnowledgeNode::getId).toList();
            List<KnowledgeEdge> filteredEdges = allEdges.stream()
                    .filter(e -> filteredIds.contains(e.getSourceId())
                            || filteredIds.contains(e.getTargetId()))
                    .toList();
            graphCanvas.setData(filtered, filteredEdges);
        }
        graphCanvas.render();
    }

    private KnowledgeNode.NodeType mapFilterToType(String filter) {
        return switch (filter) {
            case "ML Model" -> KnowledgeNode.NodeType.ML_MODEL;
            case "RL State" -> KnowledgeNode.NodeType.RL_STATE;
            case "LLM Layer" -> KnowledgeNode.NodeType.LLM_LAYER;
            case "SLM Component" -> KnowledgeNode.NodeType.SLM_COMPONENT;
            case "GenAI Module" -> KnowledgeNode.NodeType.GENAI_MODULE;
            case "Knowledge Insight" -> KnowledgeNode.NodeType.KNOWLEDGE_INSIGHT;
            case "Data Source" -> KnowledgeNode.NodeType.DATA_SOURCE;
            case "Pipeline Stage" -> KnowledgeNode.NodeType.PIPELINE_STAGE;
            case "ONNX Export" -> KnowledgeNode.NodeType.ONNX_EXPORT;
            case "Constitutional Rule" -> KnowledgeNode.NodeType.CONSTITUTIONAL_RULE;
            default -> KnowledgeNode.NodeType.ML_MODEL;
        };
    }

    private void showNodeDetail(KnowledgeNode node, List<KnowledgeEdge> connectedEdges) {
        detailTitle.setText(node.getLabel());
        detailType.setText("Type: " + node.getType().name());
        detailDescription.setText(node.getDescription());

        // Metadata
        StringBuilder meta = new StringBuilder();
        for (Map.Entry<String, Object> entry : node.getMetadata().entrySet()) {
            meta.append(entry.getKey()).append(": ").append(entry.getValue()).append("\n");
        }
        if (node.isActive())
            meta.append("STATUS: ACTIVE (in use)\n");
        detailMetadata.setText(meta.toString());

        // Connections
        KnowledgeBaseService kbs = KnowledgeBaseService.getInstance();
        detailConnections.getItems().clear();
        for (KnowledgeEdge edge : connectedEdges) {
            KnowledgeNode other;
            String direction;
            if (edge.getSourceId().equals(node.getId())) {
                other = kbs.getNode(edge.getTargetId());
                direction = "→";
            } else {
                other = kbs.getNode(edge.getSourceId());
                direction = "←";
            }
            String otherLabel = other != null ? other.getLabel() : "?";
            String trigger = edge.getTrigger().isEmpty() ? "" : " [trigger: " + edge.getTrigger() + "]";
            detailConnections.getItems().add(
                    direction + " " + edge.getRelationship().name() + " " + otherLabel + trigger);
        }

        // Path trace — from first data source
        KnowledgeNode dataNode = kbs.getAllNodes().stream()
                .filter(n -> n.getType() == KnowledgeNode.NodeType.DATA_SOURCE)
                .findFirst().orElse(null);
        if (dataNode != null && !dataNode.getId().equals(node.getId())) {
            detailPathTrace.setText(kbs.tracePathTo(dataNode.getId(), node.getId()));
        } else {
            detailPathTrace.setText("This is a root/data source node.");
        }
    }

    private void showEdgeDetail(KnowledgeEdge edge, KnowledgeNode[] endpoints) {
        KnowledgeNode src = endpoints[0];
        KnowledgeNode tgt = endpoints[1];

        detailTitle.setText(edge.getRelationship().name());
        detailType.setText("Edge: " + src.getLabel() + " → " + tgt.getLabel());

        StringBuilder desc = new StringBuilder();
        desc.append("Relationship: ").append(edge.getRelationship().name()).append("\n\n");
        if (!edge.getDescription().isEmpty()) {
            desc.append("Description:\n").append(edge.getDescription()).append("\n\n");
        }
        if (!edge.getTrigger().isEmpty()) {
            desc.append("Trigger: ").append(edge.getTrigger()).append("\n\n");
        }
        desc.append("Weight: ").append(String.format("%.2f", edge.getWeight())).append("\n");
        if (edge.isActive())
            desc.append("\nSTATUS: ACTIVE (data flowing)\n");
        detailDescription.setText(desc.toString());

        // Connections: show both endpoint details
        detailConnections.getItems().clear();
        detailConnections.getItems().add("Source: " + src.getLabel() + " (" + src.getType() + ")");
        detailConnections.getItems().add("Target: " + tgt.getLabel() + " (" + tgt.getType() + ")");

        detailMetadata.setText("Source type: " + src.getType()
                + "\nTarget type: " + tgt.getType()
                + "\nWeight: " + edge.getWeight());

        // Traversal explanation
        detailPathTrace.setText("Data traverses from [" + src.getLabel() + "] to ["
                + tgt.getLabel() + "] via " + edge.getRelationship().name()
                + (edge.getTrigger().isEmpty() ? "" : "\nActivated when: " + edge.getTrigger()));
    }

    // =========================================================================
    // Memory Tab (existing)
    // =========================================================================

    @FXML
    private void handleRemember() {
        String text = txtMemoryInput.getText().trim();
        if (text.isEmpty())
            return;

        btnRemember.setDisable(true);
        new Thread(() -> {
            boolean success = PythonExecutionService.getInstance().storeMemory(text, null);
            Platform.runLater(() -> {
                btnRemember.setDisable(false);
                if (success) {
                    txtMemoryInput.clear();
                    Alert alert = new Alert(Alert.AlertType.INFORMATION, "Memory Stored!");
                    alert.show();
                } else {
                    Alert alert = new Alert(Alert.AlertType.ERROR, "Failed to store memory.");
                    alert.show();
                }
            });
        }).start();
    }

    @FXML
    private void handleRecall() {
        String query = txtMemoryQuery.getText().trim();
        if (query.isEmpty())
            return;

        btnRecall.setDisable(true);
        listMemoryResults.getItems().clear();

        new Thread(() -> {
            List<Map<String, Object>> results = PythonExecutionService.getInstance().queryMemory(query, 5);
            Platform.runLater(() -> {
                btnRecall.setDisable(false);
                if (results.isEmpty()) {
                    listMemoryResults.getItems().add("No matches found.");
                } else {
                    for (Map<String, Object> res : results) {
                        String content = (String) res.get("text");
                        Double score = (Double) res.get("score");
                        listMemoryResults.getItems().add(String.format("[%.2f] %s", score, content));
                    }
                }
            });
        }).start();
    }

    // =========================================================================
    // Chat / Reasoning Tab (existing)
    // =========================================================================

    @FXML
    public void handleChat() {
        String input = txtChatInput.getText().trim();
        if (input.isEmpty())
            return;

        txtChatHistory.appendText("You: " + input + "\n");
        txtChatInput.clear();
        btnSendChat.setDisable(true);

        new Thread(() -> {
            List<Map<String, Object>> memories = PythonExecutionService.getInstance().queryMemory(input, 2);
            StringBuilder context = new StringBuilder();
            if (!memories.isEmpty()) {
                context.append("\n[Context Recalled]:\n");
                for (Map<String, Object> m : memories) {
                    context.append("- ").append(m.get("text")).append("\n");
                }
                Platform.runLater(() -> txtChatHistory.appendText(context.toString()));
            }

            String system = "You are an intelligent AutoML Assistant. Use the context provided.";
            String fullPrompt = input + "\n" + context.toString();
            String response = PythonExecutionService.getInstance().chatWithSLM(system, fullPrompt);

            Platform.runLater(() -> {
                txtChatHistory.appendText("Agent: " + response + "\n\n");
                btnSendChat.setDisable(false);
            });
        }).start();
    }

    @FXML
    private void clearChat() {
        txtChatHistory.clear();
    }

    // =========================================================================
    // Vision Tab (existing)
    // =========================================================================

    private void startVisionLoop() {
        if (isRunning)
            return;
        isRunning = true;
        btnStartVision.setDisable(true);
        btnStopVision.setDisable(false);
        visionLog.appendText("Starting Vision Transformer Feed...\n");

        Thread visionThread = new Thread(() -> {
            while (isRunning) {
                try {
                    long startTime = System.currentTimeMillis();
                    BufferedImage bImg = screenWatcher.captureImage(224, 224);
                    if (bImg == null)
                        continue;

                    java.io.File tempFile = java.io.File.createTempFile("vit_frame", ".png");
                    javax.imageio.ImageIO.write(bImg, "png", tempFile);
                    String path = tempFile.getAbsolutePath();

                    Map<String, Object> result = PythonExecutionService.getInstance().extractFeatures(
                            List.of(path), "vit", chkShowAttention.isSelected());

                    Platform.runLater(() -> {
                        WritableImage fxImg = SwingFXUtils.toFXImage(bImg, null);
                        visionView.setImage(fxImg);
                        if (result.containsKey("attention_maps")) {
                            @SuppressWarnings("unchecked")
                            List<List<List<List<Double>>>> attn = (List<List<List<List<Double>>>>) result
                                    .get("attention_maps");
                            if (attn != null && !attn.isEmpty()) {
                                drawAttention(attn);
                            }
                        }
                        if (result.containsKey("error")) {
                            visionLog.appendText("Error: " + result.get("error") + "\n");
                        }
                    });

                    tempFile.delete();
                    long elapsed = System.currentTimeMillis() - startTime;
                    if (elapsed < 100)
                        Thread.sleep(100 - elapsed);
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }
        });
        visionThread.setDaemon(true);
        visionThread.start();
    }

    @FXML
    private void stopVisionLoop() {
        isRunning = false;
        btnStartVision.setDisable(false);
        btnStopVision.setDisable(true);
        visionLog.appendText("Vision Feed Stopped.\n");
    }

    @FXML
    private void clearVisionLog() {
        visionLog.clear();
    }

    private void drawAttention(List<List<List<List<Double>>>> attnBatch) {
        try {
            List<List<List<Double>>> batch0 = attnBatch.get(0);
            int seqLen = batch0.get(0).size();
            double[] clsAttention = new double[seqLen];

            for (List<List<Double>> head : batch0) {
                List<Double> clsRow = head.get(0);
                for (int i = 1; i < seqLen; i++) {
                    clsAttention[i] += clsRow.get(i);
                }
            }

            double max = 0;
            for (int i = 1; i < seqLen; i++)
                if (clsAttention[i] > max)
                    max = clsAttention[i];

            GraphicsContext gc = attentionOverlay.getGraphicsContext2D();
            gc.clearRect(0, 0, attentionOverlay.getWidth(), attentionOverlay.getHeight());

            int gridSize = 14;
            double tileW = attentionOverlay.getWidth() / gridSize;
            double tileH = attentionOverlay.getHeight() / gridSize;

            for (int i = 1; i < seqLen; i++) {
                int patchIdx = i - 1;
                int r = patchIdx / gridSize;
                int c = patchIdx % gridSize;
                double intensity = clsAttention[i] / max;
                gc.setFill(Color.rgb(255, 0, 0, intensity * 0.6));
                gc.fillRect(c * tileW, r * tileH, tileW, tileH);
            }
        } catch (Exception e) {
            // Silently handle visualization errors
        }
    }
}
