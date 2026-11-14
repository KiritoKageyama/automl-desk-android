package com.example.automl_prototype_1.service.agentic;

import com.example.automl_prototype_1.model.KnowledgeNode;
import com.example.automl_prototype_1.model.KnowledgeNode.NodeType;
import com.example.automl_prototype_1.model.KnowledgeEdge;
import com.example.automl_prototype_1.model.KnowledgeEdge.Relationship;
import com.example.automl_prototype_1.registry.ClassifierRegistry;
import com.example.automl_prototype_1.service.LogService;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.stream.Collectors;

/**
 * Service to store "Knowledge" acquired by the agent.
 * Phase 10: Extended to a full graph structure (nodes + edges)
 * powering the interactive Knowledge Graph Visualizer in Neural Forge.
 *
 * Backward-compatible: addInsight() / getAllInsights() still work.
 */
public class KnowledgeBaseService {

    private static KnowledgeBaseService instance;

    // Legacy flat list (backward compat for RlVisualizerController etc.)
    private final List<Insight> insights;

    // Phase 10: Graph Structure
    private final Map<String, KnowledgeNode> nodes;
    private final List<KnowledgeEdge> edges;
    private final List<Runnable> changeListeners;

    private KnowledgeBaseService() {
        this.insights = new CopyOnWriteArrayList<>();
        this.nodes = new ConcurrentHashMap<>();
        this.edges = new CopyOnWriteArrayList<>();
        this.changeListeners = new CopyOnWriteArrayList<>();
        // Seed default
        addInsight("General", "If you crash, try not to do that again.");
    }

    public static synchronized KnowledgeBaseService getInstance() {
        if (instance == null) {
            instance = new KnowledgeBaseService();
        }
        return instance;
    }

    // =========================================================================
    // Legacy Insight API (backward compat)
    // =========================================================================

    public void addInsight(String category, String text) {
        insights.add(new Insight(category, text));
        LogService.info("KnowledgeBase", "New Insight: [" + category + "] " + text);

        // Also create a graph node for this insight
        KnowledgeNode node = new KnowledgeNode(
                text.length() > 40 ? text.substring(0, 40) + "..." : text,
                NodeType.KNOWLEDGE_INSIGHT,
                text).withMeta("category", category);
        addNode(node);
    }

    public List<Insight> getAllInsights() {
        return new ArrayList<>(insights);
    }

    public void updateInsight(int index, String newText) {
        if (index >= 0 && index < insights.size()) {
            insights.get(index).setText(newText);
            LogService.info("KnowledgeBase", "Insight updated at index " + index);
        }
    }

    // =========================================================================
    // Phase 10: Graph API
    // =========================================================================

    public void addNode(KnowledgeNode node) {
        nodes.put(node.getId(), node);
        notifyChange();
    }

    public void addEdge(KnowledgeEdge edge) {
        edges.add(edge);
        notifyChange();
    }

    public KnowledgeNode getNode(String id) {
        return nodes.get(id);
    }

    public List<KnowledgeNode> getAllNodes() {
        return new ArrayList<>(nodes.values());
    }

    public List<KnowledgeEdge> getAllEdges() {
        return new ArrayList<>(edges);
    }

    public List<KnowledgeEdge> getEdgesFrom(String nodeId) {
        return edges.stream()
                .filter(e -> e.getSourceId().equals(nodeId))
                .collect(Collectors.toList());
    }

    public List<KnowledgeEdge> getEdgesTo(String nodeId) {
        return edges.stream()
                .filter(e -> e.getTargetId().equals(nodeId))
                .collect(Collectors.toList());
    }

    /**
     * Get all edges connected to a node (both directions).
     */
    public List<KnowledgeEdge> getEdgesFor(String nodeId) {
        return edges.stream()
                .filter(e -> e.getSourceId().equals(nodeId) || e.getTargetId().equals(nodeId))
                .collect(Collectors.toList());
    }

    /**
     * Find a node by label (case-insensitive partial match).
     */
    public KnowledgeNode findNodeByLabel(String label) {
        return nodes.values().stream()
                .filter(n -> n.getLabel().equalsIgnoreCase(label))
                .findFirst().orElse(null);
    }

    /**
     * Activate a node and its outgoing edges for real-time visualization.
     */
    public void activateNode(String nodeId) {
        KnowledgeNode node = nodes.get(nodeId);
        if (node != null) {
            node.activate();
            // Also activate outgoing edges
            getEdgesFrom(nodeId).forEach(KnowledgeEdge::activate);
            notifyChange();
        }
    }

    /**
     * Deactivate all nodes and edges (reset glow state).
     */
    public void deactivateAll() {
        nodes.values().forEach(KnowledgeNode::deactivate);
        edges.forEach(KnowledgeEdge::deactivate);
        notifyChange();
    }

    /**
     * Register a UI listener to be notified on graph changes.
     */
    public void onChange(Runnable listener) {
        changeListeners.add(listener);
    }

    private void notifyChange() {
        for (Runnable listener : changeListeners) {
            try {
                listener.run();
            } catch (Exception e) {
                // Don't let UI errors crash the service
            }
        }
    }

    /**
     * Phase 10: Auto-populate the graph from the current application state.
     * Scans all registered classifiers, known services, and insights
     * to build a complete knowledge graph.
     */
    public void buildFromCurrentState() {
        nodes.clear();
        edges.clear();

        LogService.info("KnowledgeBase", "Building Knowledge Graph from current state...");

        // --- Data Sources ---
        KnowledgeNode csvSource = new KnowledgeNode("CSV Data", NodeType.DATA_SOURCE,
                "Tabular datasets loaded via CsvDataProvider")
                .withMeta("formats", "CSV, TSV");
        addNode(csvSource);

        KnowledgeNode imgSource = new KnowledgeNode("Image Data", NodeType.DATA_SOURCE,
                "Image datasets loaded from directory structure (ClassA/img.jpg)")
                .withMeta("formats", "JPG, PNG, BMP, GIF");
        addNode(imgSource);

        // --- Pipeline Stages ---
        KnowledgeNode preprocessing = new KnowledgeNode("Preprocessing", NodeType.PIPELINE_STAGE,
                "StandardScaler normalization, missing value imputation, categorical encoding");
        addNode(preprocessing);

        KnowledgeNode featureSelection = new KnowledgeNode("Feature Selection", NodeType.PIPELINE_STAGE,
                "Meta-heuristic algorithms: WWO, GA, BPSO, IGPSO, PLS, ASO, Grey Wolf")
                .withMeta("algorithms", 7);
        addNode(featureSelection);

        KnowledgeNode training = new KnowledgeNode("Model Training", NodeType.PIPELINE_STAGE,
                "GPU-accelerated training via gpu_server.py with Arrow/MMAP transfer");
        addNode(training);

        KnowledgeNode onnxExport = new KnowledgeNode("ONNX Export", NodeType.ONNX_EXPORT,
                "Models exported to ONNX format for cross-platform inference");
        addNode(onnxExport);

        // Pipeline flow edges
        addEdge(new KnowledgeEdge(csvSource.getId(), preprocessing.getId(), Relationship.FEEDS_INTO)
                .withDescription("Raw CSV data flows into preprocessing stage")
                .withTrigger("Dataset loaded by user"));
        addEdge(new KnowledgeEdge(imgSource.getId(), training.getId(), Relationship.FEEDS_INTO)
                .withDescription("Image paths sent directly to CNN/ViT training (skip tabular preprocessing)")
                .withTrigger("Image dataset detected"));
        addEdge(new KnowledgeEdge(preprocessing.getId(), featureSelection.getId(), Relationship.FEEDS_INTO)
                .withDescription("Normalized features passed to meta-heuristic feature selection"));
        addEdge(new KnowledgeEdge(featureSelection.getId(), training.getId(), Relationship.FEEDS_INTO)
                .withDescription("Selected feature subset sent to classifier training"));
        addEdge(new KnowledgeEdge(training.getId(), onnxExport.getId(), Relationship.EXPORTS_TO)
                .withDescription("Trained model serialized to ONNX via skl2onnx/onnxmltools"));

        // --- ML Classifiers (from ClassifierRegistry) ---
        List<String> classifierNames = ClassifierRegistry.getInstance().getClassifierNames();
        for (String name : classifierNames) {
            NodeType type = name.contains("GPU") ? NodeType.ML_MODEL : NodeType.ML_MODEL;
            String desc = name.contains("GPU") ? "GPU-accelerated via gpu_server.py" : "CPU classifier";
            if (name.contains("CNN"))
                desc = "Convolutional Neural Network for image classification";
            if (name.contains("BERT"))
                desc = "DistilBERT transformer for text classification";
            if (name.contains("LSTM"))
                desc = "Long Short-Term Memory for time series / sequential data";
            if (name.contains("MLP") || name.contains("ANN"))
                desc = "Multilayer Perceptron neural network";

            KnowledgeNode modelNode = new KnowledgeNode(name, NodeType.ML_MODEL, desc)
                    .withMeta("gpu", name.contains("GPU") || name.contains("Python"));
            addNode(modelNode);

            // Training edge
            addEdge(new KnowledgeEdge(training.getId(), modelNode.getId(), Relationship.TRAINS_ON)
                    .withDescription("Training stage dispatches to " + name)
                    .withTrigger("User selects " + name + " or AgenticAdvisor recommends it"));

            // ONNX export edge
            addEdge(new KnowledgeEdge(modelNode.getId(), onnxExport.getId(), Relationship.EXPORTS_TO)
                    .withDescription(name + " exported to ONNX for inference")
                    .withWeight(0.7));
        }

        // --- Agentic / LLM Layer ---
        KnowledgeNode ollamaNode = new KnowledgeNode("Ollama LLM", NodeType.LLM_LAYER,
                "Local LLM via Ollama API. Dual-Brain: System 1 (fast heuristic) + System 2 (deep reasoning)")
                .withMeta("models", "Llama, Mistral, CodeLlama");
        addNode(ollamaNode);

        KnowledgeNode tinyLlamaNode = new KnowledgeNode("TinyLlama SLM", NodeType.SLM_COMPONENT,
                "Small Language Model running on gpu_server.py for lightweight reasoning and chat")
                .withMeta("parameters", "1.1B");
        addNode(tinyLlamaNode);

        KnowledgeNode advisorNode = new KnowledgeNode("Agentic Advisor", NodeType.GENAI_MODULE,
                "Recommends optimal algorithm based on dataset meta-features and LLM analysis");
        addNode(advisorNode);

        KnowledgeNode architectNode = new KnowledgeNode("Agentic Architect", NodeType.GENAI_MODULE,
                "LLM-powered code generation: generates custom classifiers and compiles at runtime");
        addNode(architectNode);

        KnowledgeNode constitutionNode = new KnowledgeNode("Constitutional AI", NodeType.CONSTITUTIONAL_RULE,
                "7 safety rules governing LLM outputs. Strict mode blocks non-compliant responses")
                .withMeta("rules", 7);
        addNode(constitutionNode);

        // LLM edges
        addEdge(new KnowledgeEdge(advisorNode.getId(), ollamaNode.getId(), Relationship.FEEDS_INTO)
                .withDescription("Advisor queries Ollama for algorithm recommendations")
                .withTrigger("User clicks 'Get AI Recommendation'"));
        addEdge(new KnowledgeEdge(ollamaNode.getId(), constitutionNode.getId(), Relationship.GUARDS)
                .withDescription("All LLM outputs filtered through Constitutional AI safety rules")
                .withTrigger("Every LLM response"));
        addEdge(new KnowledgeEdge(architectNode.getId(), ollamaNode.getId(), Relationship.FEEDS_INTO)
                .withDescription("Architect prompts Ollama to generate classifier source code")
                .withTrigger("User submits prompt in Creator Lab"));
        addEdge(new KnowledgeEdge(advisorNode.getId(), training.getId(), Relationship.RECOMMENDS)
                .withDescription("Advisor recommendation influences which classifier is trained")
                .withTrigger("Recommendation accepted by user"));

        // --- RL Layer ---
        KnowledgeNode rlAgent = new KnowledgeNode("Q-Learning Agent", NodeType.RL_STATE,
                "Tabular Q-Learning with ε-greedy exploration, Q-table persistence, prior knowledge injection")
                .withMeta("algorithm", "Q-Learning")
                .withMeta("features", "behavioral_cloning, prior_knowledge");
        addNode(rlAgent);

        KnowledgeNode gridWorld = new KnowledgeNode("GridWorld Env", NodeType.RL_STATE,
                "Test environment for RL training. Grid-based navigation with rewards");
        addNode(gridWorld);

        addEdge(new KnowledgeEdge(rlAgent.getId(), gridWorld.getId(), Relationship.TRANSITIONS_TO)
                .withDescription("Agent takes actions in GridWorld, receives rewards, updates Q-table")
                .withTrigger("Each episode step"));
        addEdge(new KnowledgeEdge(ollamaNode.getId(), rlAgent.getId(), Relationship.LEARNED_FROM)
                .withDescription("LLM provides prior knowledge (game strategies) to seed Q-table")
                .withTrigger("GameKnowledgeService.getControls()"));

        // --- Phase 10: Deep Learning Backbone Nodes ---
        KnowledgeNode cnnNode = new KnowledgeNode("CNN (PyTorch)", NodeType.DL_BACKBONE,
                "Convolutional Neural Network using PyTorch for image classification and feature extraction")
                .withMeta("framework", "PyTorch").withMeta("input", "Images");
        addNode(cnnNode);

        KnowledgeNode lstmNode = new KnowledgeNode("LSTM (PyTorch)", NodeType.DL_BACKBONE,
                "Long Short-Term Memory network for sequential / time-series classification")
                .withMeta("framework", "PyTorch").withMeta("input", "Sequences");
        addNode(lstmNode);

        KnowledgeNode distilBertNode = new KnowledgeNode("DistilBERT", NodeType.DL_BACKBONE,
                "Lightweight HuggingFace BERT transformer for text classification and NLP tasks")
                .withMeta("params", "66M").withMeta("source", "HuggingFace");
        addNode(distilBertNode);

        KnowledgeNode vitNode = new KnowledgeNode("ViT (HuggingFace)", NodeType.DL_BACKBONE,
                "Vision Transformer — patch-based image classification from HuggingFace")
                .withMeta("source", "HuggingFace").withMeta("resolution", "224x224");
        addNode(vitNode);

        // DL backbone edges
        addEdge(new KnowledgeEdge(training.getId(), cnnNode.getId(), Relationship.TRAINS_ON)
                .withDescription("GPU server trains CNN on image data").withTrigger("CNN workspace → Run Training"));
        addEdge(new KnowledgeEdge(training.getId(), lstmNode.getId(), Relationship.TRAINS_ON)
                .withDescription("DL workspace trains LSTM on sequence data")
                .withTrigger("DL workspace → Run Training"));
        addEdge(new KnowledgeEdge(training.getId(), distilBertNode.getId(), Relationship.TRAINS_ON)
                .withDescription("LLM workspace fine-tunes DistilBERT").withTrigger("LLM workspace → Fine-Tune"));
        addEdge(new KnowledgeEdge(training.getId(), vitNode.getId(), Relationship.TRAINS_ON)
                .withDescription("CNN workspace trains ViT on image data").withTrigger("CNN workspace → Run Training"));
        addEdge(new KnowledgeEdge(cnnNode.getId(), onnxExport.getId(), Relationship.EXPORTS_TO)
                .withDescription("CNN model exported to ONNX"));
        addEdge(new KnowledgeEdge(lstmNode.getId(), onnxExport.getId(), Relationship.EXPORTS_TO)
                .withDescription("LSTM model exported to ONNX"));
        addEdge(new KnowledgeEdge(tinyLlamaNode.getId(), distilBertNode.getId(), Relationship.LEARNED_FROM)
                .withDescription("TinyLlama SLM learns from fine-tuned DistilBERT representations"));
        addEdge(new KnowledgeEdge(tinyLlamaNode.getId(), vitNode.getId(), Relationship.LEARNED_FROM)
                .withDescription("TinyLlama SLM learns from ViT visual embeddings"));

        // --- Phase 10: LCA Architecture Cycle (5 nodes) ---
        KnowledgeNode sensorLayer = new KnowledgeNode("Sensor Layer", NodeType.LCA_LAYER,
                "Collects raw input: screen pixels (VisionStateDetector), keyboard/mouse (GlobalInputListener), CSV data")
                .withMeta("services", "VisionStateDetector, ScreenWatcherService, RealTimeDataService");
        addNode(sensorLayer);

        KnowledgeNode interpretLayer = new KnowledgeNode("Interpretation", NodeType.LCA_LAYER,
                "Parses raw sensory data into structured world state (AppStateService)")
                .withMeta("services", "AppStateService, MetaFeatureExtractor");
        addNode(interpretLayer);

        KnowledgeNode deliberationLayer = new KnowledgeNode("Deliberation", NodeType.LCA_LAYER,
                "Reasons over world state using LLM + Constitutional AI to choose best action")
                .withMeta("services", "AgenticAdvisor, OllamaService, ConstitutionalService");
        addNode(deliberationLayer);

        KnowledgeNode actionLayer = new KnowledgeNode("Action Layer", NodeType.LCA_LAYER,
                "Executes decisions: triggers training, bans algorithms, updates knowledge graph")
                .withMeta("services", "ExecutionService, AutoAblationService, KnowledgeBaseService");
        addNode(actionLayer);

        KnowledgeNode metaCogLayer = new KnowledgeNode("Meta-Cognition", NodeType.LCA_LAYER,
                "Reflects on results: SelfCorrectionLoopService, PerformanceHistoryService, ErrorMonitorService")
                .withMeta("services", "SelfCorrectionLoopService, PerformanceHistoryService");
        addNode(metaCogLayer);

        // LCA cycle edges
        addEdge(new KnowledgeEdge(sensorLayer.getId(), interpretLayer.getId(), Relationship.FEEDS_INTO)
                .withDescription("Raw sensory data → structured world state"));
        addEdge(new KnowledgeEdge(interpretLayer.getId(), deliberationLayer.getId(), Relationship.FEEDS_INTO)
                .withDescription("Structured state → LLM reasoning layer"));
        addEdge(new KnowledgeEdge(deliberationLayer.getId(), actionLayer.getId(), Relationship.TRIGGERS)
                .withDescription("Decision → action execution").withTrigger("Verdict from AgenticAdvisor"));
        addEdge(new KnowledgeEdge(actionLayer.getId(), metaCogLayer.getId(), Relationship.TRIGGERS)
                .withDescription("Action result → self-reflection").withTrigger("onExecutionComplete()"));
        addEdge(new KnowledgeEdge(metaCogLayer.getId(), sensorLayer.getId(), Relationship.TRIGGERS)
                .withDescription("Meta-cognition feeds corrective signals back to sensor layer (cycle)"));
        // Cross connections
        addEdge(new KnowledgeEdge(ollamaNode.getId(), deliberationLayer.getId(), Relationship.FEEDS_INTO)
                .withDescription("Ollama LLM provides reasoning backbone for deliberation layer"));
        addEdge(new KnowledgeEdge(metaCogLayer.getId(), constitutionNode.getId(), Relationship.GUARDS)
                .withDescription("Meta-cognition observes Constitutional AI for rule violations"));

        // --- Insights → Graph ---
        // Re-add existing insights as nodes
        for (Insight insight : insights) {
            // Skip if already added during construction
            boolean exists = nodes.values().stream()
                    .anyMatch(n -> n.getType() == NodeType.KNOWLEDGE_INSIGHT
                            && n.getDescription().equals(insight.getText()));
            if (!exists) {
                KnowledgeNode iNode = new KnowledgeNode(
                        insight.getText().length() > 40
                                ? insight.getText().substring(0, 40) + "..."
                                : insight.getText(),
                        NodeType.KNOWLEDGE_INSIGHT,
                        insight.getText()).withMeta("category", insight.getCategory());
                addNode(iNode);
            }
        }

        LogService.info("KnowledgeBase",
                "Graph built: " + nodes.size() + " nodes, " + edges.size() + " edges");
        notifyChange();
    }

    /**
     * Compute the shortest path description between two nodes (BFS).
     * Returns a human-readable traversal trace.
     */
    public String tracePathTo(String fromId, String toId) {
        if (fromId == null || toId == null)
            return "No path available.";
        if (fromId.equals(toId))
            return "Already at target node.";

        // BFS
        java.util.Queue<String> queue = new java.util.LinkedList<>();
        Map<String, String> parent = new java.util.HashMap<>();
        Map<String, KnowledgeEdge> parentEdge = new java.util.HashMap<>();
        queue.add(fromId);
        parent.put(fromId, null);

        while (!queue.isEmpty()) {
            String current = queue.poll();
            if (current.equals(toId))
                break;

            for (KnowledgeEdge edge : getEdgesFrom(current)) {
                if (!parent.containsKey(edge.getTargetId())) {
                    parent.put(edge.getTargetId(), current);
                    parentEdge.put(edge.getTargetId(), edge);
                    queue.add(edge.getTargetId());
                }
            }
        }

        if (!parent.containsKey(toId))
            return "No path found between these nodes.";

        // Reconstruct path
        List<String> path = new ArrayList<>();
        String cur = toId;
        while (cur != null) {
            KnowledgeNode n = getNode(cur);
            KnowledgeEdge e = parentEdge.get(cur);
            if (e != null) {
                path.add(0, n.getLabel() + " (via " + e.getRelationship() + ": " + e.getTrigger() + ")");
            } else {
                path.add(0, n.getLabel() + " [START]");
            }
            cur = parent.get(cur);
        }

        return String.join(" → ", path);
    }

    // =========================================================================
    // Phase 11 — Graph Editing API
    // =========================================================================

    /**
     * Remove a node and all edges that reference it.
     */
    public void removeNode(String nodeId) {
        nodes.remove(nodeId);
        edges.removeIf(e -> e.getSourceId().equals(nodeId) || e.getTargetId().equals(nodeId));
        notifyChange();
        LogService.info("KnowledgeBase", "Removed node: " + nodeId);
    }

    /**
     * Export the entire graph (nodes + edges) to a JSON file.
     * Uses Jackson ObjectMapper if available, otherwise falls back to manual JSON.
     */
    public void exportToJson(java.io.File file) {
        try {
            StringBuilder sb = new StringBuilder();
            sb.append("{\n  \"nodes\": [\n");
            List<KnowledgeNode> nodeList = getAllNodes();
            for (int i = 0; i < nodeList.size(); i++) {
                KnowledgeNode n = nodeList.get(i);
                sb.append(String.format(
                        "    {\"id\":\"%s\",\"label\":\"%s\",\"type\":\"%s\",\"description\":\"%s\",\"x\":%.2f,\"y\":%.2f}",
                        n.getId(),
                        n.getLabel().replace("\"", "'"),
                        n.getType().name(),
                        n.getDescription().replace("\"", "'").replace("\n", " "),
                        n.getX(), n.getY()));
                if (i < nodeList.size() - 1)
                    sb.append(",");
                sb.append("\n");
            }
            sb.append("  ],\n  \"edges\": [\n");
            List<KnowledgeEdge> edgeList = getAllEdges();
            for (int i = 0; i < edgeList.size(); i++) {
                KnowledgeEdge e = edgeList.get(i);
                sb.append(String.format(
                        "    {\"sourceId\":\"%s\",\"targetId\":\"%s\",\"relationship\":\"%s\",\"description\":\"%s\"}",
                        e.getSourceId(), e.getTargetId(),
                        e.getRelationship().name(),
                        e.getDescription().replace("\"", "'").replace("\n", " ")));
                if (i < edgeList.size() - 1)
                    sb.append(",");
                sb.append("\n");
            }
            sb.append("  ]\n}");
            java.nio.file.Files.writeString(file.toPath(), sb.toString());
            LogService.info("KnowledgeBase", "Graph exported to " + file.getAbsolutePath());
        } catch (Exception ex) {
            LogService.error("KnowledgeBase", "Export failed: " + ex.getMessage());
        }
    }

    /**
     * Import a graph from a JSON file previously created by exportToJson().
     * Parses manually (no external library dependency).
     */
    public void importFromJson(java.io.File file) {
        try {
            String json = java.nio.file.Files.readString(file.toPath());
            // Simple parsing: extract quoted values from each object
            nodes.clear();
            edges.clear();
            // Parse nodes: find all
            // {"id":"...","label":"...","type":"...","description":"...","x":...,"y":...}
            java.util.regex.Pattern nodePattern = java.util.regex.Pattern.compile(
                    "\\{\"id\":\"([^\"]+)\",\"label\":\"([^\"]+)\",\"type\":\"([^\"]+)\",\"description\":\"([^\"]*)\",\"x\":([\\d.\\-]+),\"y\":([\\d.\\-]+)\\}");
            java.util.regex.Matcher nm = nodePattern.matcher(json);
            while (nm.find()) {
                try {
                    NodeType type = NodeType.valueOf(nm.group(3));
                    KnowledgeNode n = new KnowledgeNode(nm.group(2), type, nm.group(4))
                            .at(Double.parseDouble(nm.group(5)), Double.parseDouble(nm.group(6)));
                    nodes.put(n.getId(), n);
                } catch (Exception ignore) {
                }
            }
            // Parse edges
            java.util.regex.Pattern edgePattern = java.util.regex.Pattern.compile(
                    "\\{\"sourceId\":\"([^\"]+)\",\"targetId\":\"([^\"]+)\",\"relationship\":\"([^\"]+)\",\"description\":\"([^\"]*)\"");
            java.util.regex.Matcher em = edgePattern.matcher(json);
            while (em.find()) {
                try {
                    Relationship rel = Relationship.valueOf(em.group(3));
                    KnowledgeEdge e = new KnowledgeEdge(em.group(1), em.group(2), rel)
                            .withDescription(em.group(4));
                    edges.add(e);
                } catch (Exception ignore) {
                }
            }
            notifyChange();
            LogService.info("KnowledgeBase", "Graph imported: " + nodes.size() + " nodes, " + edges.size() + " edges");
        } catch (Exception ex) {
            LogService.error("KnowledgeBase", "Import failed: " + ex.getMessage());
        }
    }

    // =========================================================================
    // Inner Class (legacy)
    // =========================================================================

    public static class Insight {
        private String category;
        private String text;

        public Insight(String category, String text) {
            this.category = category;
            this.text = text;
        }

        public String getCategory() {
            return category;
        }

        public String getText() {
            return text;
        }

        public void setText(String text) {
            this.text = text;
        }

        @Override
        public String toString() {
            return "[" + category + "] " + text;
        }
    }
}
