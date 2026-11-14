package com.example.automl_prototype_1.model;

import java.util.HashMap;
import java.util.Map;
import java.util.UUID;

/**
 * Represents a single node in the Knowledge Graph.
 * Each node is a discrete knowledge unit — an ML model, RL state,
 * LLM layer, data source, or acquired insight.
 *
 * Used by KnowledgeBaseService for graph construction and by
 * KnowledgeGraphCanvas for interactive visualization.
 */
public class KnowledgeNode {

    public enum NodeType {
        ML_MODEL, // RF, SVM, XGBoost, CNN, BERT, etc.
        RL_STATE, // Q-table state, policy node
        LLM_LAYER, // OllamaService, TinyLlama, GPT layer
        SLM_COMPONENT, // Small Language Model component
        GENAI_MODULE, // Creator Lab, code generation
        KNOWLEDGE_INSIGHT, // Learned insight from KnowledgeBaseService
        DATA_SOURCE, // Dataset, CSV, image folder
        PIPELINE_STAGE, // Preprocessing, feature selection, training
        ONNX_EXPORT, // Exported model artifact
        CONSTITUTIONAL_RULE, // Safety rule from ConstitutionalService
        LCA_LAYER, // Layered Cognitive Agent architecture layer
        DL_BACKBONE, // CNN/LSTM/BERT feature extractors (PyTorch)
        FINE_TUNE_JOB // A training run that adapted a base model
    }

    private final String id;
    private String label;
    private NodeType type;
    private String description;
    private final Map<String, Object> metadata;

    // Layout coordinates for force-directed graph
    private double x;
    private double y;

    // Velocity for Verlet physics simulation
    private double vx = 0;
    private double vy = 0;

    // Mass — heavier hub nodes attract springs more gently
    // Set by KnowledgeGraphCanvas based on degree (edge count)
    private double mass = 1.0;

    // Pinned nodes are exempt from physics forces
    private boolean pinned = false;

    // Real-time traversal state
    private boolean active = false;
    private long lastActivatedMs = 0;

    public KnowledgeNode(String label, NodeType type, String description) {
        this.id = UUID.randomUUID().toString().substring(0, 8);
        this.label = label;
        this.type = type;
        this.description = description;
        this.metadata = new HashMap<>();
        // Random initial position for layout
        this.x = 100 + Math.random() * 600;
        this.y = 100 + Math.random() * 400;
    }

    // --- Fluent Metadata Builder ---
    public KnowledgeNode withMeta(String key, Object value) {
        this.metadata.put(key, value);
        return this;
    }

    /** Fluent position builder — positions node at specific canvas coords. */
    public KnowledgeNode at(double x, double y) {
        this.x = x;
        this.y = y;
        return this;
    }

    // --- Activation for Live Traversal ---
    public void activate() {
        this.active = true;
        this.lastActivatedMs = System.currentTimeMillis();
    }

    public void deactivate() {
        this.active = false;
    }

    /**
     * Returns true if this node was activated within the last `windowMs`
     * milliseconds.
     * Used for glow-fade animation in the canvas renderer.
     */
    public boolean isRecentlyActive(long windowMs) {
        return active || (System.currentTimeMillis() - lastActivatedMs) < windowMs;
    }

    // --- Getters & Setters ---
    public String getId() {
        return id;
    }

    public String getLabel() {
        return label;
    }

    public void setLabel(String label) {
        this.label = label;
    }

    public NodeType getType() {
        return type;
    }

    public void setType(NodeType type) {
        this.type = type;
    }

    public String getDescription() {
        return description;
    }

    public void setDescription(String description) {
        this.description = description;
    }

    public Map<String, Object> getMetadata() {
        return metadata;
    }

    public double getX() {
        return x;
    }

    public void setX(double x) {
        this.x = x;
    }

    public double getY() {
        return y;
    }

    public void setY(double y) {
        this.y = y;
    }

    public double getVx() {
        return vx;
    }

    public void setVx(double vx) {
        this.vx = vx;
    }

    public double getVy() {
        return vy;
    }

    public void setVy(double vy) {
        this.vy = vy;
    }

    public double getMass() {
        return mass;
    }

    public void setMass(double mass) {
        this.mass = mass;
    }

    public boolean isPinned() {
        return pinned;
    }

    public void setPinned(boolean pinned) {
        this.pinned = pinned;
    }

    public boolean isActive() {
        return active;
    }

    public long getLastActivatedMs() {
        return lastActivatedMs;
    }

    @Override
    public String toString() {
        return "[" + type + "] " + label;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o)
            return true;
        if (o == null || getClass() != o.getClass())
            return false;
        return id.equals(((KnowledgeNode) o).id);
    }

    @Override
    public int hashCode() {
        return id.hashCode();
    }
}
