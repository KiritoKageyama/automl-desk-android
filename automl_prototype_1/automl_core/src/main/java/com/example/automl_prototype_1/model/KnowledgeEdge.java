package com.example.automl_prototype_1.model;

import java.util.UUID;

/**
 * Represents a directed edge in the Knowledge Graph.
 * Connects two KnowledgeNodes with a typed relationship,
 * an optional trigger condition, and traversal metadata.
 *
 * Example edges:
 * Dataset --TRAINS_ON--> RF Classifier
 * AgenticAdvisor --TRIGGERS--> Algorithm Selection
 * BERT --FINE_TUNES--> DistilBERT Base
 * RL Agent --TRANSITIONS_TO--> Q-State[s3]
 */
public class KnowledgeEdge {

    public enum Relationship {
        TRAINS_ON, // Data → Model
        FEEDS_INTO, // Stage → Stage (pipeline flow)
        TRIGGERS, // Condition → Action
        LEARNED_FROM, // Agent → Experience source
        FINE_TUNES, // Child model → Base model
        TRANSITIONS_TO, // RL state → RL state
        EXPORTS_TO, // Model → ONNX artifact
        GUARDS, // Constitutional rule → Action
        RECOMMENDS, // Advisor → Model/Algorithm
        OBSERVES // Observer → Log source
    }

    private final String id;
    private final String sourceId;
    private final String targetId;
    private Relationship relationship;
    private String trigger; // What activates this edge, e.g. "accuracy < 0.8"
    private String description; // Human-readable traversal explanation
    private double weight; // Importance / strength (0.0 - 1.0)

    // Real-time traversal state
    private boolean active = false;
    private long lastActivatedMs = 0;

    public KnowledgeEdge(String sourceId, String targetId, Relationship relationship) {
        this.id = UUID.randomUUID().toString().substring(0, 8);
        this.sourceId = sourceId;
        this.targetId = targetId;
        this.relationship = relationship;
        this.trigger = "";
        this.description = "";
        this.weight = 0.5;
    }

    // --- Fluent Builder ---
    public KnowledgeEdge withTrigger(String trigger) {
        this.trigger = trigger;
        return this;
    }

    public KnowledgeEdge withDescription(String description) {
        this.description = description;
        return this;
    }

    public KnowledgeEdge withWeight(double weight) {
        this.weight = Math.max(0.0, Math.min(1.0, weight));
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

    public boolean isRecentlyActive(long windowMs) {
        return active || (System.currentTimeMillis() - lastActivatedMs) < windowMs;
    }

    // --- Getters ---
    public String getId() {
        return id;
    }

    public String getSourceId() {
        return sourceId;
    }

    public String getTargetId() {
        return targetId;
    }

    public Relationship getRelationship() {
        return relationship;
    }

    public void setRelationship(Relationship r) {
        this.relationship = r;
    }

    public String getTrigger() {
        return trigger;
    }

    public void setTrigger(String trigger) {
        this.trigger = trigger;
    }

    public String getDescription() {
        return description;
    }

    public void setDescription(String description) {
        this.description = description;
    }

    public double getWeight() {
        return weight;
    }

    public void setWeight(double weight) {
        this.weight = weight;
    }

    public boolean isActive() {
        return active;
    }

    public long getLastActivatedMs() {
        return lastActivatedMs;
    }

    @Override
    public String toString() {
        return sourceId + " --[" + relationship + "]--> " + targetId;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o)
            return true;
        if (o == null || getClass() != o.getClass())
            return false;
        return id.equals(((KnowledgeEdge) o).id);
    }

    @Override
    public int hashCode() {
        return id.hashCode();
    }
}
