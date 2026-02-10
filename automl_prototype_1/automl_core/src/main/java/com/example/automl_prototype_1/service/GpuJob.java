package com.example.automl_prototype_1.service;

import java.util.concurrent.CompletableFuture;

public class GpuJob {
    public enum Type {
        TRAIN, PREDICT
    }

    public String id;
    public Type type;
    public String algoType; // "RF", "SVM", etc.
    public double[][] features;
    public int[] labels;
    public java.util.Map<String, Object> params;
    public String modelId; // For Prediction
    public CompletableFuture<Object> future;

    public GpuJob(Type type) {
        this.id = java.util.UUID.randomUUID().toString();
        this.type = type;
        this.future = new CompletableFuture<>();
    }
}
