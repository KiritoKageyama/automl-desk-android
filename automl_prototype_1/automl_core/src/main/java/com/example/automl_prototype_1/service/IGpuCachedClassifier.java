package com.example.automl_prototype_1.service;

public interface IGpuCachedClassifier {
    /**
     * Trains the model using a dataset already cached on the GPU server.
     * 
     * @param datasetId      The UUID of the cached dataset.
     * @param featureIndices The indices of features to use (columns).
     * @param rowIndices     The indices of rows to use (training set).
     */
    void trainCached(String datasetId, int[] featureIndices, int[] rowIndices);

    /**
     * Predicts using a dataset already cached on the GPU server.
     * 
     * @param datasetId      The UUID of the cached dataset.
     * @param featureIndices The indices of features to use (columns).
     * @param rowIndices     The indices of rows to use (test set).
     * @return Predictions.
     */
    int[] predictCached(String datasetId, int[] featureIndices, int[] rowIndices);

    // --- Async / Batching Support ---

    /**
     * Async version of trainCached. Returns Future wrapping the Model ID.
     */
    java.util.concurrent.CompletableFuture<String> submitTrainCachedAsync(String datasetId, int[] featureIndices,
            int[] rowIndices);

    /**
     * Async version of predictCached. Returns Future wrapping the predictions
     * int[].
     * Requires the modelId to be passed explicitly (since state might not be shared
     * in batch).
     */
    java.util.concurrent.CompletableFuture<int[]> submitPredictCachedAsync(String modelId, String datasetId,
            int[] featureIndices, int[] rowIndices);
}
