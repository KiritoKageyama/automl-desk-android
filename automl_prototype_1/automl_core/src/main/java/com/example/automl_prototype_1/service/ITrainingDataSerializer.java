package com.example.automl_prototype_1.service;

/**
 * Strategy interface for training data serialization (JSON vs Arrow).
 * Phase 6 Step 4.
 */
public interface ITrainingDataSerializer {
    /**
     * Serializes training data for the Python server.
     * 
     * @param features Feature matrix
     * @param labels   Label array
     * @param params   Training parameters
     * @return SerializedData wrapper containing body and content type
     * @throws Exception if serialization fails
     */
    SerializedData serializeTrainingRequest(double[][] features, int[] labels, java.util.Map<String, Object> params)
            throws Exception;

    /**
     * Gets the protocol type name (e.g., "JSON", "ARROW").
     */
    String getProtocolName();
}
