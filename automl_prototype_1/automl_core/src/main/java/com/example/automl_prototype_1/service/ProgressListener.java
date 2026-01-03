package com.example.automl_prototype_1.service;

/**
 * Interface for listening to progress updates from the ProgressTrackingService.
 * Implement this interface to receive real-time updates for UI or logging.
 */
public interface ProgressListener {
    /**
     * Called when a progress update occurs (e.g., iteration complete).
     * 
     * @param update The progress update details.
     */
    void onProgress(ProgressUpdate update);

    /**
     * Called when a job (algorithm execution) is fully complete.
     * 
     * @param algorithmName The name of the completed algorithm.
     */
    void onComplete(String algorithmName);

    /**
     * Called when an error occurs during execution.
     * 
     * @param algorithmName The name of the algorithm.
     * @param errorMessage  The error message.
     */
    void onError(String algorithmName, String errorMessage);
}
