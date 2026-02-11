package com.example.automl_prototype_1.service.progress;

/**
 * Data Transfer Object for progress updates.
 * Immutable to ensure thread safety when passed to listeners.
 */
public class ProgressUpdate {
    private final String algorithmName;
    private final int currentTrial;
    private final int totalTrials;
    private final int currentIteration;
    private final int maxIterations;
    private final double bestFitness;
    private final long timeElapsedMs;
    private final String message;

    public ProgressUpdate(String algorithmName, int currentTrial, int totalTrials,
            int currentIteration, int maxIterations,
            double bestFitness, long timeElapsedMs, String message) {
        this.algorithmName = algorithmName;
        this.currentTrial = currentTrial;
        this.totalTrials = totalTrials;
        this.currentIteration = currentIteration;
        this.maxIterations = maxIterations;
        this.bestFitness = bestFitness;
        this.timeElapsedMs = timeElapsedMs;
        this.message = message;
    }

    public String getAlgorithmName() {
        return algorithmName;
    }

    public int getCurrentTrial() {
        return currentTrial;
    }

    public int getTotalTrials() {
        return totalTrials;
    }

    public int getCurrentIteration() {
        return currentIteration;
    }

    public int getMaxIterations() {
        return maxIterations;
    }

    public double getBestFitness() {
        return bestFitness;
    }

    public long getTimeElapsedMs() {
        return timeElapsedMs;
    }

    public String getMessage() {
        return message;
    }

    @Override
    public String toString() {
        return String.format("[%s] Trial %d/%d, Iter %d/%d, Best: %.4f, Time: %dms",
                algorithmName, currentTrial, totalTrials, currentIteration, maxIterations, bestFitness, timeElapsedMs);
    }
}
