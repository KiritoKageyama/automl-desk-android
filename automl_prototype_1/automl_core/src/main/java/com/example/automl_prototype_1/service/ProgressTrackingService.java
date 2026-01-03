package com.example.automl_prototype_1.service;

import com.example.automl_prototype_1.model.progress.*;
import java.util.List;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.atomic.AtomicReference;

/**
 * Singleton service for tracking algorithm execution progress.
 * Supports both polling (via snapshot) and reactive updates (via listeners).
 */
public class ProgressTrackingService {
    private static final ProgressTrackingService instance = new ProgressTrackingService();

    public static ProgressTrackingService getInstance() {
        return instance;
    }

    // Reactive Listeners
    private final List<ProgressListener> listeners = new CopyOnWriteArrayList<>();

    // Internal State for Polling (Legacy/Backup)

    private final ConcurrentHashMap<String, EtaEstimator> etaEstimators = new ConcurrentHashMap<>();
    private final AtomicReference<String> currentRunningAlgo = new AtomicReference<>();
    private final CopyOnWriteArrayList<String> algorithmQueue = new CopyOnWriteArrayList<>();
    private final ConcurrentHashMap<String, AlgoProgress> progressMap = new ConcurrentHashMap<>();

    // Global best tracking
    private final AtomicReference<String> globalBestAlgo = new AtomicReference<>("N/A");
    private final AtomicReference<Double> globalBestScore = new AtomicReference<>(Double.NEGATIVE_INFINITY);
    private final AtomicReference<Integer> globalBestTrial = new AtomicReference<>(0);
    private final AtomicReference<Integer> globalBestIter = new AtomicReference<>(0);

    private ProgressTrackingService() {
        reset();
    }

    // --- Listener Management ---

    public void addListener(ProgressListener listener) {
        if (listener != null)
            listeners.add(listener);
    }

    public void removeListener(ProgressListener listener) {
        listeners.remove(listener);
    }

    private void notifyProgress(ProgressUpdate update) {
        for (ProgressListener listener : listeners) {
            try {
                listener.onProgress(update);
            } catch (Exception e) {
                System.err.println("Error notifying listener: " + e.getMessage());
            }
        }
    }

    private void notifyComplete(String algoName) {
        for (ProgressListener listener : listeners) {
            try {
                listener.onComplete(algoName);
            } catch (Exception e) {
                System.err.println("Error notifying listener: " + e.getMessage());
            }
        }
    }

    // --- State Management ---

    public void reset() {
        this.algorithmQueue.clear();
        this.progressMap.clear();
        this.etaEstimators.clear();
        this.currentRunningAlgo.set(null);
        this.globalBestAlgo.set("N/A");
        this.globalBestScore.set(Double.NEGATIVE_INFINITY);
        globalBestTrial.set(0);
        globalBestIter.set(0);
    }

    public void reset(List<String> algoQueue) {
        reset();
        if (algoQueue != null) {
            this.algorithmQueue.addAll(algoQueue);
        }
    }

    public void jobStarted(String algoId, int trialNum, int totalTrials, int totalIters) {
        etaEstimators.put(algoId, new EtaEstimator());
        currentRunningAlgo.set(algoId);
        algorithmQueue.remove(algoId);

        AlgoProgress initialProgress = new AlgoProgress(
                algoId, Phase.RUNNING, trialNum, totalTrials, 0, totalIters,
                Double.NEGATIVE_INFINITY, 0, 0, -1L);
        progressMap.put(algoId, initialProgress);

        // Notify listeners of start (as a progress update with 0 progress)
        notifyProgress(new ProgressUpdate(algoId, trialNum, totalTrials, 0, totalIters, Double.NEGATIVE_INFINITY, 0,
                "Started"));
    }

    public void iterationComplete(String algoId, int trial, int iter, int totalIters, double currentBestScore,
            long iterTimeMs) {
        // 1. Update internal state (Legacy)
        progressMap.computeIfPresent(algoId, (key, oldProgress) -> {
            double newBestScore = oldProgress.bestScore();
            int newBestIter = oldProgress.bestIter();
            if (currentBestScore > newBestScore) {
                newBestScore = currentBestScore;
                newBestIter = iter;
            }

            EtaEstimator estimator = etaEstimators.get(key);
            if (estimator != null) {
                estimator.addSample(iterTimeMs);
            }
            long eta = estimator != null ? estimator.estimateMs(totalIters - iter) : -1L;

            return new AlgoProgress(
                    key, Phase.RUNNING, trial, oldProgress.totalTrials(),
                    iter, totalIters, newBestScore, oldProgress.bestTrial(), newBestIter, eta);
        });

        // 2. Update global best
        globalBestScore.getAndUpdate(currentGlobalBest -> {
            if (currentBestScore > currentGlobalBest) {
                globalBestAlgo.set(algoId);
                globalBestTrial.set(trial);
                globalBestIter.set(iter);
                return currentBestScore;
            }
            return currentGlobalBest;
        });

        // 3. Notify Listeners (Reactive)
        notifyProgress(new ProgressUpdate(
                algoId, trial, 1, iter, totalIters, currentBestScore, iterTimeMs, null));
    }

    public void jobFinished(String algoId) {
        currentRunningAlgo.set(null);
        progressMap.computeIfPresent(algoId, (key, oldProgress) -> new AlgoProgress(
                key, Phase.DONE, oldProgress.currentTrial(), oldProgress.totalTrials(),
                oldProgress.totalIters(), oldProgress.totalIters(),
                oldProgress.bestScore(), oldProgress.bestTrial(), oldProgress.bestIter(), 0L));
        notifyComplete(algoId);
    }

    public OrchestratorState snapshot() {
        String currentId = currentRunningAlgo.get();
        AlgoProgress currentRun = (currentId != null) ? progressMap.get(currentId) : null;

        List<AlgoProgress> queue = algorithmQueue.stream()
                .map(id -> new AlgoProgress(id, Phase.QUEUED, 0, 0, 0, 0, 0, 0, 0, -1L))
                .toList();

        return new OrchestratorState(
                currentRun,
                queue,
                globalBestAlgo.get(),
                globalBestScore.get(),
                globalBestTrial.get(),
                globalBestIter.get());
    }
}