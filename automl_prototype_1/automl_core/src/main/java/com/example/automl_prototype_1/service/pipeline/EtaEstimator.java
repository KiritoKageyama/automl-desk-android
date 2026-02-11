package com.example.automl_prototype_1.service.pipeline;

import java.util.ArrayDeque;
import java.util.Deque;

public class EtaEstimator {
    private final Deque<Long> samples = new ArrayDeque<>();
    private final int windowSize = 50; // Use last 50 iterations for a stable average

    public synchronized void addSample(long iterMs) {
        samples.addLast(iterMs);
        if (samples.size() > windowSize) {
            samples.removeFirst();
        }
    }

    public synchronized long estimateMs(int remainingIters) {
        if (samples.isEmpty()) {
            return -1; // Indicates not enough data yet
        }
        double avg = samples.stream().mapToLong(val -> val).average().orElse(0.0);
        return (long) (avg * remainingIters);
    }
}