package com.example.automl_prototype_1.classifier;

import com.aparapi.Kernel;
import com.aparapi.Range;
import com.example.automl_prototype_1.service.IClassifier;
import java.util.Arrays;
import java.util.stream.IntStream;

/**
 * A GPU-accelerated KNN Classifier using Aparapi.
 * Optimized for high-frequency calls (e.g. Feature Selection/GA).
 * 
 * Improvements:
 * - Persistent Kernel instance (avoids recompilation overhead).
 * - Parallel flattening of arrays (speeds up CPU prep).
 * - minimized memory allocation.
 */
public class GpuKnnClassifier implements IClassifier {

    private final int k;
    private static boolean gpuCircuitBreakerOpen = false; // Circuit breaker for GPU failures
    private double[][] trainFeatures;
    private int[] trainLabels;

    // Persistent Kernel to reuse OpenCL context
    private KnnKernel kernel;

    public GpuKnnClassifier(int k) {
        this.k = k;
    }

    public GpuKnnClassifier() {
        this(5);
    }

    @Override
    public String getName() {
        return "GPU-Accelerated KNN (Aparapi)";
    }

    @Override
    public void train(double[][] features, int[] labels) {
        this.trainFeatures = features;
        this.trainLabels = labels;
    }

    @Override
    public int[] predict(double[][] testFeatures) {
        if (trainFeatures == null || trainFeatures.length == 0 || testFeatures.length == 0) {
            return new int[testFeatures.length];
        }

        int numTrain = trainFeatures.length;
        int numTest = testFeatures.length;
        int numDims = trainFeatures[0].length;

        // Flatten arrays for Aparapi (Parallelized)
        final float[] flatTrain = flattenParallel(trainFeatures);
        final float[] flatTest = flattenParallel(testFeatures);
        final float[] distances = new float[numTest * numTrain];

        // Circuit Breaker Check
        if (gpuCircuitBreakerOpen) {
            calculateDistancesCpu(flatTrain, flatTest, numDims, distances);
        } else {
            try {
                // Initialize Kernel if null or explicitly forced update
                if (kernel == null) {
                    kernel = new KnnKernel();
                }

                // Update Kernel Data
                // Note: Aparapi detects modification of these array references and re-uploads
                // data automatically.
                kernel.updateData(flatTrain, flatTest, distances, numTrain, numTest, numDims);

                // Determine Target Device
                com.aparapi.device.Device selectedDevice = null;
                String selectedName = com.example.automl_prototype_1.service.AppStateService.getInstance()
                        .getSelectedGpuName();
                if (selectedName != null) {
                    selectedDevice = com.example.automl_prototype_1.service.HardwareAcceleratorService
                            .getDeviceByName(selectedName);
                }

                // Execute Kernel in 1D Mode with Explicit Local Size
                int totalWork = numTest * numTrain;
                int localSize = 256;
                // Pad global size to be multiple of localSize
                int globalSize = ((totalWork + localSize - 1) / localSize) * localSize;

                Range range;
                if (selectedDevice != null) {
                    range = selectedDevice.createRange(globalSize, localSize);
                } else {
                    range = Range.create(globalSize, localSize);
                }

                kernel.execute(range);

                // Explicitly read back distances (sometimes needed depending on mode, usually
                // auto)
                // distances = kernel.distances;

                // Note: We do NOT dispose the kernel here. We keep it for reuse.

            } catch (Throwable e) {
                com.example.automl_prototype_1.service.LogService.error("GPU-KNN",
                        "GPU Execution Failed: " + e.getMessage());
                com.example.automl_prototype_1.service.LogService.error("GPU-KNN",
                        "Tripping Circuit Breaker. Switching to CPU Fallback.");
                gpuCircuitBreakerOpen = true;
                calculateDistancesCpu(flatTrain, flatTest, numDims, distances);
                // If failed, try to dispose the corrupted kernel
                if (kernel != null) {
                    try {
                        kernel.dispose();
                    } catch (Exception ex) {
                    }
                    kernel = null;
                }
            }
        }

        // CPU Side: Find K-Nearest Neighbors from the distance matrix
        int[] predictions = new int[numTest];

        // Parallel processing for O(N) reduction instead of O(N log N) sort
        IntStream.range(0, numTest).parallel().forEach(t -> {
            int offset = t * numTrain;

            // Fixed-size arrays for top-k candidates (avoid object allocation)
            int[] bestIndices = new int[k];
            float[] bestDists = new float[k];
            Arrays.fill(bestDists, Float.MAX_VALUE);

            for (int i = 0; i < numTrain; i++) {
                float dist = distances[offset + i];
                // Find worst distance in current best-k (Since K is small ~5, linear scan is
                // fastest)
                int maxIdx = 0;
                float maxDist = bestDists[0];
                for (int j = 1; j < k; j++) {
                    if (bestDists[j] > maxDist) {
                        maxDist = bestDists[j];
                        maxIdx = j;
                    }
                }

                // Update if better
                if (dist < maxDist) {
                    bestDists[maxIdx] = dist;
                    bestIndices[maxIdx] = i;
                }
            }

            // Vote
            int[] buffer = new int[k];
            int validCount = 0;
            for (int i = 0; i < k; i++) {
                if (bestDists[i] != Float.MAX_VALUE) {
                    buffer[validCount++] = trainLabels[bestIndices[i]];
                }
            }
            // Resize buffer if needed (rare case numTrain < k)
            if (validCount < k)
                buffer = Arrays.copyOf(buffer, validCount);

            predictions[t] = majorityVote(buffer);
        });

        return predictions;
    }

    private int majorityVote(int[] labels) {
        if (labels.length == 0)
            return -1;
        if (labels.length == 1)
            return labels[0];

        // Optimized Majority Vote: Sort and Scan (Zero Allocation)
        // Since K is small (usually < 20), sorting is extremely fast (insertion sort
        // likely)
        Arrays.sort(labels);

        int maxLabel = labels[0];
        int maxCount = 1;
        int currentLabel = labels[0];
        int currentCount = 1;

        for (int i = 1; i < labels.length; i++) {
            if (labels[i] == currentLabel) {
                currentCount++;
            } else {
                if (currentCount > maxCount) {
                    maxCount = currentCount;
                    maxLabel = currentLabel;
                }
                currentLabel = labels[i];
                currentCount = 1;
            }
        }
        // Final check
        if (currentCount > maxCount) {
            maxLabel = currentLabel;
        }

        return maxLabel;
    }

    private float[] flattenParallel(double[][] matrix) {
        int rows = matrix.length;
        int cols = matrix[0].length;
        float[] flat = new float[rows * cols];

        // Parallel flatten: Each row is processed independently
        IntStream.range(0, rows).parallel().forEach(i -> {
            for (int j = 0; j < cols; j++) {
                flat[i * cols + j] = (float) matrix[i][j];
            }
        });
        return flat;
    }

    private void calculateDistancesCpu(float[] flatTrain, float[] flatTest, int numDims, float[] distances) {
        int numTest = flatTest.length / numDims;
        int numTrain = flatTrain.length / numDims;

        IntStream.range(0, numTest).parallel().forEach(testIdx -> {
            int testOffset = testIdx * numDims;
            for (int trainIdx = 0; trainIdx < numTrain; trainIdx++) {
                int trainOffset = trainIdx * numDims;
                float distSq = 0;
                for (int d = 0; d < numDims; d++) {
                    float diff = flatTest[testOffset + d] - flatTrain[trainOffset + d];
                    distSq += diff * diff;
                }
                distances[testIdx * numTrain + trainIdx] = distSq;
            }
        });
    }

    // Explicit Named Kernel Class
    public static class KnnKernel extends Kernel {

        float[] flatTrain;
        float[] flatTest;
        float[] distances;
        int numTrain;
        int numDims;
        int numTest;

        public void updateData(float[] train, float[] test, float[] dist, int nTrain, int nTest, int dims) {
            this.flatTrain = train;
            this.flatTest = test;
            this.distances = dist;
            this.numTrain = nTrain;
            this.numTest = nTest;
            this.numDims = dims;
        }

        @Override
        public void run() {
            // 1D Linearization
            int globalId = getGlobalId();
            int testIdx = globalId / numTrain;
            int trainIdx = globalId % numTrain;

            // Bounds check
            if (testIdx >= numTest || trainIdx >= numTrain)
                return;

            float distSq = 0;
            int testOffset = testIdx * numDims;
            int trainOffset = trainIdx * numDims;

            // Note: Loops with known bounds are better, but here numDims is dynamic
            for (int d = 0; d < numDims; d++) {
                float diff = flatTest[testOffset + d] - flatTrain[trainOffset + d];
                distSq += diff * diff;
            }

            distances[globalId] = distSq;
        }
    }
}
