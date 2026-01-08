package com.example.automl_prototype_1.classifier;

import com.aparapi.Kernel;
import com.aparapi.Range;
import com.example.automl_prototype_1.service.IClassifier;

/**
 * A GPU-accelerated KNN Classifier using Aparapi.
 * Implementation Details:
 * - Offloads the distance calculation matrix to the GPU.
 * - CPU handles sorting and majority voting (due to complexity of sorting on
 * GPU without specialized kernels like Bitonic Sort).
 * - This demonstrates "Explicit GPU Offloading" of the O(N*M) heavy lifting.
 */
public class GpuKnnClassifier implements IClassifier {

    private final int k;
    private static boolean gpuCircuitBreakerOpen = false; // Circuit breaker for GPU failures
    private double[][] trainFeatures;
    private int[] trainLabels;

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

        // Flatten arrays for Aparapi (Aparapi works best with 1D primitive arrays)
        final float[] flatTrain = flatten(trainFeatures);
        final float[] flatTest = flatten(testFeatures);
        final float[] distances = new float[numTest * numTrain];

        // Define the Kernel
        Kernel kernel = new Kernel() {
            @Override
            public void run() {
                // 1D Linearization to avoid Local Size Exceeded errors on 2D ranges
                int globalId = getGlobalId();
                int testIdx = globalId / numTrain;
                int trainIdx = globalId % numTrain;

                // Bounds check (vital for 1D if padding occurs, though Aparapi usually handles
                // precise ranges)
                if (testIdx >= numTest || trainIdx >= numTrain)
                    return;

                // Calculate Euclidean distance between test[testIdx] and train[trainIdx]
                float distSq = 0;
                int testOffset = testIdx * numDims;
                int trainOffset = trainIdx * numDims;

                for (int d = 0; d < numDims; d++) {
                    float diff = flatTest[testOffset + d] - flatTrain[trainOffset + d];
                    distSq += diff * diff;
                }

                // Store result
                distances[globalId] = distSq;
            }
        };

        // Circuit Breaker Check
        if (gpuCircuitBreakerOpen) {
            calculateDistancesCpu(flatTrain, flatTest, numDims, distances);
        } else {
            try {
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
                    System.out.println(
                            "CORE: Targeting Specific GPU (1D Explicit): " + selectedDevice.getShortDescription());
                } else {
                    range = Range.create(globalSize, localSize);
                }

                kernel.execute(range);

                if (System.currentTimeMillis() % 1000 == 0) {
                    System.out.println("CORE: GPU KNN Mode: " + kernel.getTargetDevice().getType());
                }
            } catch (Exception e) {
                System.err.println("CORE: GPU Execution Failed. Exception follows:");
                e.printStackTrace();
                System.err.println("CORE: Tripping Circuit Breaker. Switching to CPU Fallback.");
                gpuCircuitBreakerOpen = true; // Trip the breaker
                calculateDistancesCpu(flatTrain, flatTest, numDims, distances);
            } finally {
                kernel.dispose();
            }
        }

        // CPU Side: Find K-Nearest Neighbors from the distance matrix
        int[] predictions = new int[numTest];

        // Parallel processing for O(N) reduction instead of O(N log N) sort
        java.util.stream.IntStream.range(0, numTest).parallel().forEach(t -> {
            int offset = t * numTrain;

            // Fixed-size arrays for top-k candidates (avoid object allocation)
            int[] bestIndices = new int[k];
            float[] bestDists = new float[k];
            java.util.Arrays.fill(bestDists, Float.MAX_VALUE);

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
                buffer = java.util.Arrays.copyOf(buffer, validCount);

            predictions[t] = majorityVote(buffer);
        });

        return predictions;
    }

    private int majorityVote(int[] labels) {
        if (labels.length == 0)
            return -1;
        if (labels.length == 1)
            return labels[0];

        // Simple majority vote
        // Map<Label, Count>
        java.util.Map<Integer, Integer> counts = new java.util.HashMap<>();
        int maxCount = -1;
        int maxLabel = -1;

        for (int label : labels) {
            int c = counts.getOrDefault(label, 0) + 1;
            counts.put(label, c);
            if (c > maxCount) {
                maxCount = c;
                maxLabel = label;
            }
        }
        return maxLabel;
    }

    private float[] flatten(double[][] matrix) {
        int rows = matrix.length;
        int cols = matrix[0].length;
        float[] flat = new float[rows * cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                flat[i * cols + j] = (float) matrix[i][j];
            }
        }
        return flat;
    }

    private void calculateDistancesCpu(float[] flatTrain, float[] flatTest, int numDims, float[] distances) {
        int numTest = flatTest.length / numDims;
        int numTrain = flatTrain.length / numDims;

        java.util.stream.IntStream.range(0, numTest).parallel().forEach(testIdx -> {
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
}
