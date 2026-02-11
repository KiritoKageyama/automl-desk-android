package com.example.automl_prototype_1.classifier.gpu;

import com.aparapi.Kernel;
import com.aparapi.Range;
import com.example.automl_prototype_1.service.IClassifier;
import java.util.stream.IntStream;

/**
 * GPU-Accelerated Logistic Regression using Gradient Descent via Aparapi.
 * Includes Circuit Breaker and Auto-CPU Fallback for robustness.
 */
public class GpuLogisticRegression implements IClassifier {

    private final float learningRate;
    private final int epochs;
    private float[] weights;
    private float bias;
    private boolean isTrained = false;

    // Circuit Breaker to prevent repeated GPU failures
    private static boolean gpuCircuitBreakerOpen = false;

    public GpuLogisticRegression() {
        this(0.01f, 1000);
    }

    public GpuLogisticRegression(float learningRate, int epochs) {
        this.learningRate = learningRate;
        this.epochs = epochs;
    }

    @Override
    public String getName() {
        return "GPU-Logistic Regression (Aparapi)";
    }

    @Override
    public void train(double[][] features, int[] labels) {
        int numSamples = features.length;
        if (numSamples == 0)
            return;
        int numFeatures = features[0].length;

        // Initialize state
        this.weights = new float[numFeatures];
        this.bias = 0.0f;

        // Prepare data (Flattened for GPU/CPU common usage)
        final float[] flatFeatures = new float[numSamples * numFeatures];
        for (int i = 0; i < numSamples; i++) {
            for (int j = 0; j < numFeatures; j++) {
                flatFeatures[i * numFeatures + j] = (float) features[i][j];
            }
        }

        // Map labels
        int minLabel = Integer.MAX_VALUE;
        for (int l : labels)
            if (l < minLabel)
                minLabel = l;
        final float[] floatLabels = new float[numSamples];
        for (int i = 0; i < numSamples; i++) {
            floatLabels[i] = (labels[i] == minLabel) ? 0.0f : 1.0f;
        }

        if (gpuCircuitBreakerOpen) {
            trainCpu(flatFeatures, floatLabels, numSamples, numFeatures);
        } else {
            try {
                trainGpu(flatFeatures, floatLabels, numSamples, numFeatures);
            } catch (Throwable t) {
                System.err.println("CORE: GpuLogisticRegression Train Failed: " + t.getMessage());
                gpuCircuitBreakerOpen = true;
                trainCpu(flatFeatures, floatLabels, numSamples, numFeatures);
            }
        }
        isTrained = true;
    }

    private void trainGpu(float[] flatFeatures, float[] floatLabels, int numSamples, int numFeatures) {
        // Shared arrays
        final float[] biasArray = new float[] { this.bias };
        final float[] currentWeights = this.weights; // Reference
        final float[] currentPredictions = new float[numSamples];

        Kernel kernel = new Kernel() {
            @Override
            public void run() {
                int i = getGlobalId();
                float z = biasArray[0];
                int offset = i * numFeatures;
                for (int j = 0; j < numFeatures; j++) {
                    z += flatFeatures[offset + j] * currentWeights[j];
                }
                currentPredictions[i] = 1.0f / (1.0f + exp(-z));
            }
        };

        try {
            // Target specific device if selected
            String selectedName = com.example.automl_prototype_1.service.AppStateService.getInstance()
                    .getSelectedGpuName();
            com.aparapi.device.Device selectedDevice = null;
            if (selectedName != null) {
                selectedDevice = com.example.automl_prototype_1.gpu.HardwareAcceleratorService
                        .getDeviceByName(selectedName);
            }

            kernel.setExplicit(true);
            kernel.put(flatFeatures);
            kernel.put(biasArray);
            kernel.put(currentWeights);

            // Create Range
            Range range;
            if (selectedDevice != null) {
                range = selectedDevice.createRange(numSamples);
                System.out.println("CORE: GPU LogReg using Device: " + selectedDevice.getShortDescription());
            } else {
                range = Range.create(numSamples);
            }

            // Warmup
            kernel.execute(range);

            for (int epoch = 0; epoch < epochs; epoch++) {
                kernel.put(currentWeights);
                kernel.put(biasArray);
                kernel.execute(range);
                kernel.get(currentPredictions);

                // Backward Pass (CPU)
                float[] dWeights = new float[numFeatures];
                float dBias = 0.0f;

                for (int i = 0; i < numSamples; i++) {
                    float error = currentPredictions[i] - floatLabels[i];
                    dBias += error;
                    for (int j = 0; j < numFeatures; j++) {
                        dWeights[j] += error * flatFeatures[i * numFeatures + j];
                    }
                }

                // Update
                for (int j = 0; j < numFeatures; j++) {
                    currentWeights[j] -= learningRate * (dWeights[j] / numSamples);
                }
                biasArray[0] -= learningRate * (dBias / numSamples);
            }
            // Save back to instance
            this.bias = biasArray[0];

        } finally {
            kernel.dispose();
        }
    }

    private void trainCpu(float[] flatFeatures, float[] floatLabels, int numSamples, int numFeatures) {
        // Simple CPU Gradient Descent
        // Simple CPU Gradient Descent

        for (int epoch = 0; epoch < epochs; epoch++) {
            float[] dWeights = new float[numFeatures];
            float dBias = 0.0f;

            // Parallel Forward Pass & Error Calc
            // Note: Parallel streams for outer loop is tricky with accumulation,
            // so we'll keep it simple or use careful reduction.
            // For now, simple sequential loop to avoid race conditions on dWeights/dBias
            // without complex sync. Given small epochs/batch, this is "okay" as fallback.

            for (int i = 0; i < numSamples; i++) {
                float z = this.bias;
                int offset = i * numFeatures;
                for (int j = 0; j < numFeatures; j++) {
                    z += flatFeatures[offset + j] * this.weights[j];
                }
                float pred = 1.0f / (1.0f + (float) Math.exp(-z));
                float error = pred - floatLabels[i];

                dBias += error;
                for (int j = 0; j < numFeatures; j++) {
                    dWeights[j] += error * flatFeatures[offset + j];
                }
            }

            // Update
            for (int j = 0; j < numFeatures; j++) {
                this.weights[j] -= learningRate * (dWeights[j] / numSamples);
            }
            this.bias -= learningRate * (dBias / numSamples);
        }
    }

    @Override
    public int[] predict(double[][] features) {
        if (!isTrained)
            return new int[features.length];

        int numSamples = features.length;
        int numFeatures = features[0].length;

        final float[] flatFeatures = new float[numSamples * numFeatures];
        for (int i = 0; i < numSamples; i++) {
            for (int j = 0; j < numFeatures; j++) {
                flatFeatures[i * numFeatures + j] = (float) features[i][j];
            }
        }

        float[] predictions = new float[numSamples];

        if (gpuCircuitBreakerOpen) {
            predictCpu(flatFeatures, numSamples, numFeatures, predictions);
        } else {
            try {
                predictGpu(flatFeatures, numSamples, numFeatures, predictions);
            } catch (Throwable t) {
                System.err.println("CORE: GpuLogisticRegression Predict Failed: " + t.getMessage());
                gpuCircuitBreakerOpen = true;
                predictCpu(flatFeatures, numSamples, numFeatures, predictions);
            }
        }

        int[] results = new int[numSamples];
        for (int i = 0; i < numSamples; i++) {
            results[i] = predictions[i] >= 0.5f ? 1 : 0;
        }
        return results;
    }

    private void predictGpu(float[] flatFeatures, int numSamples, int numFeatures, float[] predictions) {
        final float finalBias = this.bias;
        final float[] finalWeights = this.weights;

        Kernel kernel = new Kernel() {
            @Override
            public void run() {
                int i = getGlobalId();
                float z = finalBias;
                int offset = i * numFeatures;
                for (int j = 0; j < numFeatures; j++) {
                    z += flatFeatures[offset + j] * finalWeights[j];
                }
                predictions[i] = 1.0f / (1.0f + exp(-z));
            }
        };

        try {
            // Target specific device logic
            String selectedName = com.example.automl_prototype_1.service.AppStateService.getInstance()
                    .getSelectedGpuName();
            com.aparapi.device.Device selectedDevice = null;
            if (selectedName != null) {
                selectedDevice = com.example.automl_prototype_1.gpu.HardwareAcceleratorService
                        .getDeviceByName(selectedName);
            }

            Range range = (selectedDevice != null) ? selectedDevice.createRange(numSamples) : Range.create(numSamples);
            kernel.execute(range);
        } finally {
            kernel.dispose();
        }
    }

    private void predictCpu(float[] flatFeatures, int numSamples, int numFeatures, float[] predictions) {
        IntStream.range(0, numSamples).parallel().forEach(i -> {
            float z = this.bias;
            int offset = i * numFeatures;
            for (int j = 0; j < numFeatures; j++) {
                z += flatFeatures[offset + j] * this.weights[j];
            }
            predictions[i] = 1.0f / (1.0f + (float) Math.exp(-z));
        });
    }
}
