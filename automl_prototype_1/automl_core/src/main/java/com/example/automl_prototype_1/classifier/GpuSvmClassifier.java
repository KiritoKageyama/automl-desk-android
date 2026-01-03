package com.example.automl_prototype_1.classifier;

import com.aparapi.Kernel;
import com.aparapi.Range;
import com.example.automl_prototype_1.service.IClassifier;
import java.util.stream.IntStream;

/**
 * GPU-Accelerated Linear SVM using Pegasos (Primal Estimated sub-GrAdient
 * SOlver for SVM).
 * Includes Circuit Breaker and Auto-CPU Fallback for robustness.
 */
public class GpuSvmClassifier implements IClassifier {

    private final float lambda; // Regularization parameter
    private final int epochs;
    private float[] weights;
    private float bias;
    private boolean isTrained = false;

    // Circuit Breaker to prevent repeated GPU failures
    private static boolean gpuCircuitBreakerOpen = false;

    public GpuSvmClassifier() {
        // Lambda = 1 / (C * n)
        this.lambda = 0.0001f;
        this.epochs = 1000;
    }

    @Override
    public String getName() {
        return "GPU-Accelerated Linear SVM (Aparapi)";
    }

    @Override
    public void train(double[][] features, int[] labels) {
        int numSamples = features.length;
        if (numSamples == 0)
            return;
        int numFeatures = features[0].length;

        // Initialize weights
        this.weights = new float[numFeatures];
        this.bias = 0.0f;

        // Flatten features
        final float[] flatFeatures = new float[numSamples * numFeatures];
        for (int i = 0; i < numSamples; i++) {
            for (int j = 0; j < numFeatures; j++) {
                flatFeatures[i * numFeatures + j] = (float) features[i][j];
            }
        }

        // Map labels to -1 / +1
        final float[] floatLabels = new float[numSamples];
        int minLabel = Integer.MAX_VALUE;
        for (int l : labels)
            if (l < minLabel)
                minLabel = l;

        for (int i = 0; i < numSamples; i++) {
            floatLabels[i] = (labels[i] == minLabel) ? -1.0f : 1.0f;
        }

        if (gpuCircuitBreakerOpen) {
            trainCpu(flatFeatures, floatLabels, numSamples, numFeatures);
        } else {
            try {
                trainGpu(flatFeatures, floatLabels, numSamples, numFeatures);
            } catch (Throwable t) {
                System.err.println("CORE: GpuSvmClassifier Train Failed: " + t.getMessage());
                gpuCircuitBreakerOpen = true;
                trainCpu(flatFeatures, floatLabels, numSamples, numFeatures);
            }
        }
        isTrained = true;
    }

    private void trainGpu(float[] flatFeatures, float[] floatLabels, int numSamples, int numFeatures) {
        final float[] currentWeights = this.weights;
        final float[] biasContainer = new float[] { 0.0f };
        final float[] weightGradients = new float[numFeatures * numSamples];
        final float[] biasGradients = new float[numSamples];

        Kernel gradientKernel = new Kernel() {
            @Override
            public void run() {
                int i = getGlobalId();
                int offset = i * numFeatures;
                float dot = 0;
                for (int j = 0; j < numFeatures; j++) {
                    dot += flatFeatures[offset + j] * currentWeights[j];
                }
                dot += biasContainer[0];
                float y = floatLabels[i];
                if (y * dot < 1.0f) {
                    biasGradients[i] = -y;
                    for (int j = 0; j < numFeatures; j++) {
                        weightGradients[offset + j] = -y * flatFeatures[offset + j];
                    }
                } else {
                    biasGradients[i] = 0;
                    for (int j = 0; j < numFeatures; j++) {
                        weightGradients[offset + j] = 0;
                    }
                }
            }
        };

        try {
            String selectedName = com.example.automl_prototype_1.service.AppStateService.getInstance()
                    .getSelectedGpuName();
            com.aparapi.device.Device selectedDevice = null;
            if (selectedName != null) {
                selectedDevice = com.example.automl_prototype_1.service.HardwareAcceleratorService
                        .getDeviceByName(selectedName);
            }

            gradientKernel.setExplicit(true);
            gradientKernel.put(flatFeatures);
            gradientKernel.put(floatLabels);
            gradientKernel.put(biasContainer);
            gradientKernel.put(currentWeights);

            Range range = (selectedDevice != null) ? selectedDevice.createRange(numSamples) : Range.create(numSamples);
            if (selectedDevice != null)
                System.out.println("CORE: GPU SVM using Device: " + selectedDevice.getShortDescription());

            // Warmup
            gradientKernel.execute(range);
            System.out.println("CORE: GPU SVM Init Mode: " + gradientKernel.getTargetDevice().getType());

            float eta = 0.01f;

            for (int t = 1; t <= epochs; t++) {
                gradientKernel.put(currentWeights);
                gradientKernel.put(biasContainer);
                gradientKernel.execute(range);
                gradientKernel.get(weightGradients);
                gradientKernel.get(biasGradients);

                float[] avgWGrad = new float[numFeatures];
                float avgBGrad = 0;

                for (int i = 0; i < numSamples; i++) {
                    int offset = i * numFeatures;
                    avgBGrad += biasGradients[i];
                    for (int j = 0; j < numFeatures; j++) {
                        avgWGrad[j] += weightGradients[offset + j];
                    }
                }

                for (int j = 0; j < numFeatures; j++) {
                    this.weights[j] = this.weights[j] - eta * (lambda * this.weights[j] + avgWGrad[j]);
                    currentWeights[j] = this.weights[j];
                }
                this.bias = this.bias - eta * avgBGrad;
                biasContainer[0] = this.bias;
            }
        } finally {
            gradientKernel.dispose();
        }
    }

    private void trainCpu(float[] flatFeatures, float[] floatLabels, int numSamples, int numFeatures) {
        float eta = 0.01f;

        for (int t = 1; t <= epochs; t++) {
            // To mock batch gradient descent, we must accumulate gradients
            // Parallel accumulation is tricky, using simplistic separate array per thread
            // is memory intensive
            // Sequential is safe for fallback

            float[] avgWGrad = new float[numFeatures];
            float avgBGrad = 0;

            for (int i = 0; i < numSamples; i++) {
                int offset = i * numFeatures;
                float dot = this.bias;
                for (int j = 0; j < numFeatures; j++) {
                    dot += flatFeatures[offset + j] * this.weights[j];
                }

                float y = floatLabels[i];
                if (y * dot < 1.0f) {
                    avgBGrad += -y;
                    for (int j = 0; j < numFeatures; j++) {
                        avgWGrad[j] += -y * flatFeatures[offset + j];
                    }
                }
            }

            for (int j = 0; j < numFeatures; j++) {
                this.weights[j] = this.weights[j] - eta * (lambda * this.weights[j] + avgWGrad[j]);
            }
            this.bias = this.bias - eta * avgBGrad;
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

        final int[] results = new int[numSamples];

        if (gpuCircuitBreakerOpen) {
            predictCpu(flatFeatures, numSamples, numFeatures, results);
        } else {
            try {
                predictGpu(flatFeatures, numSamples, numFeatures, results);
            } catch (Throwable t) {
                System.err.println("CORE: GpuSvmClassifier Predict Failed: " + t.getMessage());
                gpuCircuitBreakerOpen = true;
                predictCpu(flatFeatures, numSamples, numFeatures, results);
            }
        }
        return results;
    }

    private void predictGpu(float[] flatFeatures, int numSamples, int numFeatures, int[] results) {
        final float[] finalWeights = this.weights;
        final float finalBias = this.bias;

        Kernel predictKernel = new Kernel() {
            @Override
            public void run() {
                int i = getGlobalId();
                int offset = i * numFeatures;
                float dot = finalBias;
                for (int j = 0; j < numFeatures; j++) {
                    dot += flatFeatures[offset + j] * finalWeights[j];
                }
                results[i] = dot >= 0 ? 1 : 0;
            }
        };

        try {
            String selectedName = com.example.automl_prototype_1.service.AppStateService.getInstance()
                    .getSelectedGpuName();
            com.aparapi.device.Device selectedDevice = null;
            if (selectedName != null) {
                selectedDevice = com.example.automl_prototype_1.service.HardwareAcceleratorService
                        .getDeviceByName(selectedName);
            }
            Range range = (selectedDevice != null) ? selectedDevice.createRange(numSamples) : Range.create(numSamples);
            predictKernel.execute(range);
        } finally {
            predictKernel.dispose();
        }
    }

    private void predictCpu(float[] flatFeatures, int numSamples, int numFeatures, int[] results) {
        IntStream.range(0, numSamples).parallel().forEach(i -> {
            int offset = i * numFeatures;
            float dot = this.bias;
            for (int j = 0; j < numFeatures; j++) {
                dot += flatFeatures[offset + j] * this.weights[j];
            }
            results[i] = dot >= 0 ? 1 : 0;
        });
    }
}
