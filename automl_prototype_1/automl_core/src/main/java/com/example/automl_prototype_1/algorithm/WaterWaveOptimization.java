package com.example.automl_prototype_1.algorithm;

import com.example.automl_prototype_1.model.Dataset;
import com.example.automl_prototype_1.service.IClassifier;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.ThreadLocalRandom;
import java.util.stream.IntStream;

public class WaterWaveOptimization extends AbstractFeatureSelectionAlgorithm {

    private final int maxIterations;
    private final int populationSize;
    private final double alpha;
    private double[][] waves;
    private double[] heights;
    private double[] wavelengths;
    private double[] bestSolutionWave;
    private double bestFitness;

    private record FitnessResult(int index, double fitness, double[] wave) {
    }

    public WaterWaveOptimization(int populationSize, int maxIterations, int kFolds, int patience) {
        super(kFolds, patience);
        this.populationSize = populationSize;
        this.maxIterations = maxIterations;
        this.alpha = 1.005;
    }

    public WaterWaveOptimization(int populationSize, int maxIterations) {
        this(populationSize, maxIterations, 5, 20);
    }

    public WaterWaveOptimization() {
        this(30, 100);
    }

    @Override
    public String getAlgorithmName() {
        return "Water Wave Optimization (WWO)";
    }

    @Override
    protected int getMaxIterations() {
        return this.maxIterations;
    }

    @Override
    protected double[] getBestSolution() {
        return this.bestSolutionWave;
    }

    @Override
    protected double getBestFitness() {
        return this.bestFitness;
    }

    @Override
    protected double getAverageFitness() {
        if (heights == null || heights.length == 0)
            return 0.0;
        double sum = 0.0;
        int count = 0;
        for (double h : heights) {
            double fitness = 1.0 - h;
            sum += fitness;
            count++;
        }
        return count > 0 ? sum / count : 0.0;
    }

    @Override
    protected void initialize(int numFeatures, Dataset dataset, IClassifier classifier, List<Fold> folds) {
        this.waves = new double[populationSize][numFeatures];
        this.heights = new double[populationSize];
        this.wavelengths = new double[populationSize];
        this.bestSolutionWave = new double[numFeatures];
        this.bestFitness = Double.NEGATIVE_INFINITY;

        // Parallel Initialization
        // Smart Parallelism
        List<FitnessResult> results = new java.util.ArrayList<>();
        try {
            results = this.executorService.submit(() -> IntStream.range(0, populationSize).parallel().mapToObj(i -> {
                var rand = ThreadLocalRandom.current();
                double[] wave = new double[numFeatures];
                for (int j = 0; j < numFeatures; j++) {
                    wave[j] = rand.nextDouble() < 0.5 ? 0 : 1;
                }
                double fitness = calculateFitness(wave, classifier, folds);
                return new FitnessResult(i, fitness, wave);
            }).toList()).get();
        } catch (Exception e) {
            com.example.automl_prototype_1.service.LogService.error("WWO",
                    "Error in parallel initialization: " + e.getMessage());
        }

        // Sequential Update
        for (FitnessResult res : results) {
            this.waves[res.index] = res.wave;
            this.heights[res.index] = 1.0 - res.fitness;
            this.wavelengths[res.index] = 0.5;
            if (res.fitness > this.bestFitness) {
                this.bestFitness = res.fitness;
                this.bestSolutionWave = Arrays.copyOf(res.wave, numFeatures);
            }
        }
        com.example.automl_prototype_1.service.LogService.info("WWO", "Initial Best Fitness: " + this.bestFitness);
    }

    @Override
    protected void updatePopulation(int iteration, int numFeatures, Dataset dataset, IClassifier classifier,
            List<Fold> folds) {

        // Parallel Propagation and Evaluation
        // Smart Parallelism
        List<FitnessResult> results = new java.util.ArrayList<>();
        try {
            results = this.executorService.submit(() -> IntStream.range(0, populationSize).parallel().mapToObj(i -> {
                // Propagate (using ThreadLocalRandom for safety)
                double[] originalWave = this.waves[i];
                double wavelength = this.wavelengths[i];

                double[] newWave = Arrays.copyOf(originalWave, numFeatures);
                int numToFlip = (int) Math.round(wavelength * numFeatures);
                if (numToFlip == 0)
                    numToFlip = 1;
                var rand = ThreadLocalRandom.current();
                for (int k = 0; k < numToFlip; k++) {
                    int randomIndex = rand.nextInt(numFeatures);
                    newWave[randomIndex] = 1.0 - newWave[randomIndex];
                }

                double newFitness = calculateFitness(newWave, classifier, folds);
                return new FitnessResult(i, newFitness, newWave);
            }).toList()).get();
        } catch (Exception e) {
            com.example.automl_prototype_1.service.LogService.error("WWO",
                    "Error in parallel propagation: " + e.getMessage());
        }

        // Sequential State Update & Refraction/Breaking
        for (FitnessResult res : results) {
            int i = res.index;
            double newFitness = res.fitness;
            double newHeight = 1.0 - newFitness;

            if (newHeight < this.heights[i]) {
                this.waves[i] = res.wave;
                double oldHeight = this.heights[i]; // Store old height for refraction
                this.heights[i] = newHeight;
                this.wavelengths[i] = refract(this.wavelengths[i], oldHeight, newHeight);

                if (newFitness > this.bestFitness) {
                    this.bestFitness = newFitness;
                    this.bestSolutionWave = Arrays.copyOf(this.waves[i], numFeatures);

                    // Breaking Wave Logic (affects other particles, so must be sequential main
                    // loop)
                    int worstWaveIndex = 0;
                    double max_h = -1;
                    for (int k = 0; k < populationSize; k++) {
                        if (this.heights[k] > max_h) {
                            max_h = this.heights[k];
                            worstWaveIndex = k;
                        }
                    }

                    // Break wave (generate new random wave)
                    this.waves[worstWaveIndex] = breakWave(numFeatures);
                    // We need to calculate fitness for this broken wave.
                    // To keep it simple, we do it sequentially here (it's one evaluation)
                    this.heights[worstWaveIndex] = 1.0
                            - calculateFitness(this.waves[worstWaveIndex], classifier, folds);
                }
            }
        }

        // Wavelength reduction
        for (int i = 0; i < populationSize; i++) {
            this.wavelengths[i] = this.wavelengths[i] * alpha;
        }
        com.example.automl_prototype_1.service.LogService.info("WWO",
                "Iteration " + (iteration + 1) + " | Best Fitness: " + this.bestFitness);
    }

    // Helper methods (propagate removed/inlined to use ThreadLocalRandom)

    private double refract(double currentWavelength, double currentHeight, double newHeight) {
        if (currentHeight == 0)
            return currentWavelength; // Avoid div by zero
        return currentWavelength * (newHeight / currentHeight);
    }

    private double[] breakWave(int numFeatures) {
        double[] newWave = new double[numFeatures];
        var rand = ThreadLocalRandom.current();
        for (int j = 0; j < numFeatures; j++)
            newWave[j] = rand.nextDouble() < 0.5 ? 0 : 1;
        return newWave;
    }
}