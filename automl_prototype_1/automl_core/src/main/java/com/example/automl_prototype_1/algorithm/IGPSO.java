package com.example.automl_prototype_1.algorithm;

import com.example.automl_prototype_1.model.Dataset;
import com.example.automl_prototype_1.service.IClassifier;
import com.example.automl_prototype_1.service.LogService;

import java.util.*;
import java.util.concurrent.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * Improved Genetic Particle Swarm Optimization (IGPSO).
 * A high-performance feature selection algorithm designed for high-dimensional
 * data (Signal, Image, Text).
 * 
 * Key Features:
 * - GPU-Accelerated Fitness Evaluation (Batch Mode)
 * - Variance-based Feature Importance for Initialization & Mutation
 * - Chaos Theory Initialization (Logistic Map) to prevent premature convergence
 * - Self-Adaptive Parameter Tuning
 */
public class IGPSO extends AbstractFeatureSelectionAlgorithm {

    private final int populationSize, maxIterations;
    private final double crossoverRate;
    private double[][] positions, velocities, pBestPositions;
    private double[] fitness, pBestFitness;
    private double[] gBestPosition;
    private double gBestFitness;

    // Adaptive State
    private double[] featureImportance; // normalized [0,1]
    private double[] baseFeatureImportance; // To prevent importance decay
    private int stagnationCounter = 0;
    private static final int STAGNATION_LIMIT = 5;

    private record FitnessResult(int index, double fitness, double[] position, double[] velocity) {
    }

    public IGPSO(int populationSize, int maxIterations, int kFolds, int patience) {
        super(kFolds, patience);
        this.populationSize = populationSize;
        this.maxIterations = maxIterations;
        this.crossoverRate = 0.8;
    }

    public IGPSO(int populationSize, int maxIterations) {
        this(populationSize, maxIterations, 5, Math.max(10, maxIterations / 5));
    }

    public IGPSO() {
        this(30, 100);
    }

    @Override
    public String getAlgorithmName() {
        return "Improved Genetic PSO (IGPSO)";
    }

    @Override
    protected int getMaxIterations() {
        return (manualMaxIterations > 0) ? manualMaxIterations : this.maxIterations;
    }

    @Override
    protected double[] getBestSolution() {
        return this.gBestPosition;
    }

    @Override
    protected double getBestFitness() {
        return this.gBestFitness;
    }

    @Override
    protected double getAverageFitness() {
        if (fitness == null || fitness.length == 0)
            return 0.0;
        double sum = 0.0;
        int count = 0;
        for (double f : fitness) {
            // Check for valid fitness
            if (f != Double.NEGATIVE_INFINITY) {
                sum += f;
                count++;
            }
        }
        return count > 0 ? sum / count : 0.0;
    }

    @Override
    protected double getPopulationDiversity() {
        if (fitness == null || fitness.length <= 1)
            return 0.0;

        // Calculate Variance of Fitness
        double avg = getAverageFitness();
        double sumSqDiff = 0.0;
        int count = 0;
        for (double f : fitness) {
            if (f != Double.NEGATIVE_INFINITY) {
                sumSqDiff += Math.pow(f - avg, 2);
                count++;
            }
        }

        return count > 1 ? Math.sqrt(sumSqDiff / (count - 1)) : 0.0;
    }

    @Override
    protected void initialize(int numFeatures, Dataset dataset, IClassifier classifier, List<Fold> folds) {
        this.positions = new double[populationSize][numFeatures];
        this.velocities = new double[populationSize][numFeatures];
        this.fitness = new double[populationSize];
        this.pBestPositions = new double[populationSize][numFeatures];
        this.pBestFitness = new double[populationSize];
        this.gBestPosition = new double[numFeatures];
        this.gBestFitness = Double.NEGATIVE_INFINITY;

        // Compute Feature Importance (Variance-based) - Critical for High-Dim datasets
        // (Text/Image)
        this.featureImportance = computeFeatureImportance(dataset);
        this.baseFeatureImportance = Arrays.copyOf(featureImportance, featureImportance.length);

        // Smart Parallelism: Init Positions
        List<FitnessResult> results;
        try {
            results = this.executorService.submit(() -> IntStream.range(0, populationSize).parallel().mapToObj(i -> {
                // Deterministic Seeding (Particle-based)
                var rand = new java.util.Random(42 + i);
                double[] pos = new double[numFeatures];
                double[] vel = new double[numFeatures];

                // Chaotic Initialization (Logistic Map)
                double chaoticValue = rand.nextDouble();
                for (int j = 0; j < numFeatures; j++) {
                    chaoticValue = 4.0 * chaoticValue * (1.0 - chaoticValue);
                    // Sparse Initialization (0.1 prob)
                    pos[j] = chaoticValue < 0.1 ? 1 : 0;
                    vel[j] = -1 + (2 * rand.nextDouble());
                }
                return new FitnessResult(i, 0.0, pos, vel);
            }).toList()).get();
        } catch (Exception e) {
            LogService.error("IGPSO", "Error in parallel initialization: " + e.getMessage());
            results = new ArrayList<>();
        }

        if (results.isEmpty())
            return;

        // Batch Evaluation (GPU Optimized)
        List<double[]> popMatrix = results.stream().map(FitnessResult::position)
                .collect(Collectors.toList());
        List<Double> batchFitness = evaluatePopulationBatched(popMatrix, classifier, folds);

        for (int i = 0; i < results.size(); i++) {
            FitnessResult res = results.get(i);
            this.positions[res.index] = res.position;
            this.velocities[res.index] = res.velocity;

            double val = batchFitness.get(i);
            this.fitness[res.index] = val;
            this.pBestFitness[res.index] = val;
            this.pBestPositions[res.index] = Arrays.copyOf(res.position, numFeatures);
        }

        updateGlobalBest();
        System.out.println("IGPSO Initial Best Fitness: " + this.gBestFitness);
    }

    @Override
    protected void updatePopulation(int iteration, int numFeatures, Dataset dataset, IClassifier classifier,
            List<Fold> folds) {

        // Failure-Aware Stagnation Handling
        if (stagnationCounter >= STAGNATION_LIMIT) {
            // Temporarily weaken importance bias (Stabilization, not decay)
            for (int j = 0; j < featureImportance.length; j++) {
                featureImportance[j] = 0.9 * featureImportance[j] + 0.1 * baseFeatureImportance[j];
                // Clamp to [0,1] to prevent drift
                featureImportance[j] = Math.max(0.0, Math.min(1.0, featureImportance[j]));
            }
            stagnationCounter = 0;
        }

        // Smart Parallelism: Update Pos/Vel
        List<FitnessResult> results;
        try {
            results = this.executorService.submit(() -> IntStream.range(0, populationSize).parallel().mapToObj(i -> {
                // Deterministic Seeding
                var rand = new java.util.Random(42 + i + ((long) iteration * populationSize));
                double[] newPos = new double[numFeatures];
                double[] newVel = Arrays.copyOf(velocities[i], numFeatures);

                // Dynamic Parameters (TVAC & LTV-IW)
                double progress = (double) iteration / maxIterations;
                double currentW = 0.9 - (0.5 * progress);
                double currentC1 = 2.5 - (2.0 * progress);
                double currentC2 = 0.5 + (2.0 * progress);

                for (int j = 0; j < numFeatures; j++) {
                    double r1 = rand.nextDouble(), r2 = rand.nextDouble();
                    double cognitive = currentC1 * r1 * (pBestPositions[i][j] - positions[i][j]);
                    double social = currentC2 * r2 * (gBestPosition[j] - positions[i][j]);

                    // Importance-Guided Velocity Update
                    double importanceFactor = 0.5 + featureImportance[j]; // [0.5, 1.5]
                    double v = importanceFactor * ((currentW * newVel[j]) + cognitive + social);

                    // Dynamic Velocity Clamp (Importance-Aware)
                    double vmax = 2.0 + 2.0 * featureImportance[j];
                    if (v > vmax)
                        v = vmax;
                    else if (v < -vmax)
                        v = -vmax;
                    newVel[j] = v;

                    // Importance-Biased Sigmoid
                    double bias = 0.5 + featureImportance[j];
                    double sigmoid = 1.0 / (1.0 + Math.exp(-bias * newVel[j]));
                    newPos[j] = rand.nextDouble() < sigmoid ? 1.0 : 0.0;
                }
                return new FitnessResult(i, 0.0, newPos, newVel);
            }).toList()).get();
        } catch (Exception e) {
            LogService.error("IGPSO", "Error in parallel update: " + e.getMessage());
            results = new ArrayList<>();
        }

        if (results.isEmpty())
            return;

        // Batch Evaluation
        List<double[]> popMatrix = results.stream().map(FitnessResult::position)
                .collect(Collectors.toList());
        List<Double> batchFitness = evaluatePopulationBatched(popMatrix, classifier, folds);

        for (int i = 0; i < results.size(); i++) {
            FitnessResult res = results.get(i);
            double val = batchFitness.get(i);

            this.positions[res.index] = res.position;
            this.velocities[res.index] = res.velocity;
            this.fitness[res.index] = val;

            if (val > this.pBestFitness[res.index]) {
                this.pBestFitness[res.index] = val;
                this.pBestPositions[res.index] = Arrays.copyOf(res.position, numFeatures);
            }
        }

        int bestParticleIndex = updateGlobalBest();

        // Genetic Operations
        int worstParticleIndex = 0;
        double worstFitness = Double.POSITIVE_INFINITY;
        for (int i = 0; i < populationSize; i++) {
            if (fitness[i] < worstFitness) {
                worstFitness = fitness[i];
                worstParticleIndex = i;
            }
        }

        int parent1Index = bestParticleIndex != -1 ? bestParticleIndex : 0;
        int parent2Index = tournamentSelection(fitness, 3);

        // Adaptive Mutation Rate
        double currentMutationRate = 0.1 * Math.exp(-3.0 * ((double) iteration / maxIterations));

        if (ThreadLocalRandom.current().nextDouble() < crossoverRate) {
            double[] newChild = crossover(positions[parent1Index], positions[parent2Index]);
            mutate(newChild, currentMutationRate);

            // Single eval
            double childFitness = calculateFitness(newChild, classifier, folds);

            // Sparsity Check
            int childOnes = 0;
            for (double bit : newChild) {
                if (bit == 1.0)
                    childOnes++;
            }

            if (childFitness > fitness[worstParticleIndex] && childOnes < numFeatures * 0.7) {
                positions[worstParticleIndex] = newChild;
                fitness[worstParticleIndex] = childFitness;
                if (childFitness > pBestFitness[worstParticleIndex]) {
                    pBestFitness[worstParticleIndex] = childFitness;
                    pBestPositions[worstParticleIndex] = Arrays.copyOf(newChild, numFeatures);
                }
            }
        }
        updateGlobalBest();
        System.out.println("Iteration " + (iteration + 1) + " | Best Fitness: " + gBestFitness);
    }

    private int updateGlobalBest() {
        int bestIndex = -1;
        boolean improved = false;
        for (int i = 0; i < populationSize; i++) {
            if (pBestFitness[i] > gBestFitness) {
                gBestFitness = pBestFitness[i];
                gBestPosition = Arrays.copyOf(pBestPositions[i], pBestPositions[i].length);
                bestIndex = i;
                improved = true;
            }
        }

        if (improved) {
            stagnationCounter = 0;
        } else {
            stagnationCounter++;
        }

        return bestIndex;
    }

    private double[] crossover(double[] parent1, double[] parent2) {
        int numFeatures = parent1.length;
        double[] child = new double[numFeatures];
        var rand = ThreadLocalRandom.current();

        for (int i = 0; i < numFeatures; i++) {
            if (featureImportance[i] > 0.6) {
                child[i] = parent1[i];
            } else {
                child[i] = rand.nextBoolean() ? parent1[i] : parent2[i];
            }
        }
        return child;
    }

    private void mutate(double[] solution, double baseMutationRate) {
        for (int i = 0; i < solution.length; i++) {
            double adaptiveRate = baseMutationRate * Math.exp(-2.0 * featureImportance[i]);
            if (ThreadLocalRandom.current().nextDouble() < adaptiveRate) {
                solution[i] = 1.0 - solution[i];
            }
        }
    }

    private double[] computeFeatureImportance(Dataset dataset) {
        int numFeatures = dataset.getFeatureCount();
        double[] importance = new double[numFeatures];
        double[][] X = dataset.getFeatures();
        int numRows = dataset.getRecordCount();

        if (numRows <= 1)
            return new double[numFeatures];

        for (int j = 0; j < numFeatures; j++) {
            double sum = 0;
            for (int i = 0; i < numRows; i++) {
                sum += X[i][j];
            }
            double mean = sum / numRows;

            double sumSqDiff = 0;
            for (int i = 0; i < numRows; i++) {
                double diff = X[i][j] - mean;
                sumSqDiff += diff * diff;
            }
            importance[j] = sumSqDiff / (numRows - 1);
        }

        double max = Arrays.stream(importance).max().orElse(1e-9);
        if (max == 0)
            max = 1.0;
        for (int j = 0; j < numFeatures; j++) {
            importance[j] /= max;
        }
        return importance;
    }

    private int tournamentSelection(double[] fitness, int k) {
        var rand = ThreadLocalRandom.current();
        int bestId = rand.nextInt(populationSize);
        for (int i = 1; i < k; i++) {
            int nextId = rand.nextInt(populationSize);
            if (fitness[nextId] > fitness[bestId]) {
                bestId = nextId;
            }
        }
        return bestId;
    }
}
