package com.example.automl_prototype_1.algorithm;

import com.example.automl_prototype_1.model.Dataset;
import com.example.automl_prototype_1.service.IClassifier;
import java.util.List;

public class AtomSearchOptimization extends AbstractFeatureSelectionAlgorithm {
    private final int populationSize;
    private final int maxIterations;
    private final double alpha;
    private final double beta;
    private double[][] atoms;
    private double[][] velocities;
    private double[] fitnesses;
    private double[] masses;
    private double[] bestAtomPosition;
    private double bestFitness;

    public AtomSearchOptimization(int populationSize, int maxIterations, int kFolds, int patience) {
        super(kFolds, patience);
        this.populationSize = populationSize;
        this.maxIterations = maxIterations;
        this.alpha = 50;
        this.beta = 0.2;
    }

    public AtomSearchOptimization(int populationSize, int maxIterations) {
        this(populationSize, maxIterations, 5, 20);
    }

    public AtomSearchOptimization() {
        this(30, 100);
    }

    @Override
    public String getAlgorithmName() {
        return "Atom Search Optimization (ASO)";
    }

    @Override
    protected int getMaxIterations() {
        return this.maxIterations;
    }

    @Override
    protected double[] getBestSolution() {
        return this.bestAtomPosition;
    }

    @Override
    protected double getBestFitness() {
        return this.bestFitness;
    }

    @Override
    protected double getAverageFitness() {
        if (fitnesses == null || fitnesses.length == 0)
            return 0.0;
        double sum = 0.0;
        int count = 0;
        for (double f : fitnesses) {
            if (f != Double.NEGATIVE_INFINITY) {
                sum += f;
                count++;
            }
        }
        return count > 0 ? sum / count : 0.0;
    }

    @Override
    protected void initialize(int numFeatures, Dataset dataset, IClassifier classifier, List<Fold> folds) {
        this.atoms = new double[populationSize][numFeatures];
        this.velocities = new double[populationSize][numFeatures];
        this.fitnesses = new double[populationSize];
        this.masses = new double[populationSize];
        this.bestFitness = Double.NEGATIVE_INFINITY;
        this.bestAtomPosition = new double[numFeatures];

        for (int i = 0; i < populationSize; i++) {
            for (int j = 0; j < numFeatures; j++)
                atoms[i][j] = -1 + (2 * random.nextDouble());
            fitnesses[i] = calculateFitness(toBinaryPosition(atoms[i]), classifier, folds);
            if (fitnesses[i] > bestFitness) {
                bestFitness = fitnesses[i];
                bestAtomPosition = toBinaryPosition(atoms[i]);
            }
        }
        com.example.automl_prototype_1.service.LogService.info("ASO", "Initial Best Fitness: " + this.bestFitness);
    }

    @Override
    protected void updatePopulation(int iteration, int numFeatures, Dataset dataset, IClassifier classifier,
            List<Fold> folds) {
        calculateMasses();
        final int k = (int) Math
                .round(populationSize - (populationSize - 2) * Math.sqrt((double) iteration / maxIterations));

        // Calculate forces and update positions in parallel
        List<Integer> indices = java.util.stream.IntStream.range(0, populationSize).boxed().toList();

        // We use a temporary array for new positions/velocities to avoid race
        // conditions if reading/writing same arrays
        // But ASO reads from 'atoms' (current) and writes to 'atoms' (next).
        // In standard implementation, strict synchronization means we compute new state
        // based on OLD state.
        // The original code was updating 'atoms' in place sequentially, so particle i+1
        // saw updated particle i.
        // ASO usually assumes synchronous update (like PSO). We will stick to
        // synchronous update (new state based on old state).

        final double[][] currentAtoms = new double[populationSize][];
        final double[][] currentVelocities = new double[populationSize][];
        for (int i = 0; i < populationSize; i++) {
            currentAtoms[i] = java.util.Arrays.copyOf(atoms[i], numFeatures);
            currentVelocities[i] = java.util.Arrays.copyOf(velocities[i], numFeatures);
        }

        // Parallel processing of particles with bounded concurrency (Smart Parallelism)
        int optimalThreads = com.example.automl_prototype_1.gpu.HardwareAcceleratorService
                .calculateOptimalThreadCount();
        List<FitnessResult> results;
        @SuppressWarnings("resource")
        java.util.concurrent.ForkJoinPool customThreadPool = new java.util.concurrent.ForkJoinPool(optimalThreads);
        try {
            results = customThreadPool.submit(() -> indices.parallelStream().map(i -> {
                var rand = java.util.concurrent.ThreadLocalRandom.current();
                double[] force = new double[numFeatures];

                // Interaction force
                for (int neighborIndex = 0; neighborIndex < k; neighborIndex++) {
                    if (i == neighborIndex)
                        continue;
                    // Note: neighborIndex in original code was just 0..k loop implies k neighbors?
                    // Actually the original code loop `neighborIndex < k` means it picked first k
                    // atoms.
                    // This is a simplification of K-best neighbors or random K.
                    // Original code: `fitnesses` array is indexed 0..populationSize.
                    // We will use the same logic: interact with first K atoms (assuming they are
                    // sorted? No, original didn't sort).
                    // Just interacting with 0..k indices.

                    for (int d = 0; d < numFeatures; d++) {
                        double r = rand.nextDouble();
                        force[d] += (fitnesses[i] < fitnesses[neighborIndex] ? 1 : -1) * r * beta
                                * (currentAtoms[neighborIndex][d] - currentAtoms[i][d]);
                    }
                }

                // Constraint force from best atom
                for (int d = 0; d < numFeatures; d++) {
                    double r = rand.nextDouble();
                    force[d] += alpha * r * (bestAtomPosition[d] - currentAtoms[i][d]);
                }

                double[] newVelocity = new double[numFeatures];
                double[] newPosition = new double[numFeatures];
                double[] acceleration = new double[numFeatures];

                if (masses[i] > 1e-9) {
                    for (int d = 0; d < numFeatures; d++)
                        acceleration[d] = force[d] / masses[i];
                }

                for (int d = 0; d < numFeatures; d++) {
                    double r = rand.nextDouble();
                    newVelocity[d] = r * currentVelocities[i][d] + acceleration[d];
                    newPosition[d] = currentAtoms[i][d] + newVelocity[d];
                }

                double[] binaryPos = new double[numFeatures];
                for (int d = 0; d < numFeatures; d++) {
                    double sigmoid = 1.0 / (1.0 + Math.exp(-newPosition[d]));
                    binaryPos[d] = rand.nextDouble() < sigmoid ? 1.0 : 0.0;
                }

                double fit = calculateFitness(binaryPos, classifier, folds);
                return new FitnessResult(i, fit, newPosition, newVelocity, binaryPos);
            }).toList()).get();
        } catch (Exception e) {
            e.printStackTrace();
            results = java.util.Collections.emptyList();
        } finally {
            customThreadPool.shutdown();
        }

        // Update global state
        for (FitnessResult res : results) {
            atoms[res.i] = res.position;
            velocities[res.i] = res.velocity;
            fitnesses[res.i] = res.fitness;

            if (res.fitness > bestFitness) {
                bestFitness = res.fitness;
                bestAtomPosition = res.binaryPos;
            }
        }
        com.example.automl_prototype_1.service.LogService.info("ASO",
                "Iteration " + (iteration + 1) + " | Best Fitness: " + this.bestFitness);
    }

    private record FitnessResult(int i, double fitness, double[] position, double[] velocity, double[] binaryPos) {
    }

    private double[] toBinaryPosition(double[] continuousPosition) {
        double[] binaryPosition = new double[continuousPosition.length];
        for (int i = 0; i < continuousPosition.length; i++) {
            double sigmoid = 1.0 / (1.0 + Math.exp(-continuousPosition[i]));
            binaryPosition[i] = random.nextDouble() < sigmoid ? 1.0 : 0.0;
        }
        return binaryPosition;
    }

    private void calculateMasses() {
        double bestFit = Double.NEGATIVE_INFINITY, worstFit = Double.POSITIVE_INFINITY;
        for (double fitness : fitnesses) {
            if (fitness > bestFit)
                bestFit = fitness;
            if (fitness < worstFit)
                worstFit = fitness;
        }
        double fitRange = bestFit - worstFit;
        if (fitRange < 1e-9) {
            for (int i = 0; i < populationSize; i++)
                masses[i] = 1.0 / populationSize;
            return;
        }
        double totalMass = 0;
        for (int i = 0; i < populationSize; i++) {
            masses[i] = Math.exp((fitnesses[i] - worstFit) / fitRange);
            totalMass += masses[i];
        }
        if (totalMass > 1e-9)
            for (int i = 0; i < populationSize; i++)
                masses[i] /= totalMass;
    }
}