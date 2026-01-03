package com.example.automl_prototype_1.algorithm;

import com.example.automl_prototype_1.model.Dataset;
import com.example.automl_prototype_1.service.IClassifier;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.ThreadLocalRandom;
import java.util.stream.IntStream;

public class IGPSO extends AbstractFeatureSelectionAlgorithm {

    private final int populationSize, maxIterations;
    private final double c1, c2, w, crossoverRate, mutationRate;
    private double[][] positions, velocities, pBestPositions;
    private double[] fitness, pBestFitness;
    private double[] gBestPosition;
    private double gBestFitness;

    private record FitnessResult(int index, double fitness, double[] position, double[] velocity) {
    }

    public IGPSO(int populationSize, int maxIterations, int kFolds, int patience) {
        super(kFolds, patience);
        this.populationSize = populationSize;
        this.maxIterations = maxIterations;
        this.c1 = 1.5;
        this.c2 = 1.5;
        this.w = 0.7;
        this.crossoverRate = 0.8;
        this.mutationRate = 0.05;
    }

    public IGPSO(int populationSize, int maxIterations) {
        this(populationSize, maxIterations, 5, 20);
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
        return this.maxIterations;
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
    protected void initialize(int numFeatures, Dataset dataset, IClassifier classifier, List<Fold> folds) {
        this.positions = new double[populationSize][numFeatures];
        this.velocities = new double[populationSize][numFeatures];
        this.fitness = new double[populationSize];
        this.pBestPositions = new double[populationSize][numFeatures];
        this.pBestFitness = new double[populationSize];
        this.gBestPosition = new double[numFeatures];
        this.gBestFitness = Double.NEGATIVE_INFINITY;

        List<FitnessResult> results = IntStream.range(0, populationSize).parallel().mapToObj(i -> {
            var rand = ThreadLocalRandom.current();
            double[] pos = new double[numFeatures];
            double[] vel = new double[numFeatures];
            for (int j = 0; j < numFeatures; j++) {
                pos[j] = rand.nextDouble() < 0.5 ? 0 : 1;
                vel[j] = -1 + (2 * rand.nextDouble());
            }
            double fit = calculateFitness(pos, classifier, folds);
            return new FitnessResult(i, fit, pos, vel);
        }).toList();

        for (FitnessResult res : results) {
            this.positions[res.index] = res.position;
            this.velocities[res.index] = res.velocity;
            this.fitness[res.index] = res.fitness;
            this.pBestFitness[res.index] = res.fitness;
            this.pBestPositions[res.index] = Arrays.copyOf(res.position, numFeatures);
        }

        updateGlobalBest();
        System.out.println("IGPSO Initial Best Fitness: " + this.gBestFitness);
    }

    @Override
    protected void updatePopulation(int iteration, int numFeatures, Dataset dataset, IClassifier classifier,
            List<Fold> folds) {

        // Parallel update and evaluation
        List<FitnessResult> results = IntStream.range(0, populationSize).parallel().mapToObj(i -> {
            var rand = ThreadLocalRandom.current();
            double[] newPos = new double[numFeatures];
            double[] newVel = Arrays.copyOf(velocities[i], numFeatures); // Start with old velocity logic if needed, or
                                                                         // just new array

            // Re-calculate velocity and position
            for (int j = 0; j < numFeatures; j++) {
                double r1 = rand.nextDouble(), r2 = rand.nextDouble();
                double cognitive = c1 * r1 * (pBestPositions[i][j] - positions[i][j]);
                double social = c2 * r2 * (gBestPosition[j] - positions[i][j]);
                newVel[j] = (w * newVel[j]) + cognitive + social;
                double sigmoid = 1.0 / (1.0 + Math.exp(-newVel[j]));
                newPos[j] = rand.nextDouble() < sigmoid ? 1.0 : 0.0;
            }
            double fit = calculateFitness(newPos, classifier, folds);
            return new FitnessResult(i, fit, newPos, newVel);
        }).toList();

        // Update state sequentially with results
        for (FitnessResult res : results) {
            this.positions[res.index] = res.position;
            this.velocities[res.index] = res.velocity;
            this.fitness[res.index] = res.fitness;

            if (res.fitness > this.pBestFitness[res.index]) {
                this.pBestFitness[res.index] = res.fitness;
                this.pBestPositions[res.index] = Arrays.copyOf(res.position, numFeatures);
            }
        }

        int bestParticleIndex = updateGlobalBest();

        // Genetic Operations (Sequential for simplicity, handled on updated population)
        int worstParticleIndex = 0;
        double worstFitness = Double.POSITIVE_INFINITY;
        for (int i = 0; i < populationSize; i++) {
            if (fitness[i] < worstFitness) {
                worstFitness = fitness[i];
                worstParticleIndex = i;
            }
        }

        int parent1Index = bestParticleIndex != -1 ? bestParticleIndex : 0;
        int parent2Index = ThreadLocalRandom.current().nextInt(populationSize);
        while (parent2Index == parent1Index)
            parent2Index = ThreadLocalRandom.current().nextInt(populationSize);

        if (ThreadLocalRandom.current().nextDouble() < crossoverRate) {
            double[] newChild = crossover(positions[parent1Index], positions[parent2Index]);
            mutate(newChild);
            // Evaluate child
            double childFitness = calculateFitness(newChild, classifier, folds);

            // Replace worst
            positions[worstParticleIndex] = newChild;
            fitness[worstParticleIndex] = childFitness;
            if (childFitness > pBestFitness[worstParticleIndex]) {
                pBestFitness[worstParticleIndex] = childFitness;
                pBestPositions[worstParticleIndex] = Arrays.copyOf(newChild, numFeatures);
            }
        }
        updateGlobalBest(); // Re-check if child is new global best
        System.out.println("Iteration " + (iteration + 1) + " | Best Fitness: " + gBestFitness);
    }

    private int updateGlobalBest() {
        int bestIndex = -1;
        for (int i = 0; i < populationSize; i++) {
            if (pBestFitness[i] > gBestFitness) {
                gBestFitness = pBestFitness[i];
                gBestPosition = Arrays.copyOf(pBestPositions[i], pBestPositions[i].length);
                bestIndex = i;
            }
        }
        return bestIndex;
    }

    private double[] crossover(double[] parent1, double[] parent2) {
        int numFeatures = parent1.length;
        double[] child = new double[numFeatures];
        int crossoverPoint = 1 + ThreadLocalRandom.current().nextInt(numFeatures - 2);
        for (int i = 0; i < numFeatures; i++) {
            child[i] = (i < crossoverPoint) ? parent1[i] : parent2[i];
        }
        return child;
    }

    private void mutate(double[] solution) {
        for (int i = 0; i < solution.length; i++) {
            if (ThreadLocalRandom.current().nextDouble() < mutationRate) {
                solution[i] = 1.0 - solution[i];
            }
        }
    }
}