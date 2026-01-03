package com.example.automl_prototype_1.algorithm;

import com.example.automl_prototype_1.model.Dataset;
import com.example.automl_prototype_1.service.IClassifier;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.ThreadLocalRandom;
import java.util.stream.IntStream;

public class BinaryParticleSwarmOptimization extends AbstractFeatureSelectionAlgorithm {

    private final int populationSize;
    private final int maxIterations;
    private final double c1, c2, w;
    private double[][] positions, velocities, pBestPositions;
    private double[] fitness, pBestFitness;
    private double[] gBestPosition;
    private double gBestFitness;

    private record FitnessResult(int index, double fitness, double[] position, double[] velocity) {
    }

    public BinaryParticleSwarmOptimization(int populationSize, int maxIterations) {
        super(5, 20); // kFolds=5, patience=20
        this.populationSize = populationSize;
        this.maxIterations = maxIterations;
        this.c1 = 1.5;
        this.c2 = 1.5;
        this.w = 0.7;
    }

    public BinaryParticleSwarmOptimization() {
        this(30, 100);
    }

    @Override
    public String getAlgorithmName() {
        return "Binary Particle Swarm Optimization (BPSO)";
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

        List<FitnessResult> initialResults = IntStream.range(0, populationSize).parallel().mapToObj(i -> {
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

        for (FitnessResult res : initialResults) {
            this.positions[res.index] = res.position;
            this.velocities[res.index] = res.velocity;
            this.fitness[res.index] = res.fitness;
            this.pBestFitness[res.index] = res.fitness;
            this.pBestPositions[res.index] = Arrays.copyOf(res.position, numFeatures);
        }
        updateGlobalBest();
        System.out.println("BPSO Parallel Initial Best Fitness: " + this.gBestFitness);
        for (int i = 0; i < this.populationSize; i++) {
            fitness[i] = calculateFitness(positions[i], classifier, folds);
            pBestFitness[i] = fitness[i];
            pBestPositions[i] = Arrays.copyOf(positions[i], numFeatures);
        }
        updateGlobalBest();
    }

    @Override
    protected void updatePopulation(int iteration, int numFeatures, Dataset dataset, IClassifier classifier,
            List<Fold> folds) {
        List<FitnessResult> updatedResults = IntStream.range(0, populationSize).parallel().mapToObj(i -> {
            var rand = ThreadLocalRandom.current();
            double[] newPos = new double[numFeatures];
            double[] newVel = Arrays.copyOf(velocities[i], numFeatures);
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

        for (FitnessResult res : updatedResults) {
            this.positions[res.index] = res.position;
            this.velocities[res.index] = res.velocity;
            this.fitness[res.index] = res.fitness;
            if (res.fitness > this.pBestFitness[res.index]) {
                this.pBestFitness[res.index] = res.fitness;
                this.pBestPositions[res.index] = Arrays.copyOf(res.position, numFeatures);
            }
        }
        updateGlobalBest();
        System.out.println("Iteration " + (iteration + 1) + " | Best Fitness: " + this.gBestFitness);
    }

    private void updateGlobalBest() {
        for (int i = 0; i < populationSize; i++) {
            if (pBestFitness[i] > gBestFitness) {
                gBestFitness = pBestFitness[i];
                gBestPosition = Arrays.copyOf(pBestPositions[i], pBestPositions[i].length);
            }
        }
    }
}