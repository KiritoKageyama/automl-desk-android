package com.example.automl_prototype_1.algorithm;

import com.example.automl_prototype_1.model.Dataset;
import com.example.automl_prototype_1.service.IClassifier;
import com.example.automl_prototype_1.service.LogService;

import java.util.Arrays;
import java.util.List;
import java.util.concurrent.ThreadLocalRandom;
import java.util.stream.IntStream;

public class DifferentialEvolution extends AbstractFeatureSelectionAlgorithm {

    private final int populationSize;
    private final int maxIterations;
    private final double F = 0.5; // Differential weight
    private final double CR = 0.7; // Crossover probability

    private double[][] population;
    private double[] fitnesses;

    private double[] bestSolution;
    private double bestFitness = Double.NEGATIVE_INFINITY;

    public DifferentialEvolution(int populationSize, int maxIterations, int kFolds, int patience) {
        super(kFolds, patience);
        this.populationSize = populationSize;
        this.maxIterations = maxIterations;
    }

    public DifferentialEvolution() {
        this(30, 100, 5, 20);
    }

    @Override
    public String getAlgorithmName() {
        return "Differential Evolution (DE)";
    }

    @Override
    protected int getMaxIterations() {
        return maxIterations;
    }

    @Override
    protected double[] getBestSolution() {
        return bestSolution;
    }

    @Override
    protected double getBestFitness() {
        return bestFitness;
    }

    @Override
    protected void initialize(int numFeatures, Dataset dataset, IClassifier classifier, List<Fold> folds) {
        this.population = new double[populationSize][numFeatures];
        this.fitnesses = new double[populationSize];
        this.bestSolution = new double[numFeatures];

        for (int i = 0; i < populationSize; i++) {
            for (int j = 0; j < numFeatures; j++) {
                population[i][j] = random.nextDouble() < 0.5 ? 0 : 1;
            }
            fitnesses[i] = calculateFitness(population[i], classifier, folds);
            if (fitnesses[i] > bestFitness) {
                bestFitness = fitnesses[i];
                bestSolution = Arrays.copyOf(population[i], numFeatures);
            }
        }
        LogService.info("DE", "Initial Best Fitness: " + bestFitness);
    }

    @Override
    protected void updatePopulation(int iteration, int numFeatures, Dataset dataset, IClassifier classifier,
            List<Fold> folds) {

        // DE/rand/1/bin
        // For each target vector x_i, create trial vector u_i

        List<Integer> indices = IntStream.range(0, populationSize).boxed().toList();

        // Parallel processing of trials
        List<TrialResult> results = indices.parallelStream().map(i -> {
            var rand = ThreadLocalRandom.current();

            // 1. Mutation: Select r1, r2, r3 distinct from i
            int r1, r2, r3;
            do {
                r1 = rand.nextInt(populationSize);
            } while (r1 == i);
            do {
                r2 = rand.nextInt(populationSize);
            } while (r2 == i || r2 == r1);
            do {
                r3 = rand.nextInt(populationSize);
            } while (r3 == i || r3 == r1 || r3 == r2);

            double[] mutant = new double[numFeatures];
            for (int j = 0; j < numFeatures; j++) {
                // Continuous mutation: v = x_r1 + F * (x_r2 - x_r3)
                // Since our vectors are binary/continuous hybrid in this impl (init as binary
                // but mutated as continuous),
                // we treat them as continuous probabilities.
                double val = population[r1][j] + F * (population[r2][j] - population[r3][j]);
                mutant[j] = val;
            }

            // 2. Crossover (Binary)
            double[] trial = new double[numFeatures];
            int jRand = rand.nextInt(numFeatures);
            for (int j = 0; j < numFeatures; j++) {
                if (rand.nextDouble() < CR || j == jRand) {
                    // From mutant (apply sigmoid to discretize for fitness eval)
                    double sigmoid = 1.0 / (1.0 + Math.exp(-mutant[j]));
                    trial[j] = rand.nextDouble() < sigmoid ? 1.0 : 0.0;
                } else {
                    // From target
                    trial[j] = population[i][j];
                }
            }

            // 3. Selection
            double trialFitness = calculateFitness(trial, classifier, folds);
            return new TrialResult(i, trialFitness, trial);

        }).toList();

        // Sequential Update
        for (TrialResult res : results) {
            if (res.fitness >= fitnesses[res.index]) { // Greedy selection
                population[res.index] = res.trial;
                fitnesses[res.index] = res.fitness;
                if (res.fitness > bestFitness) {
                    bestFitness = res.fitness;
                    bestSolution = Arrays.copyOf(res.trial, numFeatures);
                }
            }
        }

        LogService.info("DE", "Iteration " + (iteration + 1) + " | Best Fitness: " + bestFitness);
    }

    private record TrialResult(int index, double fitness, double[] trial) {
    }
}
