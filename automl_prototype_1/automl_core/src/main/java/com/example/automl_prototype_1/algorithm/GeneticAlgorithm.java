package com.example.automl_prototype_1.algorithm;

import com.example.automl_prototype_1.model.Dataset;
import com.example.automl_prototype_1.service.IClassifier;
import java.util.*;
import java.util.concurrent.ThreadLocalRandom;

public class GeneticAlgorithm extends AbstractFeatureSelectionAlgorithm {

    private final int populationSize;
    private final int maxGenerations;
    private final double crossoverRate;
    private final double mutationRate;
    private final int tournamentSize;
    private final boolean elitism;
    private List<Solution> population;
    private Solution bestOverallSolution;

    private int stagnationCounter = 0;
    private double currentMutationRate;

    public GeneticAlgorithm(int populationSize, int maxIterations, int kFolds, int patience) {
        super(kFolds, patience);
        this.populationSize = populationSize;
        this.maxGenerations = maxIterations;
        this.crossoverRate = 0.8;
        this.mutationRate = 0.05;
        this.currentMutationRate = this.mutationRate; // Initialize custom rate
        this.tournamentSize = 5;
        this.elitism = true;
    }

    public GeneticAlgorithm(int populationSize, int maxIterations) {
        this(populationSize, maxIterations, 5, 20);
    }

    public GeneticAlgorithm() {
        this(30, 100);
    }

    @Override
    public String getAlgorithmName() {
        return "Genetic Algorithm (Feature Selection)";
    }

    @Override
    protected int getMaxIterations() {
        return this.maxGenerations;
    }

    @Override
    protected double[] getBestSolution() {
        return bestOverallSolution != null ? bestOverallSolution.chromosome : null;
    }

    @Override
    protected double getBestFitness() {
        return bestOverallSolution != null ? bestOverallSolution.fitness : 0.0;
    }

    @Override
    protected void initialize(int numFeatures, Dataset dataset, IClassifier classifier, List<Fold> folds) {
        this.population = new ArrayList<>(populationSize);
        for (int i = 0; i < populationSize; i++) {
            population.add(new Solution(numFeatures));
        }
        evaluatePopulation(classifier, folds);
        this.bestOverallSolution = findBestInPopulation();
        System.out.printf("GA Initial Best Fitness = %.4f%n", this.bestOverallSolution.fitness);
    }

    @Override
    protected void updatePopulation(int iteration, int numFeatures, Dataset dataset, IClassifier classifier,
            List<Fold> folds) {
        List<Solution> nextPopulation = Collections.synchronizedList(new ArrayList<>(populationSize));
        if (elitism && bestOverallSolution != null) {
            nextPopulation.add(new Solution(bestOverallSolution));
        }

        while (nextPopulation.size() < populationSize) {
            Solution parent1 = tournamentSelection();
            Solution parent2 = tournamentSelection();
            Solution child1 = new Solution(parent1);
            Solution child2 = new Solution(parent2);
            if (random.nextDouble() < crossoverRate) {
                performCrossover(child1, child2);
            }
            mutate(child1);
            mutate(child2);
            if (nextPopulation.size() < populationSize)
                nextPopulation.add(child1);
            if (nextPopulation.size() < populationSize)
                nextPopulation.add(child2);
        }

        this.population = nextPopulation;
        evaluatePopulation(classifier, folds);

        Solution bestOfGeneration = findBestInPopulation();

        // Adaptive Mutation Logic
        if (bestOfGeneration.fitness > this.bestOverallSolution.fitness) {
            this.bestOverallSolution = new Solution(bestOfGeneration);
            this.stagnationCounter = 0; // Reset
            this.currentMutationRate = 0.05; // Reset to base
        } else {
            this.stagnationCounter++;
            if (this.stagnationCounter >= 5) {
                // Boost mutation to escape local optimum
                this.currentMutationRate = Math.min(0.3, this.currentMutationRate * 1.5);
                System.out.printf("GA: Stagnation detected (%d gens). Boosting mutation to %.3f%n", stagnationCounter,
                        currentMutationRate);
            }
        }

        System.out.printf("GA Gen %d/%d: Best Fitness = %.4f (Mut: %.2f)%n", iteration + 1, maxGenerations,
                this.bestOverallSolution.fitness, this.currentMutationRate);
    }

    // --- NEW evaluatePopulation that accepts folds ---
    private void evaluatePopulation(IClassifier classifier, List<Fold> folds) {
        this.population.parallelStream()
                .forEach(sol -> sol.fitness = calculateFitness(sol.chromosome, classifier, folds));
    }

    private Solution findBestInPopulation() {
        // Find the solution with the highest fitness in the current population
        return Collections.max(this.population, Comparator.comparingDouble(s -> s.fitness));
    }

    private Solution tournamentSelection() {
        Solution bestInTournament = null;
        // Select a few random solutions from the population
        for (int i = 0; i < tournamentSize; i++) {
            Solution competitor = population.get(ThreadLocalRandom.current().nextInt(population.size()));
            // The one with the best fitness wins the tournament
            if (bestInTournament == null || competitor.fitness > bestInTournament.fitness) {
                bestInTournament = competitor;
            }
        }
        return bestInTournament;
    }

    private void performCrossover(Solution s1, Solution s2) {
        int numFeatures = s1.chromosome.length;
        if (numFeatures <= 1)
            return; // Cannot perform crossover on a single feature

        // Select a random point to split the chromosome
        int crossoverPoint = 1 + ThreadLocalRandom.current().nextInt(numFeatures - 1);

        // Swap the genetic material after the crossover point
        for (int i = crossoverPoint; i < numFeatures; i++) {
            double temp = s1.chromosome[i];
            s1.chromosome[i] = s2.chromosome[i];
            s2.chromosome[i] = temp;
        }
    }

    private void mutate(Solution solution) {
        // For each gene in the chromosome...
        for (int i = 0; i < solution.chromosome.length; i++) {
            // ...there is a small chance it will be flipped.
            if (ThreadLocalRandom.current().nextDouble() < currentMutationRate) {
                solution.chromosome[i] = 1.0 - solution.chromosome[i]; // Flip bit (0 to 1, or 1 to 0)
            }
        }
    }

    // --- Solution Inner Class for Feature Subset (Chromosome) ---
    private class Solution {
        double[] chromosome;
        double fitness = Double.NEGATIVE_INFINITY;

        // Creates a random chromosome
        public Solution(int numFeatures) {
            this.chromosome = new double[numFeatures];
            for (int i = 0; i < numFeatures; i++) {
                this.chromosome[i] = ThreadLocalRandom.current().nextDouble() < 0.5 ? 1.0 : 0.0;
            }
        }

        // Creates a copy of another solution
        public Solution(Solution other) {
            this.chromosome = Arrays.copyOf(other.chromosome, other.chromosome.length);
            this.fitness = other.fitness;
        }
    }
}