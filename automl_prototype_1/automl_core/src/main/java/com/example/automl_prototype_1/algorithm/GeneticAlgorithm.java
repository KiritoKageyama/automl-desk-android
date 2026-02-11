package com.example.automl_prototype_1.algorithm;

import com.example.automl_prototype_1.model.Dataset;
import com.example.automl_prototype_1.service.IClassifier;
import com.example.automl_prototype_1.service.LogService;
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

        // Default values
        double tempCrossover = 0.8;
        double tempMutation = 0.05;
        int tempPopSize = populationSize;

        // Check for Advanced Mode Overrides
        try {
            com.example.automl_prototype_1.service.AppStateService appState = com.example.automl_prototype_1.service.AppStateService
                    .getInstance();
            if (appState.isAdvancedMode()) {
                Double confCrossover = appState.getAlgorithmHyperparameter("GA", "crossoverRate");
                Double confMutation = appState.getAlgorithmHyperparameter("GA", "mutationRate");
                Double confPop = appState.getAlgorithmHyperparameter("GA", "populationSize");

                if (confCrossover != null)
                    tempCrossover = confCrossover;
                if (confMutation != null)
                    tempMutation = confMutation;
                if (confPop != null)
                    tempPopSize = confPop.intValue();

                LogService.info("GA", String.format("Initialized with Advanced Params: Pop=%d, Cross=%.2f, Mut=%.2f",
                        tempPopSize, tempCrossover, tempMutation));
            }
        } catch (NoClassDefFoundError | Exception e) {
            // Fallback for standalone tests without AppState
        }

        this.populationSize = tempPopSize;
        this.maxGenerations = maxIterations;
        this.crossoverRate = tempCrossover;
        this.mutationRate = tempMutation;
        this.currentMutationRate = this.mutationRate;
        this.tournamentSize = 5;
        this.elitism = true;
    }

    public GeneticAlgorithm(int populationSize, int maxIterations) {
        this(populationSize, maxIterations, 5, Math.max(10, maxIterations / 5));
    }

    public GeneticAlgorithm() {
        this(20, 50); // Optimized for speed: 20 pop, 50 gens
    }

    @Override
    public String getAlgorithmName() {
        return "Genetic Algorithm (Feature Selection)";
    }

    @Override
    protected int getMaxIterations() {
        return (manualMaxIterations > 0) ? manualMaxIterations : this.maxGenerations;
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
    protected double getAverageFitness() {
        if (population == null || population.isEmpty())
            return 0.0;
        double sum = 0.0;
        int count = 0;
        for (Solution s : population) {
            // Check for valid fitness
            if (s.fitness != Double.NEGATIVE_INFINITY) {
                sum += s.fitness;
                count++;
            }
        }
        return count > 0 ? sum / count : 0.0;
    }

    @Override
    protected void initialize(int numFeatures, Dataset dataset, IClassifier classifier, List<Fold> folds) {
        // Parent handles cache clearing
        this.population = new ArrayList<>(populationSize);
        for (int i = 0; i < populationSize; i++) {
            population.add(new Solution(numFeatures));
        }
        evaluatePopulation(classifier, folds);
        this.bestOverallSolution = findBestInPopulation();
        LogService.info("GA", String.format("Initial Best Fitness = %.4f", this.bestOverallSolution.fitness));
    }

    @Override
    protected void updatePopulation(int iteration, int numFeatures, Dataset dataset, IClassifier classifier,
            List<Fold> folds) {
        List<Solution> nextPopulation = Collections.synchronizedList(new ArrayList<>(populationSize));
        if (elitism && bestOverallSolution != null) {
            nextPopulation.add(new Solution(bestOverallSolution));
        }

        // Parallel Population Generation
        int optimalThreads = com.example.automl_prototype_1.gpu.HardwareAcceleratorService
                .calculateOptimalThreadCount();
        @SuppressWarnings("resource")
        java.util.concurrent.ForkJoinPool customThreadPool = new java.util.concurrent.ForkJoinPool(optimalThreads);

        try {
            // Generate offpsring in parallel
            // We need to generate (populationSize - existing) new solutions
            int needed = populationSize - nextPopulation.size();

            if (needed > 0) {
                List<Solution> offsprings = customThreadPool
                        .submit(() -> java.util.stream.IntStream.range(0, needed / 2 + 1) // Each iter produces 2
                                                                                          // children
                                .parallel()
                                .mapToObj(i -> {
                                    Solution parent1 = tournamentSelection();
                                    Solution parent2 = tournamentSelection();
                                    Solution child1 = new Solution(parent1);
                                    Solution child2 = new Solution(parent2);
                                    if (ThreadLocalRandom.current().nextDouble() < crossoverRate) {
                                        performCrossover(child1, child2);
                                    }
                                    mutate(child1);
                                    mutate(child2);
                                    return List.of(child1, child2);
                                })
                                .flatMap(List::stream)
                                .limit(needed)
                                .toList())
                        .get();
                nextPopulation.addAll(offsprings);
            }
        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            customThreadPool.shutdown();
        }

        this.population = nextPopulation;
        evaluatePopulation(classifier, folds);

        Solution bestOfGeneration = findBestInPopulation();

        // Adaptive Mutation Logic
        // Adaptive Mutation Logic
        if (bestOfGeneration.fitness > this.bestOverallSolution.fitness) {
            this.bestOverallSolution = new Solution(bestOfGeneration);
            this.stagnationCounter = 0; // Reset
            this.currentMutationRate = 0.05; // Reset to base
        } else {
            this.stagnationCounter++;
            if (this.stagnationCounter >= 5) {
                // Boost mutation to escape local optimum
                double oldRate = this.currentMutationRate;
                this.currentMutationRate = Math.min(0.3, this.currentMutationRate * 1.5);

                // Log only on changes or periodically
                if (Math.abs(oldRate - this.currentMutationRate) > 0.001 || this.stagnationCounter % 20 == 0) {
                    LogService.info("GA", String.format("Stagnation detected (%d gens). Boosting mutation to %.3f",
                            stagnationCounter, currentMutationRate));
                }
            }
        }

        if ((iteration + 1) % 10 == 0 || iteration == 0 || iteration == maxGenerations - 1) {
            LogService.info("GA", String.format("Gen %d/%d: Best Fitness = %.4f (Mut: %.2f)", iteration + 1,
                    maxGenerations, this.bestOverallSolution.fitness, this.currentMutationRate));
        }
    }

    // --- Batched evaluatePopulation ---
    private void evaluatePopulation(IClassifier classifier, List<Fold> folds) {
        try {
            List<double[]> chromosomes = this.population.stream()
                    .map(s -> s.chromosome)
                    .collect(java.util.stream.Collectors.toList());

            List<Double> fitnessValues = evaluatePopulationBatched(chromosomes, classifier, folds);

            for (int i = 0; i < population.size(); i++) {
                population.get(i).fitness = fitnessValues.get(i);
            }
        } catch (Exception e) {
            LogService.error("GA", "Evaluation failed: " + e.getMessage());
            // Fallback (though evaluatePopulationBatched has internal fallback, this
            // catches other errors)
            this.population.stream()
                    .forEach(sol -> sol.fitness = calculateFitness(sol.chromosome, classifier, folds));
        }
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