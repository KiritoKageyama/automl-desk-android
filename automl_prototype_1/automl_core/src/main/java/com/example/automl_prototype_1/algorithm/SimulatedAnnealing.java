package com.example.automl_prototype_1.algorithm;

import com.example.automl_prototype_1.model.Dataset;
import com.example.automl_prototype_1.service.IClassifier;
import com.example.automl_prototype_1.service.LogService;
import java.util.Arrays;
import java.util.List;

public class SimulatedAnnealing extends AbstractFeatureSelectionAlgorithm {

    private final int maxIterations;
    private final double initialTemperature;
    private final double coolingRate;

    // --- State variables for the search ---
    private double[] currentSolution;
    private double currentFitness;
    private double[] bestSolution;
    private double bestFitness;
    private double temperature;

    public SimulatedAnnealing(int maxIterations, double initialTemperature, double coolingRate) {
        super(5, 20); // Default kFolds and patience
        this.maxIterations = maxIterations;
        this.initialTemperature = initialTemperature;
        this.coolingRate = coolingRate;
    }

    // Default constructor chaining
    public SimulatedAnnealing() {
        this(30, 100.0, 0.995); // Adjusted to match existing constructor signature
    }

    // Constructor for HyperparameterTuningService
    public SimulatedAnnealing(int populationSize, int maxIterations) {
        this(maxIterations, 100.0, 0.995); // Ignore populationSize, use default temp/cooling
    }

    @Override
    public String getAlgorithmName() {
        return "Simulated Annealing (SA)";
    }

    @Override
    protected int getMaxIterations() {
        return this.maxIterations;
    }

    @Override
    protected double[] getBestSolution() {
        return this.bestSolution;
    }

    @Override
    protected double getBestFitness() {
        return this.bestFitness;
    }

    @Override
    protected double getAverageFitness() {
        // SA is single-solution. Current fitness represents the active search point.
        // This will fluctuate while Best Fitness stays monotonic.
        return this.currentFitness;
    }

    @Override
    protected void initialize(int numFeatures, Dataset dataset, IClassifier classifier, List<Fold> folds) {
        // 1. Create a single random starting solution
        this.currentSolution = new double[numFeatures];
        for (int i = 0; i < numFeatures; i++) {
            this.currentSolution[i] = random.nextDouble() < 0.5 ? 1.0 : 0.0;
        }

        // 2. Evaluate it and set initial state
        this.currentFitness = calculateFitness(this.currentSolution, classifier, folds);
        this.bestSolution = Arrays.copyOf(this.currentSolution, numFeatures);
        this.bestFitness = this.currentFitness;

        // 3. Initialize the temperature
        this.temperature = this.initialTemperature;

        LogService.info("SA", String.format("Initial Best Fitness: %.4f", this.bestFitness));
    }

    @Override
    protected void updatePopulation(int iteration, int numFeatures, Dataset dataset, IClassifier classifier,
            List<Fold> folds) {

        // Parallelize Simulated Annealing: Run multiple independent chains (Multi-Start
        // SA)
        // Since original SA only has 1 "current solution", we will treat
        // 'populationSize' (if used) as number of chains
        // If not using populationSize (SA is single solution), we can simulate it or
        // run parallel moves.
        // But standard SA is sequential Markov Chain.
        // Best approach for effectiveness: Run K independent chains in parallel for
        // this iteration block.
        // However, SA usually runs 'maxIterations' internally. Here 'updatePopulation'
        // is called once per iteration?
        // Ah, AbstractFeatureSelectionAlgorithm calls updatePopulation inside a loop
        // 0..maxIterations.
        // So this method represents ONE STEP (or batch of steps).

        // Single chain SA is hard to parallelize step-by-step.
        // Better Strategy: Run multiple independent chains (e.g. 5 threads) and sync
        // best result.
        // But the "currentSolution" is stateful.

        // Let's modify it to be a "Parallel Population-Based SA" (effectively just
        // multiple independent SA runners)
        // If 'currentSolution' is the only state, we need to make it a population.
        // But SA is defined as single-solution in this class structure?
        // Let's check init: SA has NO population in typical sense.
        // AbstractFeatureSelectionAlgorithm calls init, then loop.

        // To make it parallel effective:
        // We will perform multiple independent mutations (neighbors) in parallel, and
        // pick the best one (or accept based on Metropolis).
        // Parallel Tempering or Parallel Neighborhood Search.

        // Parallel Neighborhood Search Strategy:
        // 1. Generate N neighbors in parallel.
        // 2. Evaluate all.
        // 3. Pick the best accepted one (or just best one).

        int numNeighbors = 8; // Number of parallel neighbors to explore per step

        double temp = initialTemperature / (1 + Math.log(1 + iteration));

        // Parallel Neighbor Generation & Evaluation
        // Smart Parallelism
        List<NeighborResult> results = new java.util.ArrayList<>();
        try {
            results = this.executorService
                    .submit(() -> java.util.stream.IntStream.range(0, numNeighbors).parallel().mapToObj(i -> {
                        var rand = java.util.concurrent.ThreadLocalRandom.current();
                        double[] neighbor = java.util.Arrays.copyOf(currentSolution, numFeatures);
                        int flipIndex = rand.nextInt(numFeatures);
                        neighbor[flipIndex] = 1.0 - neighbor[flipIndex];

                        double fitness = calculateFitness(neighbor, classifier, folds);
                        return new NeighborResult(fitness, neighbor);
                    }).toList()).get();
        } catch (Exception e) {
            LogService.error("SA", "Error in parallel neighborhood search: " + e.getMessage());
        }

        // Select best neighbor (Greedy or Metropolis?)
        // Standard SA: Evaluate one, accept/reject.
        // Parallel SA: Pick best from batch, then accept/reject?
        // Or "Interaction": if any is better than current, take it?

        NeighborResult bestOfBatch = results.stream()
                .max(java.util.Comparator.comparingDouble(nr -> nr.fitness))
                .orElse(null);

        if (bestOfBatch != null) {
            double delta = bestOfBatch.fitness - currentFitness;
            if (delta > 0) {
                currentSolution = bestOfBatch.solution;
                currentFitness = bestOfBatch.fitness;
                LogService.info("SA", "Iteration " + (iteration + 1) + " | Improved: " + currentFitness);
            } else {
                if (Math.exp(delta / temp) > java.util.concurrent.ThreadLocalRandom.current().nextDouble()) {
                    currentSolution = bestOfBatch.solution;
                    currentFitness = bestOfBatch.fitness;
                    // Accepted worse solution
                }
            }
        }

        // 4. Update the best-ever-found solution
        if (this.currentFitness > this.bestFitness) {
            this.bestFitness = this.currentFitness;
            this.bestSolution = Arrays.copyOf(this.currentSolution, numFeatures);
        }

        // 5. Cool the temperature
        this.temperature *= this.coolingRate;

        if (iteration % 50 == 0 || iteration == 0 || iteration == maxIterations - 1) { // Print progress periodically
            LogService.info("SA",
                    String.format("Iteration %d | Temp: %.3f | Current Fitness: %.4f | Best Fitness: %.4f",
                            iteration, this.temperature, this.currentFitness, this.bestFitness));
        }
    }
}

class NeighborResult {
    double fitness;
    double[] solution;

    NeighborResult(double f, double[] s) {
        this.fitness = f;
        this.solution = s;
    }
}