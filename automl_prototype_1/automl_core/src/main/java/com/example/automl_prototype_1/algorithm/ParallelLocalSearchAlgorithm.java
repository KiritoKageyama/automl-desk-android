package com.example.automl_prototype_1.algorithm;

import com.example.automl_prototype_1.model.Dataset;
import com.example.automl_prototype_1.service.IClassifier;
import java.util.Arrays;
import java.util.List;
import java.util.stream.IntStream;

public class ParallelLocalSearchAlgorithm extends AbstractFeatureSelectionAlgorithm {

    private final int maxIterations;
    private double[] currentSolution;
    private double currentFitness;
    private boolean converged = false;

    // Record to hold evaluation result
    private record NeighborResult(int featureIndex, double fitness, double[] solution) {
    }

    public ParallelLocalSearchAlgorithm(int maxIterations, int kFolds, int patience) {
        super(kFolds, patience);
        this.maxIterations = maxIterations;
    }

    public ParallelLocalSearchAlgorithm(int maxIterations) {
        this(maxIterations, 5, 20);
    }

    public ParallelLocalSearchAlgorithm() {
        this(200);
    }

    @Override
    public String getAlgorithmName() {
        return "Parallel Local Search (PLS)";
    }

    @Override
    protected int getMaxIterations() {
        return this.maxIterations;
    }

    @Override
    protected double[] getBestSolution() {
        return this.currentSolution;
    }

    @Override
    protected double getBestFitness() {
        return this.currentFitness;
    }

    @Override
    protected void initialize(int numFeatures, Dataset dataset, IClassifier classifier, List<Fold> folds) {
        currentSolution = new double[numFeatures];
        for (int j = 0; j < numFeatures; j++) {
            currentSolution[j] = random.nextDouble() < 0.5 ? 0 : 1;
        }
        currentFitness = calculateFitness(currentSolution, classifier, folds);
        System.out.println("PLS Initial Fitness: " + currentFitness);
    }

    @Override
    protected void updatePopulation(int iteration, int numFeatures, Dataset dataset, IClassifier classifier,
            List<Fold> folds) {
        if (converged)
            return;

        // Parallel Steepest Ascent: Evaluate ALL neighbors
        NeighborResult bestNeighbor = IntStream.range(0, numFeatures).parallel().mapToObj(j -> {
            double[] neighborSolution = Arrays.copyOf(currentSolution, numFeatures);
            neighborSolution[j] = 1.0 - neighborSolution[j]; // Flip feature j
            double fitness = calculateFitness(neighborSolution, classifier, folds);
            return new NeighborResult(j, fitness, neighborSolution);
        }).reduce((n1, n2) -> n1.fitness > n2.fitness ? n1 : n2)
                .orElse(null);

        if (bestNeighbor != null && bestNeighbor.fitness > currentFitness) {
            currentSolution = bestNeighbor.solution;
            currentFitness = bestNeighbor.fitness;
            System.out.println("Iteration " + (iteration + 1) + " | Improved by flipping feature "
                    + bestNeighbor.featureIndex + ". New Fitness: " + currentFitness);
        } else {
            System.out.println("Converged at iteration " + (iteration + 1));
            converged = true;
        }
    }
}
