package com.example.automl_prototype_1.algorithm;

import com.example.automl_prototype_1.model.Dataset;
import com.example.automl_prototype_1.service.IClassifier;
import com.example.automl_prototype_1.service.LogService;

import java.util.List;
import java.util.concurrent.ThreadLocalRandom;
import java.util.stream.IntStream;

public class GreyWolfOptimizer extends AbstractFeatureSelectionAlgorithm {

    private final int populationSize;
    private final int maxIterations;

    private double[][] positions;
    private double[] fitnesses;

    // Alpha, Beta, Delta wolves (Best 3 solutions)
    private double[] alphaPos;
    private double alphaScore = Double.NEGATIVE_INFINITY;

    private double[] betaPos;
    private double betaScore = Double.NEGATIVE_INFINITY;

    private double[] deltaPos;
    private double deltaScore = Double.NEGATIVE_INFINITY;

    public GreyWolfOptimizer(int populationSize, int maxIterations, int kFolds, int patience) {
        super(kFolds, patience);

        int tempPopSize = populationSize;
        try {
            com.example.automl_prototype_1.service.AppStateService appState = com.example.automl_prototype_1.service.AppStateService
                    .getInstance();
            if (appState.isAdvancedMode()) {
                Double confPop = appState.getAlgorithmHyperparameter("GWO", "populationSize");
                if (confPop != null) {
                    tempPopSize = confPop.intValue();
                    LogService.info("GWO", "Initialized with Advanced Params: Pop=" + tempPopSize);
                }
            }
        } catch (NoClassDefFoundError | Exception e) {
        }

        this.populationSize = tempPopSize;
        this.maxIterations = maxIterations;
    }

    public GreyWolfOptimizer(int populationSize, int maxIterations) {
        this(populationSize, maxIterations, 5, 20);
    }

    public GreyWolfOptimizer() {
        this(30, 100, 5, 20);
    }

    @Override
    public String getAlgorithmName() {
        return "Grey Wolf Optimizer (GWO)";
    }

    @Override
    protected int getMaxIterations() {
        return this.maxIterations;
    }

    @Override
    protected double[] getBestSolution() {
        return this.alphaPos;
    }

    @Override
    protected double getBestFitness() {
        return this.alphaScore;
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
        this.positions = new double[populationSize][numFeatures];
        this.fitnesses = new double[populationSize];

        this.alphaPos = new double[numFeatures];
        this.betaPos = new double[numFeatures];
        this.deltaPos = new double[numFeatures];

        // Init population
        for (int i = 0; i < populationSize; i++) {
            for (int j = 0; j < numFeatures; j++) {
                positions[i][j] = random.nextDouble() < 0.5 ? 0 : 1;
            }
            // Initial fitness
            fitnesses[i] = calculateFitness(positions[i], classifier, folds);
            updateHierarchy(positions[i], fitnesses[i]);
        }
        LogService.info("GWO", "Initial Alpha Score: " + alphaScore);
    }

    private void updateHierarchy(double[] pos, double score) {
        if (score > alphaScore) {
            // Demote Alpha to Beta, Beta to Delta
            deltaScore = betaScore;
            System.arraycopy(betaPos, 0, deltaPos, 0, betaPos.length);

            betaScore = alphaScore;
            System.arraycopy(alphaPos, 0, betaPos, 0, alphaPos.length);

            alphaScore = score;
            System.arraycopy(pos, 0, alphaPos, 0, pos.length);
        } else if (score > betaScore) {
            // Demote Beta to Delta
            deltaScore = betaScore;
            System.arraycopy(betaPos, 0, deltaPos, 0, betaPos.length);

            betaScore = score;
            System.arraycopy(pos, 0, betaPos, 0, pos.length);
        } else if (score > deltaScore) {
            deltaScore = score;
            System.arraycopy(pos, 0, deltaPos, 0, pos.length);
        }
    }

    @Override
    protected void updatePopulation(int iteration, int numFeatures, Dataset dataset, IClassifier classifier,
            List<Fold> folds) {
        // a decreases linearly from 2 to 0
        double a = 2.0 - (2.0 * iteration / maxIterations);

        // Update positions using Alpha, Beta, Delta

        // Parallel update calculation
        // Note: ThreadLocalRandom is needed inside parallel stream
        // Smart Parallelism Refactor: Calculate Fitness Inside Thread Pool
        int optimalThreads = com.example.automl_prototype_1.gpu.HardwareAcceleratorService
                .calculateOptimalThreadCount();
        @SuppressWarnings("resource")
        java.util.concurrent.ForkJoinPool customThreadPool = new java.util.concurrent.ForkJoinPool(optimalThreads);

        record GwoResult(int index, double[] newPosition, double fitness) {
        }

        List<GwoResult> results = new java.util.ArrayList<>();
        try {
            results = customThreadPool.submit(() -> IntStream.range(0, populationSize).parallel().mapToObj(i -> {
                double[] newPos = new double[numFeatures];
                var rand = ThreadLocalRandom.current();
                for (int j = 0; j < numFeatures; j++) {
                    // Update based on Alpha
                    double r1 = rand.nextDouble();
                    double r2 = rand.nextDouble();
                    double A1 = 2 * a * r1 - a;
                    double C1 = 2 * r2;
                    double D_alpha = Math.abs(C1 * alphaPos[j] - positions[i][j]);
                    double X1 = alphaPos[j] - A1 * D_alpha;

                    // Update based on Beta
                    r1 = rand.nextDouble();
                    r2 = rand.nextDouble();
                    double A2 = 2 * a * r1 - a;
                    double C2 = 2 * r2;
                    double D_beta = Math.abs(C2 * betaPos[j] - positions[i][j]);
                    double X2 = betaPos[j] - A2 * D_beta;

                    // Update based on Delta
                    r1 = rand.nextDouble();
                    r2 = rand.nextDouble();
                    double A3 = 2 * a * r1 - a;
                    double C3 = 2 * r2;
                    double D_delta = Math.abs(C3 * deltaPos[j] - positions[i][j]);
                    double X3 = deltaPos[j] - A3 * D_delta;

                    // Average
                    double continuousX = (X1 + X2 + X3) / 3.0;

                    // Discretization (Sigmoid)
                    double sigmoid = 1.0 / (1.0 + Math.exp(-continuousX));
                    newPos[j] = rand.nextDouble() < sigmoid ? 1.0 : 0.0;
                }
                double fit = calculateFitness(newPos, classifier, folds);
                return new GwoResult(i, newPos, fit);
            }).toList()).get();
        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            customThreadPool.shutdown();
        }

        // Evaluate and Update Hierarchy (Sequential update from results)
        if (results != null) {
            for (GwoResult res : results) {
                positions[res.index] = res.newPosition;
                fitnesses[res.index] = res.fitness;
                updateHierarchy(positions[res.index], res.fitness);
            }
        }

        LogService.info("GWO", "Iteration " + (iteration + 1) + " | Alpha Score: " + alphaScore);
    }
}
