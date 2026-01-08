package com.example.automl_prototype_1.algorithm;

import com.example.automl_prototype_1.service.ProgressTrackingService;
import com.example.automl_prototype_1.model.Dataset;
import com.example.automl_prototype_1.model.ExecutionResult;
import com.example.automl_prototype_1.service.IClassifier;
import com.example.automl_prototype_1.classifier.RandomForestClassifier;
import com.example.automl_prototype_1.classifier.SvmClassifier;
import com.example.automl_prototype_1.classifier.XGBoostClassifier;
import smile.validation.metric.Accuracy;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public abstract class AbstractFeatureSelectionAlgorithm implements OptimizationAlgorithm {

    private static final Logger logger = LoggerFactory.getLogger(AbstractFeatureSelectionAlgorithm.class);

    protected final Random random; // Instance-specific random
    protected final int kFolds;
    protected final int patience;
    private final ProgressTrackingService progressService = ProgressTrackingService.getInstance();
    // Cache for fitness values to avoid redundant calculations (Memoization)
    private final Map<String, Double> fitnessCache = new java.util.concurrent.ConcurrentHashMap<>();

    protected AbstractFeatureSelectionAlgorithm(int kFolds, int patience) {
        this.kFolds = kFolds > 1 ? kFolds : 2;
        this.patience = patience;
        // Use ThreadLocalRandom with current time to ensure different behavior across
        // algorithms
        this.random = new Random(System.nanoTime() + Thread.currentThread().hashCode());
    }

    @Override
    public final ExecutionResult optimize(Dataset dataset, IClassifier classifier, int trialNum, int totalTrials) {
        long startTime = System.currentTimeMillis();
        String algoName = getAlgorithmName();
        int maxIters = getMaxIterations();

        String targetColumn = dataset.getTargetColumnName();
        if (targetColumn == null || targetColumn.isEmpty()) {
            targetColumn = "Class";
        }
        final String TARGET_COLUMN = targetColumn;
        double[][] allFeatures = dataset.getFeatureMatrix(TARGET_COLUMN);
        int[] allLabels = dataset.getLabelVector(TARGET_COLUMN);

        if (allFeatures.length < kFolds) {
            throw new IllegalArgumentException("Dataset has fewer samples than k-folds.");
        }

        int numFeatures = allFeatures[0].length;

        IClassifier searchClassifier = getSearchClassifier(classifier);
        System.out.println("Using fast search classifier: " + searchClassifier.getName());

        List<Fold> folds = createStratifiedFolds(allFeatures, allLabels, this.kFolds);
        double bestFitnessSoFar = Double.NEGATIVE_INFINITY;
        int epochsWithoutImprovement = 0;

        progressService.jobStarted(algoName, trialNum, totalTrials, maxIters);

        initialize(numFeatures, dataset, searchClassifier, folds);

        List<Double> fitnessHistory = new ArrayList<>();

        for (int t = 0; t < getMaxIterations(); t++) {
            long iterStartTime = System.currentTimeMillis();
            updatePopulation(t, numFeatures, dataset, searchClassifier, folds);

            long iterTimeMs = System.currentTimeMillis() - iterStartTime;
            double currentBestFitness = getBestFitness();
            fitnessHistory.add(currentBestFitness); // Track history

            progressService.iterationComplete(algoName, trialNum, t + 1, maxIters, currentBestFitness, iterTimeMs);

            if (currentBestFitness > bestFitnessSoFar + 1e-6) {
                bestFitnessSoFar = currentBestFitness;
                epochsWithoutImprovement = 0;
            } else {
                epochsWithoutImprovement++;
            }

            if (epochsWithoutImprovement >= this.patience) {
                System.out.println("--- Early stopping triggered at iteration " + (t + 1) + " ---");
                break;
            }
        }

        progressService.jobFinished(getAlgorithmName());

        double[] bestSolution = getBestSolution();
        System.out.println(
                "Search complete. Evaluating best solution with high-precision classifier: " + classifier.getName());

        // --- Final Evaluation with Advanced Metrics ---
        double finalAccuracy = 0.0;
        double finalPrecision = 0.0;
        double finalRecall = 0.0;
        double finalF1 = 0.0;

        // Normalize labels to 0..N-1 for metrics and confusion matrix
        // Find unique labels and sort them to ensure deterministic mapping
        int[] uniqueLabels = Arrays.stream(allLabels).distinct().sorted().toArray();
        Map<Integer, Integer> labelMap = new HashMap<>();
        for (int i = 0; i < uniqueLabels.length; i++) {
            labelMap.put(uniqueLabels[i], i);
        }
        int numClasses = uniqueLabels.length;
        int[][] confusionMatrix = new int[numClasses][numClasses];

        try {
            // Use the original classifier for final evaluation
            for (int i = 0; i < this.kFolds; i++) {
                List<double[]> trainFeaturesList = new ArrayList<>();
                List<Integer> trainLabelsList = new ArrayList<>();

                double[][] testFeaturesWithSelection = filterFeatures(folds.get(i).features, bestSolution);
                int[] testLabels = folds.get(i).labels;

                if (testFeaturesWithSelection.length == 0)
                    continue;

                for (int j = 0; j < this.kFolds; j++) {
                    if (i == j)
                        continue;
                    trainFeaturesList.addAll(Arrays.asList(filterFeatures(folds.get(j).features, bestSolution)));
                    for (int label : folds.get(j).labels) {
                        trainLabelsList.add(label);
                    }
                }

                if (trainFeaturesList.isEmpty())
                    continue;

                double[][] trainFeatures = trainFeaturesList.toArray(new double[0][]);
                int[] trainLabels = trainLabelsList.stream().mapToInt(Integer::intValue).toArray();

                classifier.train(trainFeatures, trainLabels);
                int[] predictions = classifier.predict(testFeaturesWithSelection);

                finalAccuracy += Accuracy.of(testLabels, predictions);

                // Map labels and predictions to 0..N-1 range for metrics
                int[] mappedTestLabels = Arrays.stream(testLabels).map(l -> labelMap.getOrDefault(l, -1)).toArray();
                int[] mappedPredictions = Arrays.stream(predictions).map(l -> labelMap.getOrDefault(l, -1)).toArray();

                // Update Confusion Matrix
                for (int k = 0; k < mappedTestLabels.length; k++) {
                    int trueLabel = mappedTestLabels[k];
                    int predLabel = mappedPredictions[k];
                    if (trueLabel >= 0 && trueLabel < numClasses && predLabel >= 0 && predLabel < numClasses) {
                        confusionMatrix[trueLabel][predLabel]++;
                    }
                }

                // Calculate metrics using mapped labels
                // Check if binary (2 classes) to use binary metrics, otherwise use macro/micro
                if (numClasses == 2) {
                    // For binary, Smile expects 0 and 1. Our mapping guarantees 0 and 1.
                    finalPrecision += smile.validation.metric.Precision.of(mappedTestLabels, mappedPredictions);
                    finalRecall += smile.validation.metric.Recall.of(mappedTestLabels, mappedPredictions);
                    finalF1 += smile.validation.metric.FScore.of(1.0, mappedTestLabels, mappedPredictions);
                } else {
                    // Multiclass: Calculate Macro-Average manually to ensure correctness
                    double sumPrecision = 0.0;
                    double sumRecall = 0.0;
                    double sumF1 = 0.0;

                    for (int c = 0; c < numClasses; c++) {
                        int tp = 0;
                        int fp = 0;
                        int fn = 0;

                        for (int k = 0; k < mappedTestLabels.length; k++) {
                            int trueLbl = mappedTestLabels[k];
                            int predLbl = mappedPredictions[k];

                            if (trueLbl == c && predLbl == c)
                                tp++;
                            if (trueLbl != c && predLbl == c)
                                fp++;
                            if (trueLbl == c && predLbl != c)
                                fn++;
                        }

                        double p = (tp + fp) > 0 ? (double) tp / (tp + fp) : 0.0;
                        double r = (tp + fn) > 0 ? (double) tp / (tp + fn) : 0.0;
                        double f = (p + r) > 0 ? 2 * p * r / (p + r) : 0.0;

                        sumPrecision += p;
                        sumRecall += r;
                        sumF1 += f;
                    }

                    finalPrecision += sumPrecision / numClasses;
                    finalRecall += sumRecall / numClasses;
                    finalF1 += sumF1 / numClasses;
                }
            }
            finalAccuracy /= this.kFolds;
            finalPrecision /= this.kFolds;
            finalRecall /= this.kFolds;
            finalF1 /= this.kFolds;

        } catch (Exception e) {
            logger.error("Error during final evaluation", e);
        }

        long endTime = System.currentTimeMillis();
        int selectedFeaturesCount = 0;
        if (bestSolution != null) {
            for (double feature : bestSolution) {
                if (feature == 1.0) {
                    selectedFeaturesCount++;
                }
            }
        }

        return new ExecutionResult(getAlgorithmName(), finalAccuracy, finalPrecision, finalRecall, finalF1,
                numFeatures, selectedFeaturesCount, endTime - startTime, fitnessHistory, confusionMatrix);
    }

    @Override
    public final ExecutionResult execute(Dataset dataset, Map<String, Object> configuration) {
        if (configuration == null || !(configuration.get("classifier") instanceof IClassifier)) {
            throw new IllegalArgumentException("A valid classifier must be provided.");
        }
        IClassifier classifier = (IClassifier) configuration.get("classifier");
        return this.optimize(dataset, classifier, 1, 1);
    }

    protected abstract void initialize(int numFeatures, Dataset dataset, IClassifier classifier, List<Fold> folds);

    protected abstract void updatePopulation(int iteration, int numFeatures, Dataset dataset, IClassifier classifier,
            List<Fold> folds);

    protected abstract double[] getBestSolution();

    protected abstract double getBestFitness();

    protected abstract int getMaxIterations();

    protected double calculateFitness(double[] position, IClassifier classifier, List<Fold> folds) {
        try {
            double lowCardinalityBonus = 0.0;
            double[][] sampleFeatures = folds.get(0).features;

            for (int j = 0; j < position.length; j++) {
                if (position[j] == 1.0) {
                    Set<Double> uniqueValues = new HashSet<>();
                    for (int row = 0; row < sampleFeatures.length; row++) {
                        uniqueValues.add(sampleFeatures[row][j]);
                        if (uniqueValues.size() > 10)
                            break;
                    }
                    if (uniqueValues.size() <= 10) {
                        lowCardinalityBonus += 0.01;
                    }
                }
            }

            double penalty = 0.0;
            double totalAccuracy = 0.0;

            String cacheKey = Arrays.toString(position);
            if (fitnessCache.containsKey(cacheKey)) {
                return fitnessCache.get(cacheKey);
            }

            int searchFolds = this.kFolds;

            for (int i = 0; i < searchFolds; i++) {
                List<double[]> trainFeaturesList = new ArrayList<>();
                List<Integer> trainLabelsList = new ArrayList<>();

                double[][] testFeaturesWithSelection = filterFeatures(folds.get(i).features, position);
                int[] testLabels = folds.get(i).labels;

                if (testFeaturesWithSelection.length > 0 && testFeaturesWithSelection[0].length == 0) {
                    return 0.0;
                }

                for (int j = 0; j < this.kFolds; j++) {
                    if (i == j)
                        continue;
                    trainFeaturesList.addAll(Arrays.asList(filterFeatures(folds.get(j).features, position)));
                    for (int label : folds.get(j).labels) {
                        trainLabelsList.add(label);
                    }
                }

                if (trainFeaturesList.isEmpty() || testFeaturesWithSelection.length == 0) {
                    continue;
                }

                double[][] trainFeatures = trainFeaturesList.toArray(new double[0][]);
                int[] trainLabels = trainLabelsList.stream().mapToInt(Integer::intValue).toArray();

                classifier.train(trainFeatures, trainLabels);
                int[] predictions = classifier.predict(testFeaturesWithSelection);

                boolean useBalanced = com.example.automl_prototype_1.service.AppStateService.getInstance()
                        .isBalancedOptimizationEnabled();

                if (useBalanced) {
                    java.util.Set<Integer> uniqueClasses = new java.util.HashSet<>();
                    for (int t : testLabels)
                        uniqueClasses.add(t);
                    double sumRecall = 0.0;
                    for (int cls : uniqueClasses) {
                        int tp = 0, fn = 0;
                        for (int k = 0; k < testLabels.length; k++) {
                            if (testLabels[k] == cls) {
                                if (predictions[k] == cls)
                                    tp++;
                                else
                                    fn++;
                            }
                        }
                        double recall = (tp + fn) > 0 ? (double) tp / (tp + fn) : 0.0;
                        sumRecall += recall;
                    }
                    double balancedAcc = uniqueClasses.isEmpty() ? 0.0 : sumRecall / uniqueClasses.size();
                    totalAccuracy += balancedAcc;
                } else {
                    totalAccuracy += Accuracy.of(testLabels, predictions);
                }

                // Calculate F1
                java.util.Set<Integer> uniqueClasses = new java.util.HashSet<>();
                for (int t : testLabels)
                    uniqueClasses.add(t);
                double sumF1 = 0.0;
                for (int cls : uniqueClasses) {
                    int tp = 0, fp = 0, fn = 0;
                    for (int k = 0; k < testLabels.length; k++) {
                        int trueLabel = testLabels[k];
                        int predLabel = predictions[k];
                        if (trueLabel == cls && predLabel == cls)
                            tp++;
                        if (trueLabel != cls && predLabel == cls)
                            fp++;
                        if (trueLabel == cls && predLabel != cls)
                            fn++;
                    }
                    double p = (tp + fp) > 0 ? (double) tp / (tp + fp) : 0.0;
                    double r = (tp + fn) > 0 ? (double) tp / (tp + fn) : 0.0;
                    double f1 = (p + r) > 0 ? 2 * p * r / (p + r) : 0.0;
                    sumF1 += f1;
                }
                double macroF1 = uniqueClasses.isEmpty() ? 0.0 : sumF1 / uniqueClasses.size();
                totalAccuracy += macroF1;
            }

            // Combined Metric = Average of (Acc + F1) / 2
            double combinedMetric = (totalAccuracy / searchFolds) / 2.0;

            lowCardinalityBonus = Math.min(lowCardinalityBonus, 0.05);

            double fitness = Math.max(0.0, combinedMetric - penalty + lowCardinalityBonus);

            fitnessCache.put(cacheKey, fitness);
            return fitness;

        } catch (Exception e) {
            return 0.0;
        }
    }

    protected static class Fold {
        final double[][] features;
        final int[] labels;

        Fold(double[][] features, int[] labels) {
            this.features = features;
            this.labels = labels;
        }

    }

    private List<Fold> createStratifiedFolds(double[][] features, int[] labels, int k) {
        Map<Integer, List<Integer>> indicesByClass = IntStream.range(0, labels.length).boxed()
                .collect(Collectors.groupingBy(i -> labels[i]));

        List<List<Integer>> foldIndicesList = new ArrayList<>();
        for (int i = 0; i < k; i++) {
            foldIndicesList.add(new ArrayList<>());
        }

        for (List<Integer> classIndices : indicesByClass.values()) {
            Collections.shuffle(classIndices, this.random);
            for (int i = 0; i < classIndices.size(); i++) {
                foldIndicesList.get(i % k).add(classIndices.get(i));
            }
        }

        List<Fold> folds = new ArrayList<>();
        for (List<Integer> foldIndices : foldIndicesList) {
            double[][] foldFeatures = new double[foldIndices.size()][];
            int[] foldLabels = new int[foldIndices.size()];
            for (int i = 0; i < foldIndices.size(); i++) {
                int originalIndex = foldIndices.get(i);
                foldFeatures[i] = features[originalIndex];
                foldLabels[i] = labels[originalIndex];
            }
            folds.add(new Fold(foldFeatures, foldLabels));
        }
        return folds;
    }

    protected double[][] filterFeatures(double[][] originalFeatures, double[] position) {
        int numRecords = originalFeatures.length;
        if (numRecords == 0) {
            return new double[0][0];
        }

        int selectedFeatureCount = 0;
        for (double v : position) {
            if (v == 1.0) {
                selectedFeatureCount++;
            }
        }

        if (selectedFeatureCount == 0) {
            return new double[numRecords][0];
        }

        double[][] filteredFeatures = new double[numRecords][selectedFeatureCount];
        for (int i = 0; i < numRecords; i++) {
            int newColIndex = 0;
            for (int j = 0; j < originalFeatures[0].length; j++) {
                if (position[j] == 1.0) {
                    filteredFeatures[i][newColIndex++] = originalFeatures[i][j];
                }
            }
        }
        return filteredFeatures;
    }

    protected IClassifier getSearchClassifier(IClassifier originalClassifier) {
        if (originalClassifier instanceof RandomForestClassifier) {
            logger.info("Using search version of Random Forest (40 trees).");
            return new RandomForestClassifier(40);
        }
        if (originalClassifier instanceof SvmClassifier) {
            logger.info("Using search version of SVM (Linear Kernel).");
            // Standard tolerance for stability
            return new SvmClassifier("linear", 1.0, 1e-3);
        }
        if (originalClassifier instanceof XGBoostClassifier) {
            logger.info("Using search version of XGBoost (10 trees, depth 3).");
            // Balanced optimization
            return new XGBoostClassifier(10, 3);
        }
        return originalClassifier;
    }
}
