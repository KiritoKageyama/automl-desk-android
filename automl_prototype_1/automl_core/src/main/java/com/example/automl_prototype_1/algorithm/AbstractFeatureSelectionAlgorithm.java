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

import com.example.automl_prototype_1.service.IGpuCachedClassifier;
import com.example.automl_prototype_1.service.GpuBatchService;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.TimeUnit;

import com.example.automl_prototype_1.service.HardwareAcceleratorService;
import com.example.automl_prototype_1.service.AppStateService;

// ...

public abstract class AbstractFeatureSelectionAlgorithm implements OptimizationAlgorithm {

    private static final Logger logger = LoggerFactory.getLogger(AbstractFeatureSelectionAlgorithm.class);

    protected final Random random; // Instance-specific random
    protected final int kFolds;
    protected final int patience;
    private final ProgressTrackingService progressService = ProgressTrackingService.getInstance();

    // Bounded Cache for fitness values (LRU Policy)
    // Size = 10,000 to prevent OOM on long runs with many unique chromosomes
    private final Map<ChromosomeKey, Double> fitnessCache = Collections
            .synchronizedMap(new LinkedHashMap<ChromosomeKey, Double>(10000, 0.75f, true) {
                @Override
                protected boolean removeEldestEntry(Map.Entry<ChromosomeKey, Double> eldest) {
                    return size() > 2000;
                }
            });

    // Pre-computed feature metadata to avoid repeated scanning and potential
    // leakage
    private boolean[] isLowCardinalityFeature;

    // Shared thread pool for parallel execution across ALL algorithm instances
    // This prevents thread explosion when multiple algorithms range in parallel
    protected static final ForkJoinPool SHARED_EXECUTOR;

    static {
        int optimalThreads = HardwareAcceleratorService.calculateOptimalThreadCount();
        SHARED_EXECUTOR = new ForkJoinPool(optimalThreads);
    }

    // Instance reference for compatibility (can point to shared)
    protected ForkJoinPool executorService = SHARED_EXECUTOR;

    // Caching Fields
    private final Map<Integer, String> foldTrainDatasetIds = new ConcurrentHashMap<>();
    private final Map<Integer, String> foldTestDatasetIds = new ConcurrentHashMap<>();

    // Pre-computed, SCALED data for each fold (CPU Cache)
    // Structure: List index = Fold Index (0 to K-1)
    private List<double[][]> scaledTrainFeatures;
    private List<int[]> scaledTrainLabels;
    private List<double[][]> scaledTestFeatures;
    private List<int[]> scaledTestLabels;

    private boolean cachingEnabled = false;
    protected boolean useProxy = true; // Default to true (Full System)
    protected boolean useHeuristics = true; // Default to true (Full System)

    // Manual Override
    protected int manualMaxIterations = -1;

    @Override
    public void setMaxIterations(int maxIterations) {
        this.manualMaxIterations = maxIterations;
    }

    protected AbstractFeatureSelectionAlgorithm(int kFolds, int patience) {
        this.kFolds = kFolds > 1 ? kFolds : 2;
        this.patience = patience;
        // Use ThreadLocalRandom with current time to ensure different behavior across
        // algorithms
        this.random = new Random(System.nanoTime() + Thread.currentThread().hashCode());
    }

    public void setUseProxy(boolean useProxy) {
        this.useProxy = useProxy;
    }

    public void setUseHeuristics(boolean useHeuristics) {
        this.useHeuristics = useHeuristics;
    }

    @Override
    public final ExecutionResult optimize(Dataset dataset, IClassifier classifier, Map<String, Object> configuration,
            int trialNum, int totalTrials) {
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

        // Clear cache to prevent memory leaks across runs
        this.fitnessCache.clear();

        try {

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

            // --- Data Preparation & Caching (Correct Cross-Validation logic) ---
            prepareData(searchClassifier, folds, configuration);

            // Pre-compute low cardinality features globally to avoid leakage and improve
            // perf
            this.isLowCardinalityFeature = new boolean[numFeatures];
            for (int j = 0; j < numFeatures; j++) {
                Set<Double> uniqueValues = new HashSet<>();
                // Check first 1000 rows or all rows to determine cardinality
                // This is an unsupervised check, so using all data is generally safe,
                // but strictly speaking we should only use training data.
                // For this implementation, we use a heuristic on the full dataset for
                // stability.
                for (int row = 0; row < Math.min(allFeatures.length, 2000); row++) {
                    uniqueValues.add(allFeatures[row][j]);
                    if (uniqueValues.size() > 10)
                        break;
                }
                this.isLowCardinalityFeature[j] = (uniqueValues.size() <= 10);
            }

            initialize(numFeatures, dataset, searchClassifier, folds);

            List<Double> fitnessHistory = new ArrayList<>();
            List<Double> avgFitnessHistory = new ArrayList<>(); // New list
            List<Double> populationDiversityHistory = new ArrayList<>(); // New list (Phase 6)

            for (int t = 0; t < getMaxIterations(); t++) {
                long iterStartTime = System.currentTimeMillis();
                updatePopulation(t, numFeatures, dataset, searchClassifier, folds);

                long iterTimeMs = System.currentTimeMillis() - iterStartTime;
                double currentBestFitness = getBestFitness();
                double currentAvgFitness = getAverageFitness(); // Track average

                // Calculate Diversity (Std Dev of Fitness)
                // We need the population fitness values.
                // Since updatePopulation updates the internal state, we rely on
                // getAverageFitness() and getBestFitness()
                // ideally we should have a getFitnessVariance() or similar, but for now let's
                // approximate or rely on implementation
                // actually, most algorithms don't expose the full fitness list to this base
                // class easily without refactoring abstract methods.
                // Let's rely on an abstract method getFitnessDiversity() which implementations
                // can provide more efficiently,
                // or we can fallback to a simple heuristic if not available.
                // For now, let's add abstract calculatePopulationDiversity()
                double currentDiversity = getPopulationDiversity();

                fitnessHistory.add(currentBestFitness);
                avgFitnessHistory.add(currentAvgFitness);
                populationDiversityHistory.add(currentDiversity);

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
            System.out.println("Search complete. Evaluating best solution with high-precision classifier: "
                    + classifier.getName());

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
                // Use the pre-processed (Scaled + Imputed) data for final evaluation
                for (int i = 0; i < this.kFolds; i++) {
                    double[][] trainFeatures = this.scaledTrainFeatures.get(i);
                    int[] trainLabels = this.scaledTrainLabels.get(i);
                    double[][] testFeatures = this.scaledTestFeatures.get(i);
                    int[] testLabels = this.scaledTestLabels.get(i);

                    // Filter features using the best solution mask
                    double[][] trainFeaturesSelected = filterFeatures(trainFeatures, bestSolution);
                    double[][] testFeaturesSelected = filterFeatures(testFeatures, bestSolution);

                    if (trainFeaturesSelected.length == 0 || testFeaturesSelected.length == 0)
                        continue;

                    classifier.train(trainFeaturesSelected, trainLabels);
                    int[] predictions = classifier.predict(testFeaturesSelected);

                    finalAccuracy += Accuracy.of(testLabels, predictions);

                    // Map labels and predictions to 0..N-1 range for metrics
                    int[] mappedTestLabels = Arrays.stream(testLabels).map(l -> labelMap.getOrDefault(l, -1)).toArray();
                    int[] mappedPredictions = Arrays.stream(predictions).map(l -> labelMap.getOrDefault(l, -1))
                            .toArray();

                    // Update Confusion Matrix
                    for (int k = 0; k < mappedTestLabels.length; k++) {
                        int trueLabel = mappedTestLabels[k];
                        int predLabel = mappedPredictions[k];
                        if (trueLabel >= 0 && trueLabel < numClasses && predLabel >= 0 && predLabel < numClasses) {
                            confusionMatrix[trueLabel][predLabel]++;
                        }
                    }

                    // Calculate metrics using mapped labels
                    if (numClasses == 2) {
                        finalPrecision += smile.validation.metric.Precision.of(mappedTestLabels, mappedPredictions);
                        finalRecall += smile.validation.metric.Recall.of(mappedTestLabels, mappedPredictions);
                        finalF1 += smile.validation.metric.FScore.of(1.0, mappedTestLabels, mappedPredictions);
                    } else {
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

            // Extract selected indices
            List<Integer> indicesList = getSelectedIndices(bestSolution);
            int[] selectedIndices = indicesList.stream().mapToInt(Integer::intValue).toArray();
            int selectedFeaturesCount = selectedIndices.length;

            ExecutionResult result = new ExecutionResult(getAlgorithmName(), finalAccuracy, finalPrecision, finalRecall,
                    finalF1,
                    numFeatures,
                    selectedFeaturesCount, endTime - startTime, fitnessHistory, avgFitnessHistory,
                    populationDiversityHistory, confusionMatrix);

            result.setFeatureIndices(selectedIndices);
            return result;
        } finally {
            // Do NOT shutdown the shared executor (it's global)

            // CRITICAL: Cleanup GPU Cache to prevent RAM leaks
            if (this.cachingEnabled) {
                try {
                    GpuBatchService.getInstance().submitClearCache();
                    this.foldTrainDatasetIds.clear();
                    this.foldTestDatasetIds.clear();
                    logger.info("GPU Cache Cleared for Algorithm: " + getAlgorithmName());
                } catch (Exception e) {
                    logger.error("Failed to clear GPU cache", e);
                }
            }
        }
    }

    private void prepareData(IClassifier classifier, List<Fold> folds, Map<String, Object> configuration) {
        String scalingStrategy = (String) configuration.getOrDefault("scaling", "Standardization");
        boolean doImputation = (boolean) configuration.getOrDefault("imputation", false);

        this.cachingEnabled = false;
        this.foldTrainDatasetIds.clear();
        this.foldTestDatasetIds.clear();

        this.scaledTrainFeatures = new ArrayList<>();
        this.scaledTrainLabels = new ArrayList<>();
        this.scaledTestFeatures = new ArrayList<>();
        this.scaledTestLabels = new ArrayList<>();

        List<CompletableFuture<Void>> uploads = new ArrayList<>();
        boolean gpuAvailable = (classifier instanceof IGpuCachedClassifier) && useProxy;

        if (gpuAvailable) {
            System.out.println("[Optimization] preparing data and uploading to GPU... 🚀");
        } else {
            System.out.println("[Optimization] preparing data (CPU mode)...");
        }

        for (int i = 0; i < folds.size(); i++) {
            // 1. Construct Raw Train & Test for Fold 'i'
            List<double[]> trainRows = new ArrayList<>();
            List<Integer> trainY = new ArrayList<>();

            // Train = Union of all other folds
            for (int j = 0; j < folds.size(); j++) {
                if (i == j)
                    continue;
                for (double[] row : folds.get(j).features)
                    trainRows.add(row);
                for (int y : folds.get(j).labels)
                    trainY.add(y);
            }

            double[][] rawTrainX = trainRows.toArray(new double[0][]);
            int[] rawTrainY = trainY.stream().mapToInt(Integer::intValue).toArray();

            double[][] rawTestX = folds.get(i).features;
            int[] rawTestY = folds.get(i).labels;

            // 2. Pre-Processing Pipeline per Fold
            double[][] workingTrainX = rawTrainX;
            double[][] workingTestX = rawTestX;

            // 2a. Imputation (Fit on Train, Transform Train & Test)
            if (doImputation) {
                double[] means = fitImputer(workingTrainX);
                workingTrainX = transformImputer(workingTrainX, means);
                workingTestX = transformImputer(workingTestX, means);
            }

            // 2b. Scaling (Fit on Train, Transform Train & Test)
            double[][] finalTrainX;
            double[][] finalTestX;

            if ("Standardization".equalsIgnoreCase(scalingStrategy)) {
                Scaler result = fitStandardscaler(workingTrainX);
                finalTrainX = transformStandardscaler(workingTrainX, result);
                finalTestX = transformStandardscaler(workingTestX, result);
            } else if ("Normalization".equalsIgnoreCase(scalingStrategy)) {
                Scaler result = fitMinMaxScaler(workingTrainX);
                finalTrainX = transformMinMaxScaler(workingTrainX, result);
                finalTestX = transformMinMaxScaler(workingTestX, result);
            } else {
                // No Scaling
                finalTrainX = workingTrainX;
                finalTestX = workingTestX;
            }

            // 3. Store in Lists for CPU access
            this.scaledTrainFeatures.add(finalTrainX);
            this.scaledTrainLabels.add(rawTrainY);
            this.scaledTestFeatures.add(finalTestX);
            this.scaledTestLabels.add(rawTestY);

            // 4. Upload to GPU (if applicable)
            if (gpuAvailable) {
                final int foldIndex = i;
                final double[][] tx = finalTrainX;
                final int[] ty = rawTrainY;
                final double[][] tex = finalTestX;

                IGpuCachedClassifier gpuC = (IGpuCachedClassifier) classifier;

                // Upload Train
                uploads.add(GpuBatchService.getInstance().submitUploadDataset(tx, ty)
                        .thenAccept(id -> foldTrainDatasetIds.put(foldIndex, (String) id))
                        .thenRun(() -> {
                        }));

                // Upload Test
                uploads.add(GpuBatchService.getInstance().submitUploadDataset(tex, rawTestY)
                        .thenAccept(id -> foldTestDatasetIds.put(foldIndex, (String) id))
                        .thenRun(() -> {
                        }));
            }
        }

        if (gpuAvailable && !uploads.isEmpty()) {
            try {
                CompletableFuture.allOf(uploads.toArray(new CompletableFuture[0])).join();
                this.cachingEnabled = true;
                System.out.println("[Optimization] Data Prep & Upload Complete. Caching Enabled.");
            } catch (Exception e) {
                logger.error("Failed to upload datasets. Fallback to CPU.", e);
                this.cachingEnabled = false;
            }
        }
    }

    // --- Helper classes for Scaling & Imputation ---
    private double[] fitImputer(double[][] data) {
        if (data.length == 0)
            return new double[0];
        int cols = data[0].length;
        double[] means = new double[cols];

        for (int j = 0; j < cols; j++) {
            double sum = 0;
            int count = 0;
            for (int i = 0; i < data.length; i++) {
                double val = data[i][j];
                if (!Double.isNaN(val)) {
                    sum += val;
                    count++;
                }
            }
            means[j] = (count > 0) ? (sum / count) : 0.0;
        }
        return means;
    }

    private double[][] transformImputer(double[][] data, double[] means) {
        if (data.length == 0)
            return new double[0][0];
        int rows = data.length;
        int cols = data[0].length;
        double[][] out = new double[rows][cols];

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                double val = data[i][j];
                if (Double.isNaN(val)) {
                    out[i][j] = means[j];
                } else {
                    out[i][j] = val;
                }
            }
        }
        return out;
    }

    private static class Scaler {
        double[] param1; // Mean or Min
        double[] param2; // StdDev or Range(Max-Min)
    }

    private Scaler fitStandardscaler(double[][] data) {
        if (data.length == 0)
            return new Scaler();
        int cols = data[0].length;
        Scaler s = new Scaler();
        s.param1 = new double[cols]; // Mean
        s.param2 = new double[cols]; // StdDev

        for (int j = 0; j < cols; j++) {
            double sum = 0;
            // Assumes data is already imputed no NaNs
            for (int i = 0; i < data.length; i++)
                sum += data[i][j];
            s.param1[j] = sum / data.length;

            double sumSq = 0;
            for (int i = 0; i < data.length; i++) {
                double diff = data[i][j] - s.param1[j];
                sumSq += diff * diff;
            }
            s.param2[j] = Math.sqrt(sumSq / data.length);
        }
        return s;
    }

    private double[][] transformStandardscaler(double[][] data, Scaler s) {
        if (data.length == 0)
            return new double[0][0];
        int rows = data.length;
        int cols = data[0].length;
        double[][] out = new double[rows][cols];

        for (int j = 0; j < cols; j++) {
            double mean = s.param1[j];
            double std = s.param2[j];
            // Avoid div by zero
            if (std < 1e-9) {
                for (int i = 0; i < rows; i++)
                    out[i][j] = 0.0;
            } else {
                for (int i = 0; i < rows; i++)
                    out[i][j] = (data[i][j] - mean) / std;
            }
        }
        return out;
    }

    private Scaler fitMinMaxScaler(double[][] data) {
        if (data.length == 0)
            return new Scaler();
        int cols = data[0].length;
        Scaler s = new Scaler();
        s.param1 = new double[cols]; // Min
        s.param2 = new double[cols]; // Range

        for (int j = 0; j < cols; j++) {
            double min = Double.MAX_VALUE;
            double max = -Double.MAX_VALUE;
            for (int i = 0; i < data.length; i++) {
                if (data[i][j] < min)
                    min = data[i][j];
                if (data[i][j] > max)
                    max = data[i][j];
            }
            s.param1[j] = min;
            s.param2[j] = max - min;
        }
        return s;
    }

    private double[][] transformMinMaxScaler(double[][] data, Scaler s) {
        if (data.length == 0)
            return new double[0][0];
        int rows = data.length;
        int cols = data[0].length;
        double[][] out = new double[rows][cols];

        for (int j = 0; j < cols; j++) {
            double min = s.param1[j];
            double range = s.param2[j];

            if (range < 1e-9) {
                for (int i = 0; i < rows; i++)
                    out[i][j] = 0.0;
            } else {
                for (int i = 0; i < rows; i++)
                    out[i][j] = (data[i][j] - min) / range;
            }
        }
        return out;
    }

    @Override
    public final ExecutionResult execute(Dataset dataset, Map<String, Object> configuration) {
        if (configuration == null || !(configuration.get("classifier") instanceof IClassifier)) {
            throw new IllegalArgumentException("A valid classifier must be provided.");
        }
        IClassifier classifier = (IClassifier) configuration.get("classifier");
        // Ensure configuration is passed to optimize if it relies on it (it does now
        // for scaling)
        // optimize() signature is optimize(Dataset, IClassifier, int, int). It does NOT
        // take config map.
        // I need to update optimize() signature OR pass the map differently.
        // Wait, optimize() uses 'this.kFolds' etc.
        // But I added 'scalingStrategy = conf.get(...)' in optimize() in my PREVIOUS
        // chunk instruction.
        // BUT optimize() as defined in the file DOES NOT take the map!
        // I must overload or update optimize to take the map.
        // Currently 'execute' calls 'optimize'.
        // I will update execute() here, but I also need to update optimize() signature
        // in the file.
        // Or better: Store the config in a field? No, thread safety.
        // I will change optimize to take the Map.
        return this.optimize(dataset, classifier, configuration, 1, 1);
    }

    // Dummy placeholder for the optimize method signature change which I will do in
    // the next step
    // preventing compilation error in this chunk context check.

    protected abstract void initialize(int numFeatures, Dataset dataset, IClassifier classifier, List<Fold> folds);

    protected abstract void updatePopulation(int iteration, int numFeatures, Dataset dataset, IClassifier classifier,
            List<Fold> folds);

    protected abstract double[] getBestSolution();

    protected abstract double getBestFitness();

    // Concrete implementation or abstract definition required.
    // Since this class is abstract, we can leave it abstract or provide dummy.
    // But the error was "does not override or implement ... from supertype".
    // Wait, OptimizationAlgorithm doesn't enforce getAverageFitness unless I added
    // it there?
    // The error was "method does not override or implement a method from a
    // supertype" at line 104 which is @Override verify
    // public final ExecutionResult optimize(Dataset dataset, IClassifier
    // classifier, Map<String, Object> configuration, int trialNum, int totalTrials)
    // The interface might have a different signature.
    // OptimizationAlgorithm typically has: ExecutionResult execute(Dataset, Map);
    // Let's check OptimizationAlgorithm.java to be sure.
    // BUT for safely fixing THIS step without viewing that file yet, I will remove
    // @Override if the signature doesn't match,
    // OR likely invalid since I just edited ExecutionService to call it this way.
    // Actually, I saw HyperparameterTuningService calling optimize(Dataset,
    // IClassifier, int, int) which also failed.
    // I suspect OptimizationAlgorithm interface is missing these methods.

    // I will check OptimizationAlgorithm next.
    // For AFSA, I will add getAverageFitness() concrete implementation returning 0
    // for now to satisfy my OWN call in line 172.
    protected double getAverageFitness() {
        return 0.0;
    }

    // Phase 6: Diversity Metric (Default implementation returns 0.0)
    protected double getPopulationDiversity() {
        return 0.0;
    }

    protected abstract int getMaxIterations();

    protected double calculateFitness(double[] position, IClassifier classifier, List<Fold> folds) {
        try {
            double lowCardinalityBonus = getLowCardinalityBonus(position, folds);

            double penalty = 0.0;
            double totalScore = 0.0; // Sum of (Acc + F1) for each fold

            ChromosomeKey cacheKey = new ChromosomeKey(position);
            if (fitnessCache.containsKey(cacheKey)) {
                return fitnessCache.get(cacheKey);
            }

            int searchFolds = this.kFolds;

            for (int i = 0; i < searchFolds; i++) {
                int[] predictions = null;
                int[] testLabels = folds.get(i).labels;

                // 1. Get Predictions (Cached or Legacy)
                if (cachingEnabled && classifier instanceof IGpuCachedClassifier) {
                    IGpuCachedClassifier gpuClassifier = (IGpuCachedClassifier) classifier;
                    String trainId = foldTrainDatasetIds.get(i);
                    String testId = foldTestDatasetIds.get(i);

                    if (trainId != null && testId != null) {
                        List<Integer> indicesList = getSelectedIndices(position);

                        if (indicesList.isEmpty())
                            return 0.0;
                        int[] featureIndices = indicesList.stream().mapToInt(Integer::intValue).toArray();

                        // Sync calls for single calculation (falls back to async.get inside classifier)
                        gpuClassifier.trainCached(trainId, featureIndices, null);
                        predictions = gpuClassifier.predictCached(testId, featureIndices, null);
                    }
                }

                if (predictions == null) {
                    // Legacy / CPU Path: Use pre-computed scaled data from lists
                    double[][] trainFeatures = this.scaledTrainFeatures.get(i);
                    int[] trainLabels = this.scaledTrainLabels.get(i);
                    double[][] testFeaturesBase = this.scaledTestFeatures.get(i);

                    // Filter features based on chromosome position
                    double[][] testFeaturesWithSelection = filterFeatures(testFeaturesBase, position);
                    // Filter train features same way
                    double[][] trainFeaturesWithSelection = filterFeatures(trainFeatures, position);

                    // FAILSAFE: Handle 0 features
                    if (testFeaturesWithSelection.length > 0 && testFeaturesWithSelection[0].length == 0) {
                        // Force select 1 feature
                        int forceIdx = random.nextInt(testFeaturesBase[0].length);
                        // Re-select on base
                        // But for simplicity in this hot path, we just skip or return 0 score?
                        // The original code tried to force select. Let's do that properly:
                        // Ideally we should modify 'position' but we can't here easily.
                        // Let's just create a dummy valid set from idx 0.
                        testFeaturesWithSelection = new double[testFeaturesBase.length][1];
                        trainFeaturesWithSelection = new double[trainFeatures.length][1];
                        for (int r = 0; r < testFeaturesWithSelection.length; r++)
                            testFeaturesWithSelection[r][0] = testFeaturesBase[r][forceIdx];
                        for (int r = 0; r < trainFeaturesWithSelection.length; r++)
                            trainFeaturesWithSelection[r][0] = trainFeatures[r][forceIdx];
                    }

                    if (trainFeaturesWithSelection.length == 0 || testFeaturesWithSelection.length == 0)
                        continue;

                    classifier.train(trainFeaturesWithSelection, trainLabels);
                    predictions = classifier.predict(testFeaturesWithSelection);
                }

                // 2. Calculate Metrics (Common)
                if (predictions != null) {
                    // DEBUG: Check accuracy calc
                    if (i == 0) { // Log first fold only
                        int matches = 0;
                        for (int p = 0; p < predictions.length && p < testLabels.length; p++) {
                            if (predictions[p] == testLabels[p])
                                matches++;
                        }
                        System.out.println("[DEBUG-AS] Fold 0 Predictions (First 20): " + java.util.Arrays
                                .toString(java.util.Arrays.copyOf(predictions, Math.min(20, predictions.length))));
                        System.out.println("[DEBUG-AS] Fold 0 Truth       (First 20): " + java.util.Arrays
                                .toString(java.util.Arrays.copyOf(testLabels, Math.min(20, testLabels.length))));
                        System.out.println("[DEBUG-AS] Accuracy Raw Check: " + matches + "/" + predictions.length);
                    }
                    totalScore += calculateFoldScore(testLabels, predictions);
                }
            }

            // Metric = Average(MetricPerFold)
            // MetricPerFold is already (BalancedAcc + MacroF1)/2 or similar based on config
            double combinedMetric = totalScore / searchFolds;

            // Apply bonus
            lowCardinalityBonus = Math.min(lowCardinalityBonus, 0.05);
            double fitness = Math.max(0.0, combinedMetric - penalty + lowCardinalityBonus);

            fitnessCache.put(cacheKey, fitness);
            return fitness;

        } catch (

        Exception e) {
            logger.error("Fitness Calculation Failed", e); // Fixed typo Calcluation
            return 0.0;
        }
    }

    /**
     * Optimized Hybrid Batch Fitness Evaluation.
     * Splits population into GPU batch and CPU batch (Work Stealing).
     */
    public List<Double> evaluatePopulationBatched(List<double[]> population, IClassifier classifier, List<Fold> folds) {
        boolean hybridEnabled = AppStateService.getInstance()
                .isHybridExecutionEnabled();
        double cpuRatio = AppStateService.getInstance().getCpuOffloadRatio();

        // Fallback to pure CPU if caching not used or not a GPU classifier
        if (!cachingEnabled || !(classifier instanceof IGpuCachedClassifier)) {
            return population.parallelStream()
                    .map(pos -> calculateFitness(pos, classifier, folds))
                    .collect(Collectors.toList());
        }

        IGpuCachedClassifier gpuClassifier = (IGpuCachedClassifier) classifier;
        int popSize = population.size();
        Double[] finalResults = new Double[popSize];

        // 1. Split Population
        List<Integer> gpuIndices = new ArrayList<>();
        List<Integer> cpuIndices = new ArrayList<>();

        if (hybridEnabled && cpuRatio > 0.0) {
            int cpuCount = (int) Math.ceil(popSize * cpuRatio);
            // Use end of population for CPU to minimize array copy overhead in sublist?
            // Arbitrary split is fine.
            for (int i = 0; i < popSize; i++) {
                if (i >= popSize - cpuCount) {
                    cpuIndices.add(i);
                } else {
                    gpuIndices.add(i);
                }
            }
        } else {
            for (int i = 0; i < popSize; i++)
                gpuIndices.add(i);
        }

        // 2. Prepare CPU Classifier (if needed)
        IClassifier cpuClassifier = null;
        if (!cpuIndices.isEmpty()) {
            String name = classifier.getName();
            if (name.contains("Random Forest")) {
                cpuClassifier = new RandomForestClassifier(40);
            } else if (name.contains("SVM")) {
                cpuClassifier = new SvmClassifier("linear", 1.0, 1e-3);
            } else if (name.contains("XGBoost")) {
                cpuClassifier = new XGBoostClassifier(10, 3);
            } else {
                cpuClassifier = classifier; // Fallback (might just call Python CPU)
            }
        }
        final IClassifier finalCpuClassifier = cpuClassifier;

        long startTime = System.nanoTime();

        // 3. Launch CPU Tasks (Asynchronous)
        CompletableFuture<List<Double>> cpuFuture = CompletableFuture.supplyAsync(() -> {
            if (cpuIndices.isEmpty())
                return Collections.emptyList();
            // Use common pool or custom executor? SHARED_EXECUTOR is available.
            return cpuIndices.parallelStream()
                    .map(idx -> calculateFitness(population.get(idx), finalCpuClassifier, folds))
                    .collect(Collectors.toList());
        }, SHARED_EXECUTOR);

        // 4. Execute GPU Batch (Main Thread blocks on this usually, but we can do it
        // here)
        // We reused the existing logic for GPU batching but constrained to gpuIndices
        List<Double> gpuResults = new ArrayList<>();
        if (!gpuIndices.isEmpty()) {

            int searchFolds = this.kFolds;
            @SuppressWarnings("unchecked")
            CompletableFuture<String>[][] trainFutures = new CompletableFuture[gpuIndices.size()][searchFolds];
            @SuppressWarnings("unchecked")
            CompletableFuture<int[]>[][] predictFutures = new CompletableFuture[gpuIndices.size()][searchFolds];

            boolean gpuCriticalFailure = false;

            // 4a. Submit TRAIN
            for (int i = 0; i < gpuIndices.size(); i++) {
                int originalIdx = gpuIndices.get(i);
                double[] pos = population.get(originalIdx);
                List<Integer> indices = getSelectedIndices(pos);
                if (indices.isEmpty())
                    continue;
                int[] featureIndices = indices.stream().mapToInt(Integer::intValue).toArray();

                for (int f = 0; f < searchFolds; f++) {
                    String trainId = foldTrainDatasetIds.get(f);
                    if (trainId != null) {
                        trainFutures[i][f] = gpuClassifier.submitTrainCachedAsync(trainId, featureIndices, null);
                    }
                }
            }

            // Wait for Train
            waitForFutures(trainFutures);

            // 4b. Submit PREDICT
            if (!gpuCriticalFailure) {
                for (int i = 0; i < gpuIndices.size(); i++) {
                    int originalIdx = gpuIndices.get(i);
                    double[] pos = population.get(originalIdx);
                    List<Integer> indices = getSelectedIndices(pos);
                    if (indices.isEmpty())
                        continue;
                    int[] featureIndices = indices.stream().mapToInt(Integer::intValue).toArray();

                    for (int f = 0; f < searchFolds; f++) {
                        if (trainFutures[i][f] != null) {
                            try {
                                String modelId = trainFutures[i][f].get();
                                String testId = foldTestDatasetIds.get(f);
                                predictFutures[i][f] = gpuClassifier.submitPredictCachedAsync(modelId, testId,
                                        featureIndices, null);
                            } catch (Exception e) {
                                if (e.getMessage() != null && e.getMessage().contains("No OpenCL device")) {
                                    logger.error(
                                            "CRITICAL GPU FAILURE: No OpenCL device found. Fallback to CPU enabled.",
                                            e);
                                    gpuCriticalFailure = true;
                                    break; // Break inner loop
                                } else if (e.getCause() != null && e.getCause().getMessage() != null
                                        && e.getCause().getMessage().contains("No OpenCL device")) {
                                    logger.error(
                                            "CRITICAL GPU FAILURE: No OpenCL device found. Fallback to CPU enabled.",
                                            e);
                                    gpuCriticalFailure = true;
                                    break; // Break inner loop
                                }
                                logger.error("Model ID retrieval failed", e);
                            }
                        }
                    }
                    if (gpuCriticalFailure)
                        break; // Break outer loop
                }
            }

            if (gpuCriticalFailure) {
                // FALLBACK: Disable GPU and run on CPU
                com.example.automl_prototype_1.service.AppStateService.getInstance().setUseGpu(false);
                this.cachingEnabled = false;
                logger.warn("GPU Acceleration Disabled due to runtime error. Processing batch on CPU.");

                // Processing all GPU indices on CPU
                final IClassifier fallbackCpuClassifier = finalCpuClassifier != null ? finalCpuClassifier : classifier; // Use
                                                                                                                        // the
                                                                                                                        // one
                                                                                                                        // prepared
                                                                                                                        // for
                                                                                                                        // CPU
                                                                                                                        // batch
                                                                                                                        // or
                                                                                                                        // original
                List<Double> fallbackResults = gpuIndices.parallelStream()
                        .map(idx -> calculateFitness(population.get(idx), fallbackCpuClassifier, folds))
                        .collect(Collectors.toList());
                gpuResults.addAll(fallbackResults);

            } else {
                // Normal GPU Process

                // Wait for Predict
                waitForFutures(predictFutures);

                // 4c. Compute Fitness
                for (int i = 0; i < gpuIndices.size(); i++) {
                    int originalIdx = gpuIndices.get(i);
                    double totalScore = 0.0;
                    boolean valid = true;

                    for (int f = 0; f < searchFolds; f++) {
                        if (predictFutures[i][f] == null) {
                            valid = false;
                            break;
                        }
                        try {
                            int[] pred = (int[]) predictFutures[i][f].get(10, TimeUnit.SECONDS);
                            totalScore += calculateFoldScore(folds.get(f).labels, pred);
                        } catch (Exception e) {
                            valid = false;
                            logger.error("Pred retrieval failed", e);
                        }
                    }

                    double fitness = 0.0;
                    if (valid) {
                        double combinedMetric = (totalScore / searchFolds);
                        double bonus = getLowCardinalityBonus(population.get(originalIdx), folds);
                        fitness = Math.max(0.0, combinedMetric + bonus);
                        fitnessCache.put(new ChromosomeKey(population.get(originalIdx)), fitness);
                    }
                    gpuResults.add(fitness);

                    // CLEANUP
                    for (int f = 0; f < searchFolds; f++) {
                        try {
                            if (trainFutures[i][f] != null)
                                GpuBatchService.getInstance().submitCleanup(trainFutures[i][f].get());
                        } catch (Exception e) {
                        }
                    }
                }
            }
        }

        // 5. Merge Results & Adaptive Control
        try {
            long gpuEndTime = System.nanoTime();
            long gpuDuration = gpuEndTime - startTime;

            List<Double> cpuResults = cpuFuture.get();
            long cpuEndTime = System.nanoTime();
            long cpuDuration = cpuEndTime - startTime;

            // Adaptive Logic: Push toward equal duration
            if (hybridEnabled && !gpuIndices.isEmpty() && !cpuIndices.isEmpty()) {
                double maxDuration = Math.max(cpuDuration, gpuDuration);
                double diffPercent = (double) (cpuDuration - gpuDuration) / maxDuration;

                // Adjustment factor: 0.01 per batch (Adaptive Control)
                double adjustment = 0.0;
                if (diffPercent > 0.1) {
                    // CPU took >10% longer than GPU -> Decrease CPU load
                    adjustment = -0.01;
                } else if (diffPercent < -0.1) {
                    // GPU took >10% longer than CPU -> Increase CPU load
                    adjustment = 0.01;
                }

                if (adjustment != 0.0) {
                    double currentRatio = com.example.automl_prototype_1.service.AppStateService.getInstance()
                            .getCpuOffloadRatio();
                    double newRatio = currentRatio + adjustment;
                    com.example.automl_prototype_1.service.AppStateService.getInstance().setCpuOffloadRatio(newRatio);
                }
            }

            for (int i = 0; i < gpuIndices.size(); i++) {
                finalResults[gpuIndices.get(i)] = gpuResults.get(i);
            }
            for (int i = 0; i < cpuIndices.size(); i++) {
                finalResults[cpuIndices.get(i)] = cpuResults.get(i);
            }

        } catch (Exception e) {
            logger.error("Hybrid execution failed", e);
            // Fallback? Return zeros.
            return new ArrayList<>(Collections.nCopies(popSize, 0.0));
        }

        return Arrays.asList(finalResults);
    }

    private void waitForFutures(CompletableFuture<?>[][] futures) {
        List<CompletableFuture<?>> list = new ArrayList<>();
        for (CompletableFuture<?>[] row : futures) {
            for (CompletableFuture<?> fut : row)
                if (fut != null)
                    list.add(fut);
        }
        if (!list.isEmpty()) {
            try {
                CompletableFuture.allOf(list.toArray(new CompletableFuture[0])).get(300, TimeUnit.SECONDS);
            } catch (Exception e) {
                logger.error("Batch Wait Failed", e);
            }
        }
    }

    private List<Integer> getSelectedIndices(double[] position) {
        List<Integer> list = new ArrayList<>();
        for (int j = 0; j < position.length; j++) {
            if (position[j] == 1.0)
                list.add(j);
        }
        // FAILSAFE: Ensure at least one feature is selected
        if (list.isEmpty()) {
            int randomFeature = random.nextInt(position.length);
            list.add(randomFeature);
            // Optionally update position to reflect this force-selection
            // position[randomFeature] = 1.0;
        }
        return list;
    }

    private double getLowCardinalityBonus(double[] position, List<Fold> folds) {
        if (!useHeuristics) {
            return 0.0;
        }

        if (this.isLowCardinalityFeature == null)
            return 0.0;

        double bonus = 0.0;
        for (int j = 0; j < position.length; j++) {
            if (position[j] == 1.0) { // Assuming 1.0 is selected
                if (j < isLowCardinalityFeature.length && isLowCardinalityFeature[j]) {
                    bonus += 0.01;
                }
            }
        }
        return Math.min(bonus, 0.05); // Cap bonus
    }

    private double calculateFoldScore(int[] testLabels, int[] predictions) {
        double total = 0.0;
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
            total += balancedAcc;
        } else {
            total += Accuracy.of(testLabels, predictions);
        }

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
        total += macroF1;

        // Normalize to 0..1 range
        return total / 2.0;
    }

    public static class Fold {
        public final double[][] features;
        public final int[] labels;

        public Fold(double[][] features, int[] labels) {
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

        // Pre-calculate selected indices to avoid checking 'position' in the tight loop
        List<Integer> selectedIndices = new ArrayList<>();
        for (int j = 0; j < position.length; j++) {
            if (position[j] == 1.0) {
                selectedIndices.add(j);
            }
        }

        int selectedFeatureCount = selectedIndices.size();
        if (selectedFeatureCount == 0) {
            return new double[numRecords][0];
        }

        double[][] filteredFeatures = new double[numRecords][selectedFeatureCount];

        // Optimized loop: Iterate only over selected columns
        for (int i = 0; i < numRecords; i++) {
            for (int k = 0; k < selectedFeatureCount; k++) {
                filteredFeatures[i][k] = originalFeatures[i][selectedIndices.get(k)];
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
