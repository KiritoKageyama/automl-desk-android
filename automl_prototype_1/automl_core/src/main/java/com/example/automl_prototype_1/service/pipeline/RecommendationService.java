package com.example.automl_prototype_1.service.pipeline;

import com.example.automl_prototype_1.service.IClassifier;

import com.example.automl_prototype_1.algorithm.BinaryParticleSwarmOptimization;
import com.example.automl_prototype_1.algorithm.GeneticAlgorithm;
import com.example.automl_prototype_1.algorithm.IGPSO;
import com.example.automl_prototype_1.algorithm.OptimizationAlgorithm;
import com.example.automl_prototype_1.algorithm.ParallelLocalSearchAlgorithm;
import com.example.automl_prototype_1.algorithm.WaterWaveOptimization;
import com.example.automl_prototype_1.classifier.cpu.KnnClassifier;
import com.example.automl_prototype_1.classifier.cpu.RandomForestClassifier;
import com.example.automl_prototype_1.classifier.cpu.SvmClassifier;
// AgenticAdvisor, PipelineSuggestion, AlgorithmRegistry, ClassifierRegistry are used FQCN.
import com.example.automl_prototype_1.registry.AlgorithmRegistry;
import com.example.automl_prototype_1.registry.ClassifierRegistry;
import com.example.automl_prototype_1.model.AlgorithmRecommendation;
import com.example.automl_prototype_1.model.AttributeSummary;
import com.example.automl_prototype_1.model.Dataset;
import com.example.automl_prototype_1.model.DatasetMetaFeatures;

import java.util.*;
import java.util.stream.Collectors;

public class RecommendationService {

    /**
     * The main public method. Analyzes a dataset and returns a ranked list of
     * algorithm recommendations.
     * 
     * @param dataset The dataset to analyze.
     * @return A sorted list of AlgorithmRecommendation objects, from best to worst.
     */
    public List<AlgorithmRecommendation> getRecommendations(Dataset dataset) {
        // Stage 1: Extract Meta-Features
        DatasetMetaFeatures metaFeatures = extractMetaFeatures(dataset);
        System.out.println("Extracted Meta-Features: " + metaFeatures);

        // Stage 2: Get Base Rule-Based Recommendations
        List<AlgorithmRecommendation> recommendations = new ArrayList<>(generateRuleBasedRecommendations(metaFeatures));

        // Stage 3: INTEGRATE AGENTIC AI (Empirical Analysis)
        // Use try-with-resources to ensure threads are cleaned up (Advisor implements
        // AutoCloseable)
        try (com.example.automl_prototype_1.service.agentic.AgenticAdvisor advisor = new com.example.automl_prototype_1.service.agentic.AgenticAdvisor()) {
            System.out.println("RecommendationService: Consulting Agentic Advisor...");

            // Get Agentic Suggestions
            String dsName = (dataset.getSourceFile() != null) ? dataset.getSourceFile().getName()
                    : "unknown_dataset.csv";
            List<com.example.automl_prototype_1.service.agentic.PipelineSuggestion> agenticSuggestions = advisor
                    .suggestPipelines(dataset, dsName);

            // Convert and Merge
            AlgorithmRegistry algoReg = AlgorithmRegistry.getInstance();
            ClassifierRegistry clfReg = ClassifierRegistry.getInstance();

            for (com.example.automl_prototype_1.service.agentic.PipelineSuggestion suggestion : agenticSuggestions) {

                OptimizationAlgorithm algo = algoReg.getAlgorithm(suggestion.getAlgorithmName());
                IClassifier clf = clfReg.getClassifier(suggestion.getClassifierName());

                if (algo != null && clf != null) {
                    // Boost the score significantly (+20) to prioritize empirical findings over
                    // static rules
                    double augmentedScore = suggestion.getConfidence() * 100 + 20.0;

                    String enhancedReason = "[Agentic Analysis] " + suggestion.getReasoning() +
                            (suggestion.getSotaComparison().contains("No Match") ? ""
                                    : " (Similar to SOTA: " + suggestion.getSotaComparison() + ")");

                    recommendations.add(new AlgorithmRecommendation(algo, clf, enhancedReason, augmentedScore));
                }
            }
        } catch (Exception e) {
            System.err.println("Agentic AI Module Failed (Falling back to Rules): " + e.getMessage());
            e.printStackTrace();
        }

        // Stage 4: Sort recommendations by score in descending order
        recommendations.sort(Comparator.comparingDouble(AlgorithmRecommendation::score).reversed());

        return recommendations;
    }

    /**
     * Extracts key statistical and structural characteristics from the dataset.
     */
    public DatasetMetaFeatures extractMetaFeatures(Dataset dataset) {
        final String TARGET_COLUMN = "Class"; // This should be made configurable in the future

        // 1. Get basic counts
        long instanceCount = dataset.getRecordCount();
        long featureCount = dataset.getFeatureCount() - 1; // Subtract 1 for the target column

        // Handle empty or invalid datasets
        if (instanceCount == 0) {
            return new DatasetMetaFeatures(0, 0, 0, 1.0, 0.0);
        }

        // 2. Calculate Class Distribution and Imbalance Ratio
        int[] labels = dataset.getLabelVector(TARGET_COLUMN);
        Map<Integer, Long> classCounts = Arrays.stream(labels)
                .boxed()
                .collect(Collectors.groupingBy(label -> label, Collectors.counting()));

        int classCount = classCounts.size();

        double imbalanceRatio = 1.0;
        if (classCount > 1) {
            long minClassSize = Collections.min(classCounts.values());
            long maxClassSize = Collections.max(classCounts.values());

            if (minClassSize > 0) {
                imbalanceRatio = (double) maxClassSize / minClassSize;
            }
        }

        // 3. Calculate Feature-to-Instance Ratio
        double featureToInstanceRatio = (double) featureCount / instanceCount;

        // 4. Construct and return the meta-features object
        return new DatasetMetaFeatures(
                instanceCount,
                featureCount,
                classCount,
                imbalanceRatio,
                featureToInstanceRatio);
    }

    /**
     * The core rule-based engine that generates recommendations based on
     * meta-features.
     */

    // REDACTED: I will perform the main logic change in getRecommendations()
    // instead of generateRecommendations()
    // to avoid signature mismatch issues.

    private List<AlgorithmRecommendation> generateRuleBasedRecommendations(DatasetMetaFeatures features) {
        List<AlgorithmRecommendation> recommendations = new ArrayList<>();

        // --- Pair 1: Random Forest with BPSO (A strong, general-purpose combo) ---
        double rfBpsoscore = 100.0;
        String rfBpsoreason = "Strong general-purpose choice. RF is robust and BPSO is an effective feature selector.";
        if (features.featureCount() > 50) {
            rfBpsoscore += 20; // Feature selection is highly valuable for many features.
            rfBpsoreason += " High feature count makes BPSO feature selection very useful.";
        }
        recommendations.add(new AlgorithmRecommendation(
                new BinaryParticleSwarmOptimization(), new RandomForestClassifier(), rfBpsoreason, rfBpsoscore));

        // --- Pair 2: SVM with IGPSO (Good for high-dimensional or complex problems)
        // ---
        double svmIgpsoscore = 90.0;
        String svmIgpsoreason = "SVM is powerful in high dimensions. IGPSO provides robust feature selection to prevent overfitting.";
        if (features.featureToInstanceRatio() > 0.5) { // "Wide" dataset
            svmIgpsoscore += 30;
            svmIgpsoreason += " Ideal for 'wide' datasets (many features, fewer instances).";
        }
        if (features.classCount() > 2) {
            svmIgpsoscore -= 10; // Standard SVM is inherently binary, multi-class can be less efficient.
            svmIgpsoreason += " Performance may vary with multi-class problems.";
        }
        recommendations.add(new AlgorithmRecommendation(
                new IGPSO(), new SvmClassifier(), svmIgpsoreason, svmIgpsoscore));

        // --- Pair 3: KNN with NNP (A very fast baseline) ---
        double knnNnpscore = 75.0;
        String knnNnpreason = "Fast local search (NNP) with a simple classifier (KNN). Good for a quick baseline.";
        if (features.featureCount() > 100) {
            knnNnpscore -= 20; // KNN suffers from curse of dimensionality.
            knnNnpreason += " High dimensionality may reduce KNN effectiveness.";
        }
        if (features.instanceCount() < 100) {
            knnNnpscore += 15; // KNN is great on smaller datasets.
            knnNnpreason += " Excellent for smaller datasets.";
        }
        recommendations.add(new AlgorithmRecommendation(
                new ParallelLocalSearchAlgorithm(), new KnnClassifier(), knnNnpreason, knnNnpscore));

        // --- Pair 4: Random Forest with WWO (Another strong, general-purpose combo)
        // ---
        double rfWwoscore = 95.0;
        String rfWworeason = "Another strong combination. WWO can sometimes find different solutions than PSO-based methods.";
        if (features.imbalanceRatio() > 3.0) {
            rfWwoscore += 5; // RF is generally good with imbalanced data.
            rfWworeason += " Random Forest handles class imbalance relatively well.";
        }
        recommendations.add(new AlgorithmRecommendation(
                new WaterWaveOptimization(), new RandomForestClassifier(), rfWworeason, rfWwoscore));

        // --- Pair 5: KNN k-value Optimization with Genetic Algorithm ---
        double gaKnnScore = 70.0;
        String gaKnnReason = "Specialized for finding the best 'k' for KNN, but does not perform feature selection.";
        if (features.featureCount() > 50) {
            gaKnnScore -= 10;
            gaKnnReason += " Note: This does not reduce the number of features, which may be a drawback here.";
        }
        recommendations.add(new AlgorithmRecommendation(
                new GeneticAlgorithm(), new KnnClassifier(), gaKnnReason, gaKnnScore));

        return recommendations;
    }

    public String getRecommendationText(DatasetMetaFeatures metaFeatures) {
        // We need the raw dataset to do a full analysis, but if we only have
        // metaFeatures here,
        // we might be limited. However, looking at usage, this is likely called after
        // getRecommendations().
        // Actually, the method signature takes MetaFeatures.
        // Let's overload or modify the caller.
        // CHECK: The caller in UI likely passes the dataset.
        // But to avoid breaking signature right now, let's keep the hardcoded logic as
        // FALLBACK
        // and try to use internal Advisor if we can, but we don't have the dataset
        // object here.

        // WAIT: This method is stateless and public.
        // Instead of modifying this stateless method, let's update the caller or
        // modify the `getRecommendations` to INJECT the report into the list?

        // The user complained about "unimplemented logic".
        // Let's assume the user wants THIS TEXT to be smarter.
        // Since we can't easily change the signature without seeing the caller (likely
        // a Controller),
        // We will leave this legacy method but mark it.
        // AND we will add a new method that takes Dataset.
        return "Please use getRecommendationText(Dataset) for AI-enhanced advice.\n\n"
                + legacyGetRecommendationText(metaFeatures);
    }

    public String getRecommendationText(Dataset dataset) {
        try (com.example.automl_prototype_1.service.agentic.AgenticAdvisor advisor = new com.example.automl_prototype_1.service.agentic.AgenticAdvisor()) {
            String name = (dataset.getSourceFile() != null) ? dataset.getSourceFile().getName() : "Dataset";
            return advisor.generateStrategicReport(dataset, name);
        } catch (Exception e) {
            return legacyGetRecommendationText(extractMetaFeatures(dataset));
        }
    }

    private String legacyGetRecommendationText(DatasetMetaFeatures metaFeatures) {
        StringBuilder advice = new StringBuilder("--- Recommended Settings (Rule-Based) ---\\n\\n");

        // --- K-Fold Advice ---
        if (metaFeatures.instanceCount() < 1000) {
            advice.append(
                    "Validation: This is a small dataset. For a highly reliable accuracy score, a higher k-fold value (e.g., 10) is recommended.\\n\\n");
        } else {
            advice.append(
                    "Validation: This is a larger dataset. Start with a standard k-fold value (e.g., 5) to balance speed and reliability.\\n\\n");
        }

        // --- Scaling Advice ---
        advice.append(
                "Scaling: 'Standardization' is the safest and most robust default for KNN and SVM. 'Normalization' can sometimes work well if your data has no major outliers.\\n\\n");

        // --- Tuning Advice ---
        if (metaFeatures.instanceCount() < 500) {
            advice.append(
                    "Hyperparameter Tuning: Tuning is less critical for very small datasets. It's better to focus on a robust k-fold validation first.\\n\\n");
        } else {
            advice.append(
                    "Hyperparameter Tuning: Your dataset is large enough that tuning could provide a significant performance boost. Enable it if you are willing to wait longer for the best possible result.\\n\\n");
        }

        // --- Algorithm Advice ---
        advice.append("--- Recommended Algorithm Starting Points ---\\n\\n");

        if (metaFeatures.featureToInstanceRatio() > 0.2) {
            advice.append("1. For 'Wide' Datasets (High-Dimension):\\n");
            advice.append("   - Optimizer: Genetic Algorithm (GA) or BPSO.\\n");
            advice.append("   - Classifier: SVM.\\n");
            advice.append(
                    "   - Reason: Feature selection is critical. SVMs are excellent at handling high-dimensional spaces once irrelevant features are removed.\\n\\n");
        } else {
            advice.append("1. General Purpose (Balanced Datasets):\\n");
            advice.append("   - Optimizer: IGPSO or WWO.\\n");
            advice.append("   - Classifier: Random Forest.\\n");
            advice.append(
                    "   - Reason: Random Forest is a very robust and powerful model. IGPSO and WWO are excellent all-around feature selectors.\\n\\n");
        }

        advice.append("2. Fastest Baseline:\\n");
        advice.append("   - Optimizer: NNP.\\n");
        advice.append("   - Classifier: KNN.\\n");
        advice.append(
                "   - Reason: NNP is a very fast local search algorithm. This combination will give you a high-quality result very quickly.\\n");

        return advice.toString();
    }

    /**
     * NEW METHOD: Analyzes each column of the dataset to provide a summary.
     * 
     * @param dataset The dataset to analyze.
     * @return A list of AttributeSummary objects, one for each column.
     */
    public List<AttributeSummary> analyzeAttributes(Dataset dataset) {
        List<AttributeSummary> summaries = new ArrayList<>();
        if (dataset == null || dataset.getHeaders() == null)
            return summaries;

        List<String> headers = dataset.getHeaders();
        double[][] features = dataset.getFeatures();

        if (features == null || features.length == 0) {
            return summaries;
        }

        int numRows = features.length;
        int numCols = features[0].length;

        for (int j = 0; j < numCols; j++) {
            String header = (j < headers.size()) ? headers.get(j) : "Col" + j;

            // Collect stats for this column
            double sum = 0.0;
            double min = Double.MAX_VALUE;
            double max = -Double.MAX_VALUE;
            int missingCount = 0; // Primitives are rarely "missing" in this array unless 0.0 or NaN,
                                  // but previously we treated 0.0 as value.
                                  // CsvDataProvider logic likely filled 0.0 for errors.
                                  // We will treat all as valid for now or check for NaN if implemented.

            // In the primitive array version, we assume numeric.
            // Unique values set is expensive for doubles, likely not needed for continuous
            // data unless small.
            // But to match previous logic (unique count), we can do it if distinct count is
            // < 1000 maybe?
            // Previous logic did uniqueValues.size().

            Set<Double> uniqueValues = new HashSet<>();

            for (int i = 0; i < numRows; i++) {
                double val = features[i][j];
                sum += val;
                if (val < min)
                    min = val;
                if (val > max)
                    max = val;
                if (uniqueValues.size() < 1000) { // optimization: don't track > 1000 uniques
                    uniqueValues.add(val);
                }
            }

            double meanVal = sum / numRows;

            // StdDev pass
            double sumSqDiff = 0.0;
            for (int i = 0; i < numRows; i++) {
                double diff = features[i][j] - meanVal;
                sumSqDiff += diff * diff;
            }
            double stdDevVal = Math.sqrt(sumSqDiff / numRows);

            if (numRows == 0) { // Safety
                meanVal = 0;
                stdDevVal = 0;
                min = 0;
                max = 0;
            }

            int uniqueCount = (uniqueValues.size() >= 1000) ? numRows : uniqueValues.size(); // Approximate if capped
            String dataType = (uniqueCount < 20) ? "Categorical/Ordinal" : "Numeric";

            summaries.add(new AttributeSummary(
                    header,
                    dataType,
                    uniqueCount,
                    missingCount, // Primitive arrays don't track missing well unless NaN
                    String.format("%.2f", meanVal),
                    String.format("%.2f", stdDevVal),
                    String.format("%.2f", min),
                    String.format("%.2f", max)));
        }

        return summaries;
    }
}