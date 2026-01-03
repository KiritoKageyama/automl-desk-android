package com.example.automl_prototype_1.service;

import com.example.automl_prototype_1.algorithm.*;
import com.example.automl_prototype_1.classifier.*;
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

        // Stage 2: Apply rules to generate recommendations
        List<AlgorithmRecommendation> recommendations = generateRecommendations(metaFeatures);

        // Stage 3: Sort recommendations by score in descending order
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
    private List<AlgorithmRecommendation> generateRecommendations(DatasetMetaFeatures features) {
        List<AlgorithmRecommendation> recommendations = new ArrayList<>();

        // --- Rule-based Scoring ---
        // We start with a base score for each pairing and then add or subtract points
        // based on rules.

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
        StringBuilder advice = new StringBuilder("--- Recommended Settings ---\\n\\n");

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
        List<Map<String, String>> records = dataset.getRecords();

        for (String header : headers) {
            Set<String> uniqueValues = new HashSet<>();
            List<Double> numericValues = new ArrayList<>();
            int missingCount = 0;

            for (Map<String, String> record : records) {
                String value = record.get(header);
                if (value == null || value.trim().isEmpty()) {
                    missingCount++;
                } else {
                    uniqueValues.add(value);
                    try {
                        numericValues.add(Double.parseDouble(value));
                    } catch (NumberFormatException e) {
                        // Not a numeric value, ignore
                    }
                }
            }

            boolean isNumeric = !numericValues.isEmpty();
            String dataType = isNumeric ? "Numeric" : "Categorical";
            String mean = "N/A";
            String stdDev = "N/A";
            String min = "N/A";
            String max = "N/A";

            if (isNumeric) {
                DoubleSummaryStatistics stats = numericValues.stream()
                        .filter(Objects::nonNull)
                        .mapToDouble(Double::doubleValue)
                        .summaryStatistics();

                mean = String.format("%.2f", stats.getAverage());
                min = String.format("%.2f", stats.getMin());
                max = String.format("%.2f", stats.getMax());

                // Calculate standard deviation
                double avg = stats.getAverage();
                double variance = numericValues.stream()
                        .mapToDouble(v -> Math.pow(v - avg, 2))
                        .average()
                        .orElse(0.0);
                stdDev = String.format("%.2f", Math.sqrt(variance));
            }

            summaries.add(new AttributeSummary(
                    header,
                    dataType,
                    uniqueValues.size(),
                    missingCount,
                    mean,
                    stdDev,
                    min,
                    max));
        }

        return summaries;
    }
}