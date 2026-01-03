package com.example.automl_prototype_1.service.agentic;

import com.example.automl_prototype_1.model.Dataset;
import java.util.ArrayList;
import java.util.List;

/**
 * The "Brain" of the Agentic AI layer.
 * Analyzes datasets and suggests optimal machine learning pipelines.
 */
public class AgenticAdvisor {

        private final MetaFeatureExtractor metaFeatureExtractor;
        private final EmpiricalBenchmarker empiricalBenchmarker;
        private final SotaLookupService sotaLookupService;

        public AgenticAdvisor() {
                this.metaFeatureExtractor = new MetaFeatureExtractor();
                this.empiricalBenchmarker = new EmpiricalBenchmarker();
                this.sotaLookupService = new SotaLookupService();
        }

        /**
         * Analyzes the dataset and returns a list of recommended pipelines.
         * Now includes Empirical Benchmarking and SOTA Lookup.
         * 
         * @param dataset     The dataset to analyze.
         * @param datasetName The name of the dataset file (for SOTA lookup).
         * @return List of PipelineSuggestion objects sorted by confidence.
         */
        public List<PipelineSuggestion> suggestPipelines(Dataset dataset, String datasetName) {
                List<PipelineSuggestion> suggestions = new ArrayList<>();
                MetaFeatures meta = metaFeatureExtractor.extract(dataset);

                System.out.println("AgenticAdvisor: Analyzing dataset properties: " + meta);

                // 1. Run Internal Empirical Benchmark (Quick Scan)
                System.out.println("AgenticAdvisor: Running internal empirical benchmark...");
                java.util.Map<String, Double> benchmarks = empiricalBenchmarker.benchmark(dataset);

                // Find best base model from benchmarks
                String bestBaseModel = benchmarks.entrySet().stream()
                                .max((e1, e2) -> Double.compare(e1.getValue(), e2.getValue()))
                                .map(java.util.Map.Entry::getKey)
                                .orElse("Random Forest"); // Fallback
                double bestScore = benchmarks.getOrDefault(bestBaseModel, 0.0);

                // 2. Perform External SOTA Lookup
                System.out.println("AgenticAdvisor: Searching for SOTA results...");
                String sotaInfo = sotaLookupService.searchSota(datasetName, meta.getNumInstances(),
                                meta.getNumFeatures());

                // 3. Generate Suggestions using Expanded Rules

                // --- BASELINE: Empirical Winner ---
                suggestions.add(new PipelineSuggestion(
                                "GA", // Standard powerful optimizer
                                bestBaseModel,
                                "Empirical analysis identified " + bestBaseModel + " as the strongest baseline (Score: "
                                                + String.format("%.2f", bestScore)
                                                + "). Genetic Algorithm (GA) is a robust default for feature selection.",
                                0.85,
                                bestScore,
                                sotaInfo));

                // --- RULE SET ---

                // Rule: Very Small Data (< 200 rows) -> Local Search + Instance Based
                if (meta.getNumInstances() < 200) {
                        suggestions.add(new PipelineSuggestion(
                                        "NNP",
                                        "KNN",
                                        "Dataset is very small (" + meta.getNumInstances()
                                                        + " instances). Nearest Neighbor Pattern (NNP) search with KNN is effective for fine-grained local optimization on small samples.",
                                        0.92,
                                        benchmarks.getOrDefault("KNN", 0.0),
                                        sotaInfo));
                }

                // Rule: High Scalability (Large Data > 2000 rows) -> Swarm Intelligence + Fast
                // Tree
                if (meta.getNumInstances() > 2000) {
                        suggestions.add(new PipelineSuggestion(
                                        "WWO",
                                        "XGBoost",
                                        "Large dataset detected. Water Wave Optimization (WWO) offers efficient exploration with fewer evaluations, and XGBoost scales well.",
                                        0.90,
                                        benchmarks.getOrDefault("XGBoost", 0.0),
                                        sotaInfo));
                }

                // Rule: High Sparsity (> 50%) -> Binary Swarm
                if (meta.getSparsity() > 0.5) {
                        suggestions.add(new PipelineSuggestion(
                                        "BPSO",
                                        "XGBoost",
                                        "Dataset is sparse (" + String.format("%.1f", meta.getSparsity() * 100)
                                                        + "% zeros). Binary PSO (BPSO) combined with XGBoost is effective for sparse feature spaces.",
                                        0.94,
                                        benchmarks.getOrDefault("XGBoost", 0.0),
                                        sotaInfo));
                }

                // Rule: High Dimensionality (> 50 feat) -> Hybrid Evolutionary
                if (meta.getNumFeatures() > 50) {
                        suggestions.add(new PipelineSuggestion(
                                        "IGPSO",
                                        "Random Forest",
                                        "High dimensionality (" + meta.getNumFeatures()
                                                        + " features). Improved Genetic PSO (IGPSO) combines swarming with crossover to escape local optima in complex spaces.",
                                        0.91,
                                        benchmarks.getOrDefault("Random Forest", 0.0),
                                        sotaInfo));
                }

                // Rule: Complex Landscape (Medium Data, High Features) -> Atom Search
                if (meta.getNumInstances() > 500 && meta.getNumFeatures() > 20) {
                        suggestions.add(new PipelineSuggestion(
                                        "ASO",
                                        bestBaseModel,
                                        "Atom Search Optimization (ASO) models interaction forces, providing a strong balance of exploration and exploitation for medium-complexity datasets.",
                                        0.89,
                                        bestScore,
                                        sotaInfo));
                }

                // Rule: High Imbalance (Provisional check if we had imbalance metric, else
                // heuristics)
                // Assuming imbalance is available or we infer it from accuracy vs F1 potential
                // (omitted for now as extraction needs verification)

                // Sort by confidence
                suggestions.sort((s1, s2) -> Double.compare(s2.getConfidence(), s1.getConfidence()));

                // Limit to top 3 suggestions
                if (suggestions.size() > 3) {
                        return suggestions.subList(0, 3);
                }

                return suggestions;
        }

        // Overload for backward compatibility (if needed, though we should update
        // callers)
        public List<PipelineSuggestion> suggestPipelines(Dataset dataset) {
                return suggestPipelines(dataset, "unknown_dataset.csv");
        }
}
