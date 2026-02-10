package com.example.automl_prototype_1.service.agentic;

import com.example.automl_prototype_1.model.Dataset;
import java.util.ArrayList;
import java.util.List;

/**
 * The "Brain" of the Agentic AI layer.
 * Analyzes datasets and suggests optimal machine learning pipelines.
 */
import com.example.automl_prototype_1.service.SotaLookupService; // Moved from .agentic

/**
 * The "Brain" of the Agentic AI layer.
 * Analyzes datasets and suggests optimal machine learning pipelines.
 */
public class AgenticAdvisor implements AutoCloseable {

        private final MetaFeatureExtractor metaFeatureExtractor;
        private final EmpiricalBenchmarker empiricalBenchmarker;
        private final SotaLookupService sotaLookupService;
        private final LocalLlmService llmService;
        private final ConstitutionalService constitutionalService;
        private final IInferenceEngine inferenceEngine;

        public AgenticAdvisor() {
                this.metaFeatureExtractor = new MetaFeatureExtractor();
                this.empiricalBenchmarker = new EmpiricalBenchmarker();
                this.sotaLookupService = new SotaLookupService();
                this.llmService = new LocalLlmService();
                this.constitutionalService = new ConstitutionalService(); // Initialize Service

                // Initialize inference engine with ONNX preference, Python fallback
                IInferenceEngine engine = null;
                try {
                        OnnxInferenceEngine onnx = new OnnxInferenceEngine();
                        if (onnx.isAvailable()) {
                                engine = onnx;
                                com.example.automl_prototype_1.service.LogService.info("AgenticAdvisor",
                                                "Using ONNX_RUNTIME inference engine");
                        }
                } catch (Exception e) {
                        com.example.automl_prototype_1.service.LogService.warn("AgenticAdvisor",
                                        "ONNX engine failed to initialize: " + e.getMessage());
                }
                if (engine == null) {
                        engine = new PythonInferenceEngine();
                        com.example.automl_prototype_1.service.LogService.info("AgenticAdvisor",
                                        "Using PYTHON_SERVER inference engine (fallback)");
                }
                this.inferenceEngine = engine;
        }

        @Override
        public void close() {
                if (empiricalBenchmarker != null) {
                        empiricalBenchmarker.shutdown();
                }
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
                System.out.println("AgenticAdvisor: Running internal empirical benchmark ('Tiny Training')...");
                java.util.Map<String, Double> benchmarks = empiricalBenchmarker.benchmark(dataset);

                // 1. CAPTURE EMPIRICAL TRUTH
                String bestBaseModel = benchmarks.entrySet().stream()
                                .max((e1, e2) -> Double.compare(e1.getValue(), e2.getValue()))
                                .map(java.util.Map.Entry::getKey)
                                .orElse("Random Forest");
                double bestScore = benchmarks.getOrDefault(bestBaseModel, 0.0);

                System.out.println("AgenticAdvisor: Empirical Winner is " + bestBaseModel + " with score "
                                + String.format("%.2f", bestScore));

                // 2. EXTERNAL SOTA LOOKUP
                System.out.println("AgenticAdvisor: Searching for SOTA results...");
                String sotaInfo = sotaLookupService.searchSota(dataset); // Using new SotaLookup

                // 3. GAP ANALYSIS ("Ambition" Factor)
                // Parse SOTA or use 99% as target
                double targetScore = 0.99;
                if (sotaInfo.contains("%")) {
                        try {
                                // simple parser for "98.5%"
                                String num = sotaInfo.replaceAll("[^0-9.]", "");
                                if (!num.isEmpty())
                                        targetScore = Double.parseDouble(num) / 100.0;
                        } catch (Exception e) {
                        }
                }

                double gap = targetScore - bestScore;
                String strategyMode = (gap > 0.15) ? "EXPLORATION" : (gap < 0.02) ? "OPTIMIZATION" : "BALANCED";
                System.out.println("AgenticAdvisor: Gap Analysis -> Mode: " + strategyMode + " (Gap: "
                                + String.format("%.2f", gap) + ")");

                // 4. CANDIDATE GENERATION (Rule-Based + Empirical)
                String[] allAlgos = { "BPSO", "IGPSO", "GA", "WWO", "ASO", "PLS", "SA", "DE", "GWO" };

                for (String algo : allAlgos) {
                        // A. Base Confidence = Empirical Score
                        double score = bestScore;
                        String classifier = bestBaseModel;
                        String reason = "Empirical baseline.";

                        // B. Apply Strategy Modifiers
                        if (strategyMode.equals("EXPLORATION")) {
                                if (algo.equals("IGPSO") || algo.equals("WWO") || algo.equals("DE")) {
                                        score += 0.15; // Boost heavy explorers
                                        reason = "Deep exploration required (Wide Performance Gap). " + algo
                                                        + " excels at escaping local optima.";
                                }
                        } else if (strategyMode.equals("OPTIMIZATION")) {
                                if (algo.equals("PLS") || algo.equals("SA")) {
                                        score += 0.10; // Boost fast local searchers
                                        reason = "Problem nearly solved. " + algo
                                                        + " is efficient for final fine-tuning.";
                                }
                        }

                        // C. Apply Meta-Feature Rules (The "Wisdom")
                        // Feature Selection Importance
                        if (meta.getNumFeatures() > 50) {
                                if (algo.equals("BPSO") || algo.equals("GA") || algo.equals("IGPSO")) {
                                        score += 0.05;
                                        reason += " High dimensionality favors " + algo + " for feature selection.";
                                }
                        }

                        // Large Data Handling
                        if (meta.getNumInstances() > 5000) {
                                if (algo.equals("WWO")) {
                                        score += 0.05;
                                        reason += " WWO scales well with instances.";
                                }
                        }

                        // Sparsity Specialization
                        if (meta.getSparsity() > 0.5) {
                                if (algo.equals("BPSO")) {
                                        score += 0.10;
                                        reason += " BPSO is designed for sparse binary feature selection.";
                                }
                                // Sparse data often works better with Linear models or XGBoost
                                if (!bestBaseModel.contains("XGBoost")) {
                                        // Consider switching classifier for this suggestion if sparsity is extreme
                                        if (meta.getSparsity() > 0.7) {
                                                classifier = "XGBoost"; // Provisional override
                                                reason += " [Override] XGBoost preferred for high sparsity.";
                                        }
                                }
                        }

                        // 5. THE CONSTITUTION (System Guardrails)
                        // Enforce system limits DYNAMICALLY using ConstitutionalService
                        ConstitutionalService.Verdict verdict = constitutionalService.evaluate(meta, algo, classifier);

                        if (!verdict.allowed) {
                                // If vetoed, we either kill it or downgrade heavily
                                score = 0.0;
                                reason = "[VETOED by CONST.] " + verdict.reason;
                        } else {
                                if (verdict.warning != null) {
                                        score -= 0.15;
                                        reason += " [Warning] " + verdict.warning;
                                }
                        }

                        if (score > 0.0) {
                                suggestions.add(new PipelineSuggestion(
                                                algo,
                                                classifier,
                                                reason,
                                                Math.min(0.99, score),
                                                bestScore,
                                                sotaInfo));
                        }
                }

                suggestions.sort((s1, s2) -> Double.compare(s2.getConfidence(), s1.getConfidence()));

                // 6. AGENTIC VERIFICATION (LLM)
                if (!suggestions.isEmpty() && llmService.isAvailable()) {
                        int topK = Math.min(suggestions.size(), 3);
                        for (int i = 0; i < topK; i++) {
                                PipelineSuggestion p = suggestions.get(i);

                                // Inject the Constitution into the prompt
                                String promptContext = String.format(
                                                "SYSTEM PROTOCOL: You are an AutoML Logic Engine. " +
                                                                "CONSULT THE DYNAMIC CONSTITUTION for rules."
                                                                +
                                                                "TASK: Verify recommendation '%s + %s' for dataset (%d rows, %d cols). "
                                                                +
                                                                "If valid, Explain WHY. If invalid, explain WHY and suggest alternative.",
                                                p.getAlgorithmName(), p.getClassifierName(), meta.getNumInstances(),
                                                meta.getNumFeatures());

                                String aiFeedback = llmService.queryOllama(promptContext, "active");
                                if (aiFeedback != null && !aiFeedback.isEmpty()) {
                                        p.setReasoning(p.getReasoning() + "\n[AI Audit]: " + aiFeedback);
                                }
                        }
                }

                return suggestions;
        }

        // Overload for backward compatibility
        public List<PipelineSuggestion> suggestPipelines(Dataset dataset) {
                return suggestPipelines(dataset, "unknown_dataset.csv");
        }

        /**
         * Generates a comprehensive natural language report on the dataset strategy.
         */
        public String generateStrategicReport(Dataset dataset, String datasetName) {
                MetaFeatures meta = metaFeatureExtractor.extract(dataset);

                if (llmService.isAvailable()) {
                        String stats = String.format("Rows: %d, Feats: %d, Sparsity: %.2f",
                                        meta.getNumInstances(), meta.getNumFeatures(), meta.getSparsity());

                        String prompt = String.format(
                                        "Act as a Senior Data Scientist. For dataset '%s' (%s), provide a strategic summary covering:\n"
                                                        +
                                                        "1. Recommended Cross-Validation Strategy (k-fold vs holdout)\n"
                                                        +
                                                        "2. Data Scaling Requirements\n" +
                                                        "3. Hyperparameter Tuning importance\n" +
                                                        "Keep it professional and concise (max 150 words).",
                                        datasetName, stats);

                        String report = llmService.queryOllama(prompt, "active");
                        if (report != null && !report.isEmpty()) {
                                return "--- AI Strategic Analysis ---\n\n" + report;
                        }
                }

                // Fallback (or could move the logic from RecommendationService here)
                return "--- Automated Strategy ---\n\n" +
                                "Analysis based on Meta-Features:\n" +
                                "- Size: " + meta.getNumInstances() + " instances\n" +
                                "- Dimensions: " + meta.getNumFeatures() + " features\n" +
                                "Recommendation: Use robust cross-validation (5-fold) and standardization.";
        }

        /**
         * Returns the configured inference engine for model predictions.
         * 
         * @return The active IInferenceEngine (ONNX or Python)
         */
        public IInferenceEngine getInferenceEngine() {
                return inferenceEngine;
        }
}
