package com.example.automl_prototype_1.service.agentic;

import com.example.automl_prototype_1.model.Dataset;
import com.example.automl_prototype_1.service.agentic.inference.IInferenceEngine;
import com.example.automl_prototype_1.service.agentic.inference.OnnxInferenceEngine;
import com.example.automl_prototype_1.service.agentic.inference.PythonInferenceEngine;
import java.util.ArrayList;
import java.util.List;
import com.example.automl_prototype_1.model.Domain;

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
        private final com.example.automl_prototype_1.service.OllamaService llmService;
        private final com.example.automl_prototype_1.service.AnthropicService anthropicService;
        private final ConstitutionalService constitutionalService;
        private final IInferenceEngine inferenceEngine;

        public AgenticAdvisor() {
                this.metaFeatureExtractor = new MetaFeatureExtractor();
                this.empiricalBenchmarker = new EmpiricalBenchmarker();
                this.sotaLookupService = new SotaLookupService();
                this.llmService = com.example.automl_prototype_1.service.OllamaService.getInstance();
                this.constitutionalService = ConstitutionalService.getInstance(); // Initialize Service
                this.anthropicService = com.example.automl_prototype_1.service.AnthropicService.getInstance();

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

                // 6. AGENTIC VERIFICATION (Multi-Tier LLM)
                if (!suggestions.isEmpty()) {
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

                                String aiFeedback = null;
                                
                                // Priority 1: Claude 3.7 (System 2 Thinking)
                                if (anthropicService.isAvailable()) {
                                        com.example.automl_prototype_1.service.LogService.info("AgenticAdvisor", "Invoking Claude 3.7 for High-Precision Audit...");
                                        aiFeedback = anthropicService.queryClaude(promptContext);
                                }
                                
                                // Priority 2: Ollama (System 1/2 Thinking)
                                if (aiFeedback == null && llmService.isAvailable()) {
                                        com.example.automl_prototype_1.service.LogService.info("AgenticAdvisor", "Falling back to Ollama for AI Audit...");
                                        aiFeedback = llmService.queryOllama(promptContext, "active");
                                }

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
        public String generateStrategicReport(Dataset dataset, String datasetName, Domain currentDomain) {
                MetaFeatures meta = metaFeatureExtractor.extract(dataset);
                String domainStr = (currentDomain != null) ? currentDomain.name() : "General ML";

                if (llmService.isAvailable()) {
                        String stats = String.format("Rows: %d, Feats: %d, Sparsity: %.2f",
                                        meta.getNumInstances(), meta.getNumFeatures(), meta.getSparsity());

                        String prompt = String.format(
                                        "Act as a Senior AI Architect specializing in %s. For dataset '%s' (%s), provide a strategic summary covering:\n"
                                                        +
                                                        "1. Recommended Architectures / Algorithms for %s\n"
                                                        +
                                                        "2. Key Preprocessing or Data Prep Requirements\n" +
                                                        "3. Important Hyperparameters to tune\n" +
                                                        "Keep it professional and concise (max 150 words).",
                                        domainStr, datasetName, stats, domainStr);

                        String report = null;
                        
                        // Priority 1: Claude
                        if (anthropicService.isAvailable()) {
                             report = anthropicService.queryClaude(prompt);
                        }
                        
                        // Priority 2: Ollama
                        if (report == null && llmService.isAvailable()) {
                             report = llmService.queryOllama(prompt, "active");
                        }

                        if (report != null && !report.isEmpty()) {
                                return "--- Agentic Strategic Report ---\n\n" + report;
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

        /**
         * Analyzes the dataset and suggests a preprocessing strategy (Phase 5).
         * 
         * @param dataset The dataset to analyze.
         * @return A string summary of recommendations (e.g. "Transform: Log, Outliers:
         *         True").
         */
        public String suggestPreprocessingStrategy(Dataset dataset) {
                MetaFeatures meta = metaFeatureExtractor.extract(dataset);

                // 1. Simple Heuristics (Fallback)
                // Skewness > 1.0 -> Log
                // Sparsity > 0.5 -> Interaction? Or PCA (not impl).
                // Outliers? No cheap check for outliers here without full pass, but meta might
                // have kurtosis?
                // MetaFeatures currently has: inst, feats, classes, missing, sparsity.

                // 2. LLM Advice
                if (llmService.isAvailable()) {
                        String stats = String.format("Rows: %d, Feats: %d, Sparsity: %.2f",
                                        meta.getNumInstances(), meta.getNumFeatures(), meta.getSparsity());

                        String prompt = String.format(
                                        "Act as an expert Data Scientist. For a dataset with these stats: [%s], recommend a Preprocessing Strategy.\n"
                                                        +
                                                        "Options:\n" +
                                                        "- Transformation: [None, Log, Sqrt, Auto-Skew]\n" +
                                                        "- Feature Creation: [None, Interaction, Polynomial]\n" +
                                                        "- Outlier Removal: [Yes/No] (IQR Method)\n" +
                                                        "\n" +
                                                        "Output ONLY the decision in this format: 'Transform: X, Creation: Y, Outliers: Z'.\n"
                                                        +
                                                        "Reasoning: Brief one-line explanation.",
                                        stats);

                        String response = llmService.queryOllama(prompt, "active");
                        if (response != null && !response.isBlank()) {
                                return response;
                        }
                }

                // Fallback Logic
                StringBuilder recommendation = new StringBuilder();
                recommendation.append("Transform: ");
                if (meta.getSparsity() > 0.8) {
                        recommendation.append("None (High Sparsity)");
                } else {
                        recommendation.append("Auto-Skew (Safe Bet)");
                }

                recommendation.append(", Creation: ");
                if (meta.getNumFeatures() < 10 && meta.getNumInstances() > 500) {
                        recommendation.append("Interaction (Low Dim, Sufficient Data)");
                } else {
                        recommendation.append("None");
                }

                recommendation.append(", Outliers: ");
                if (meta.getNumInstances() > 1000) {
                        recommendation.append("True (Robustness)");
                } else {
                        recommendation.append("False (Preserve Data)");
                }

                return recommendation.toString();
        }

        /**
         * Quick domain-level advice when no dataset is loaded.
         * Falls back to a hardcoded heuristic if LLM is unavailable.
         */
        public String quickAdvice(com.example.automl_prototype_1.model.Domain domain) {
                String domainName = domain != null ? domain.name() : "ML";
                if (llmService != null && llmService.isAvailable()) {
                        String prompt = "Give 3 concise algorithm recommendations for a " + domainName
                                        + " task (no dataset loaded). " +
                                        "Format as: Algorithm → Reason. Max 3 lines.";
                        return llmService.queryOllama(prompt, "default");
                }
                return switch (domainName) {
                        case "RL" ->
                                "Start with DQN for discrete action spaces; PPO for continuous.\nUse IGPSO for hyperparameter search.";
                        case "LLM" ->
                                "Use Mistral-7B (via Ollama) for instruction following.\nFine-tune with LoRA for domain adaptation.";
                        case "DL" ->
                                "ResNet-50 for image tasks; LSTM or Transformer for sequences.\nUse WW0 for architecture search.";
                        default ->
                                "Random Forest or XGBoost are strong baselines for tabular ML.\nRun IGPSO or GA for feature selection.";
                };
        }

        /**
         * Post-run advice based on execution results.
         */
        public String quickAdvice(List<com.example.automl_prototype_1.model.ExecutionResult> results) {
                if (results == null || results.isEmpty())
                        return "No results to analyze.";

                com.example.automl_prototype_1.model.ExecutionResult best = results.stream()
                                .max(java.util.Comparator.comparingDouble(
                                                com.example.automl_prototype_1.model.ExecutionResult::getAccuracy))
                                .orElse(results.get(0));

                if (llmService != null && llmService.isAvailable()) {
                        String prompt = String.format("The best algorithm was %s with an accuracy of %.2f%%. " +
                                        "Give a very concise, 1-2 sentence recommendation on what the user should try next to improve this.",
                                        best.getAlgorithmName(), best.getAccuracy() * 100);
                        return llmService.queryOllama(prompt, "active");
                }

                if (best.getAccuracy() < 0.70) {
                        return "Accuracy is low (" + String.format("%.1f", best.getAccuracy() * 100)
                                        + "%). Try enabling Feature Selection or Hyperparameter Tuning.";
                } else if (best.getAccuracy() > 0.95) {
                        return "Excellent accuracy (" + String.format("%.1f", best.getAccuracy() * 100)
                                        + "%). Watch out for overfitting. Consider testing on a new dataset.";
                } else {
                        return "Solid performance (" + String.format("%.1f", best.getAccuracy() * 100)
                                        + "%). You might extract more juice by using Stacked Generalization or Neural Networks.";
                }
        }
}
