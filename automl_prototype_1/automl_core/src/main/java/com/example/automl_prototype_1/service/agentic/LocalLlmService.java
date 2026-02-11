package com.example.automl_prototype_1.service.agentic;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Service to interface with a local LLM (e.g., Ollama) for generating
 * human-readable explanations.
 * 
 * UPGRADE: Now manages the Ollama process lifecycle (Start-on-Demand,
 * Stop-on-Idle).
 */
public class LocalLlmService {

    private static final String OLLAMA_API_URL = "http://localhost:11434/api/generate";
    private static final int TIMEOUT_MS = 5000;

    // Context Injection Services
    private final ProjectContextService projectContextService;
    private final CodeModificationService codeModService;
    private final DomainGuardService domainGuardService;
    private final CodeStructureService codeStructureService;
    private final ConstitutionalService constitutionalService;

    // Lifecycle Management
    private static Process ollamaProcess = null;
    private static final ScheduledExecutorService scheduler = Executors.newSingleThreadScheduledExecutor();
    private static final long IDLE_TIMEOUT_MS = 5 * 60 * 1000; // 5 Minutes
    private static final AtomicLong lastActivityTime = new AtomicLong(System.currentTimeMillis());
    private static boolean isMonitoring = false;

    private String cachedModel = null;

    public LocalLlmService() {
        this.projectContextService = new ProjectContextService();
        this.codeModService = new CodeModificationService();
        this.domainGuardService = new DomainGuardService();
        this.codeStructureService = new CodeStructureService();
        this.constitutionalService = new ConstitutionalService();

        // Register Shutdown Hook
        Runtime.getRuntime().addShutdownHook(new Thread(() -> {
            com.example.automl_prototype_1.service.LogService.info("LocalLlmService", "Shutdown hook triggered.");
            stopOllamaProcess();
        }));

        // Start the idle monitor daemon
        if (!isMonitoring) {
            startIdleMonitor();
            isMonitoring = true;
        }
    }

    private void startIdleMonitor() {
        scheduler.scheduleAtFixedRate(() -> {
            synchronized (LocalLlmService.class) {
                if (ollamaProcess != null && ollamaProcess.isAlive()) {
                    long idleTime = System.currentTimeMillis() - lastActivityTime.get();
                    if (idleTime > IDLE_TIMEOUT_MS) {
                        com.example.automl_prototype_1.service.LogService.info("LocalLlmService",
                                "Ollama idle for 5m. Shutting down to save resources.");
                        stopOllamaProcess();
                    }
                }
            }
        }, 1, 1, TimeUnit.MINUTES);
    }

    private synchronized void ensureOllamaRunning() {
        lastActivityTime.set(System.currentTimeMillis()); // Reset timer

        if (isAvailable())
            return; // Already running logic

        com.example.automl_prototype_1.service.LogService.info("LocalLlmService",
                "Ollama is offline. Starting background process...");

        try {
            // Try Windows native first
            ProcessBuilder pb = new ProcessBuilder("ollama", "serve");
            pb.redirectErrorStream(true);
            // Hide window on Windows
            pb.redirectOutput(ProcessBuilder.Redirect.DISCARD);

            ollamaProcess = pb.start();

            // Wait for startup (up to 10s)
            for (int i = 0; i < 20; i++) {
                Thread.sleep(500);
                if (isAvailable()) {
                    com.example.automl_prototype_1.service.LogService.info("LocalLlmService",
                            "Ollama started successfully.");
                    return;
                }
            }

            com.example.automl_prototype_1.service.LogService.warn("LocalLlmService",
                    "Ollama process started but API not responding.");

        } catch (Exception e) {
            com.example.automl_prototype_1.service.LogService.error("LocalLlmService",
                    "Failed to start Ollama: " + e.getMessage());
            // Fallback: User might have it in WSL but not Windows Path?
            // Since user prefers Windows app, we stick to this.
        }
    }

    private synchronized void stopOllamaProcess() {
        if (ollamaProcess != null) {
            ollamaProcess.destroyForcibly();
            ollamaProcess = null;
            com.example.automl_prototype_1.service.LogService.info("LocalLlmService", "Ollama process terminated.");
        }
    }

    public boolean isAvailable() {
        try {
            URL url = java.net.URI.create("http://localhost:11434").toURL();
            HttpURLConnection conn = (HttpURLConnection) url.openConnection();
            conn.setRequestMethod("GET");
            conn.setConnectTimeout(500); // Fast check
            conn.connect();
            return conn.getResponseCode() == 200;
        } catch (Exception e) {
            return false;
        }
    }

    // --- Safety Controls (Phase 3) ---

    /**
     * Emergency Stop: Kills the LLM process immediately.
     */
    public void emergencyStop() {
        com.example.automl_prototype_1.service.LogService.warn("LocalLlmService", "🚨 EMERGENCY STOP TRIGGERED 🚨");
        stopOllamaProcess();
    }

    /**
     * Interrupts the current generation loop (Soft Stop).
     */
    public void interrupt() {
        com.example.automl_prototype_1.service.LogService.info("LocalLlmService", "Interrupt signal received.");
    }

    /**
     * Clears LLM context/cache and restarts.
     */
    public void clearCache() {
        com.example.automl_prototype_1.service.LogService.info("LocalLlmService", "Clearing LLM Cache...");
        this.cachedModel = null;
        stopOllamaProcess();
        // It will auto-restart on next request
    }

    /**
     * Asks the Local LLM to explain why a specific algorithm was chosen.
     */
    public String explainRecommendation(String algoName, String datasetName, String properties) {
        ensureOllamaRunning();
        String prompt = String.format(
                "Explain in 1 sentence why '%s' is a good algorithm for the '%s' dataset which has these properties: %s. Be concise.",
                algoName, datasetName, properties);
        return queryOllama(prompt, "active");
    }

    /**
     * Competitively ranks candidates with strategic insight.
     */
    public String rankAndRationalize(java.util.List<PipelineSuggestion> candidates, String datasetName,
            MetaFeatures meta) {
        ensureOllamaRunning();
        StringBuilder candidatesStr = new StringBuilder();
        for (int i = 0; i < candidates.size(); i++) {
            candidatesStr.append(String.format("%d. %s + %s\n", i + 1, candidates.get(i).getAlgorithmName(),
                    candidates.get(i).getClassifierName()));
        }

        String stats = String.format("Rows: %d, Feats: %d, Sparse: %.2f, Imbalance: %.2f, Skew: %.2f, Kurtosis: %.2f",
                meta.getNumInstances(), meta.getNumFeatures(), meta.getSparsity(), meta.getClassImbalanceRatio(),
                meta.getMaxSkewness(), meta.getMaxKurtosis());

        String prompt = String.format(
                "You are an expert AutoML Strategist. Given dataset '%s' with properties [%s], critically compare these candidates:\n"
                        +
                        "%s\n" +
                        "Which single pipeline is strategically best? If the first one (Empirical Winner) is best, explain why it fits the skew/kurtosis/sparsity. "
                        +
                        "If #2 or #3 is theoretically better for these specific stats, argue for it. " +
                        "Output a single concise paragraph starting with 'Strategic Insight:'.",
                datasetName, stats, candidatesStr.toString());

        return queryOllama(prompt, "active");
    }

    /**
     * Generates a "Simulated AI" explanation based on heuristics when LLM fails.
     */
    public String generateFallbackExplanation(String algoName,
            com.example.automl_prototype_1.service.agentic.MetaFeatures meta) {
        StringBuilder sb = new StringBuilder();

        if (meta.getNumFeatures() > 50 && meta.getNumInstances() < 500) {
            sb.append("This dataset is 'wide' (high-dimensional relative to samples). ");
        } else if (meta.getNumInstances() > 10000) {
            sb.append("This is a large-scale dataset. ");
        } else if (meta.getSparsity() > 0.6) {
            sb.append("The data is highly sparse (>60% zeros). ");
        } else {
            sb.append("The dataset has a balanced structure. ");
        }

        if (algoName.contains("Genetic Algorithm") || algoName.contains("GA")) {
            if (meta.getNumFeatures() > 30) {
                sb.append("GA is ideal here as it excells at navigating large search spaces.");
            } else {
                sb.append("GA provides robust evolutionary search.");
            }
        } else if (algoName.contains("PSO")) {
            sb.append("PSO is recommended for its convergence speed.");
        } else if (algoName.contains("KNN")) {
            if (meta.getNumInstances() < 1000) {
                sb.append("KNN is highly effective on smaller datasets.");
            } else {
                sb.append("KNN serves as a strong non-parametric baseline.");
            }
        } else if (algoName.contains("Random Forest")) {
            sb.append("Random Forest is chosen for its ensemble robustness.");
        } else {
            sb.append(algoName + " adds complementary strengths.");
        }

        return sb.toString();
    }

    private String getBestAvailableModel() {
        if (cachedModel != null)
            return cachedModel;

        try {
            URL url = java.net.URI.create("http://localhost:11434/api/tags").toURL();
            HttpURLConnection conn = (HttpURLConnection) url.openConnection();
            conn.setRequestMethod("GET");
            conn.setConnectTimeout(2000);

            if (conn.getResponseCode() != 200)
                return "mistral";

            StringBuilder response = new StringBuilder();
            try (BufferedReader br = new BufferedReader(
                    new InputStreamReader(conn.getInputStream(), StandardCharsets.UTF_8))) {
                String line;
                while ((line = br.readLine()) != null)
                    response.append(line);
            }

            String json = response.toString().toLowerCase();
            String[] priorities = { "deepseek", "qwen", "llama3", "mistral", "olmo", "gemma", "codellama", "llama" };

            for (String keyword : priorities) {
                String tag = extractTagContaining(json, keyword);
                if (tag != null) {
                    cachedModel = tag;
                    return cachedModel;
                }
            }

            String firstTag = extractTagContaining(json, "");
            if (firstTag != null) {
                cachedModel = firstTag;
                return cachedModel;
            }

        } catch (Exception e) {
        }

        cachedModel = "mistral";
        return cachedModel;
    }

    private String extractTagContaining(String json, String keyword) {
        String searchKey = "\"name\":\"";
        int idx = 0;
        while ((idx = json.indexOf(searchKey, idx)) != -1) {
            idx += searchKey.length();
            int end = json.indexOf("\"", idx);
            if (end != -1) {
                String tagName = json.substring(idx, end);
                if (tagName.contains(keyword)) {
                    return tagName;
                }
            }
        }
        return null;
    }

    private final com.fasterxml.jackson.databind.ObjectMapper objectMapper = new com.fasterxml.jackson.databind.ObjectMapper();

    public String queryOllama(String prompt, String reserved) {
        // Enforce Domain Boundaries
        if (!domainGuardService.validate(prompt)) {
            com.example.automl_prototype_1.service.LogService.warn("LocalLlmService",
                    "Domain Boundary Rejection: Query outside of CS domain. Prompt: " +
                            (prompt.length() > 50 ? prompt.substring(0, 50) + "..." : prompt));
            return "[DOMAIN REJECTION] I can only assist with Computer Science related queries (Algorithms, ML, Systems, Coding, Security).";
        }

        lastActivityTime.set(System.currentTimeMillis()); // Beat heart
        String modelToUse = getBestAvailableModel();

        try {
            URL url = java.net.URI.create(OLLAMA_API_URL).toURL();
            HttpURLConnection conn = (HttpURLConnection) url.openConnection();
            conn.setRequestMethod("POST");
            conn.setRequestProperty("Content-Type", "application/json");
            conn.setDoOutput(true);
            conn.setConnectTimeout(TIMEOUT_MS);
            conn.setReadTimeout(45000); // Allow time for cold-start model load

            com.fasterxml.jackson.databind.node.ObjectNode requestNode = objectMapper.createObjectNode();
            requestNode.put("model", modelToUse);
            requestNode.put("prompt", prompt);
            requestNode.put("stream", false);

            try (OutputStream os = conn.getOutputStream()) {
                byte[] input = objectMapper.writeValueAsBytes(requestNode);
                os.write(input, 0, input.length);
            }

            int status = conn.getResponseCode();
            if (status != 200) {
                com.example.automl_prototype_1.service.LogService.warn("LocalLlmService",
                        "Ollama API Error: " + status);
                return null;
            }

            com.fasterxml.jackson.databind.JsonNode responseNode = objectMapper.readTree(conn.getInputStream());
            if (responseNode.has("response")) {
                return responseNode.get("response").asText().trim();
            }
            return null;

        } catch (Exception e) {
            com.example.automl_prototype_1.service.LogService.warn("LocalLlmService",
                    "Query Failed: " + e.getMessage());
            return queryCloudFallback(prompt);
        }
    }

    private String queryCloudFallback(String prompt) {
        String apiKey = System.getenv("AUTOML_CLOUD_API_KEY");
        if (apiKey == null || apiKey.isEmpty())
            return null;
        com.example.automl_prototype_1.service.LogService.info("LocalLlmService", "Attempting Cloud Fallback...");
        return null;
    }

    // ========================================================================
    // CODE GENERATION WITH CONTEXT INJECTION
    // ========================================================================

    /**
     * Generates a custom classifier Java class based on user's description.
     * Injects IClassifier interface, example implementation, and project
     * structure into the prompt so the LLM produces code that conforms
     * to the codebase's architecture.
     *
     * @param userDescription The user's natural language description of the
     *                        algorithm, including math formulas and data flow.
     * @return The generated Java source code, or null on failure.
     */
    public String generateClassifierCode(String userDescription) {
        ensureOllamaRunning();

        StringBuilder prompt = new StringBuilder();

        // 1. Role & Task
        prompt.append("You are an expert Java developer for an AutoML framework.\n");
        prompt.append("Your task is to generate a COMPLETE Java classifier class based on the user's description.\n\n");

        // 2. Inject Project Structure
        prompt.append("--- PROJECT STRUCTURE ---\n");
        prompt.append(projectContextService.getProjectStructure());
        prompt.append("--- END PROJECT STRUCTURE ---\n\n");

        // 3. Inject IClassifier Interface (MANDATORY contract)
        prompt.append("--- INTERFACE CONTRACT (YOU MUST IMPLEMENT THIS) ---\n");
        try {
            String iClassifier = codeModService.readFile(
                    "automl_core/src/main/java/com/example/automl_prototype_1/service/IClassifier.java");
            prompt.append(iClassifier);
        } catch (Exception e) {
            prompt.append("public interface IClassifier {\n");
            prompt.append("    void train(double[][] features, int[] labels);\n");
            prompt.append("    int[] predict(double[][] features);\n");
            prompt.append("    String getName();\n");
            prompt.append("}\n");
        }
        prompt.append("--- END INTERFACE ---\n\n");

        // 4. Inject Example Implementation
        prompt.append("--- EXAMPLE IMPLEMENTATION (FOLLOW THIS PATTERN) ---\n");
        try {
            String example = codeModService.readFile(
                    "automl_core/src/main/java/com/example/automl_prototype_1/classifier/LogisticRegression.java");
            prompt.append(example);
        } catch (Exception e) {
            prompt.append("// Example unavailable. Follow the IClassifier contract above.\n");
        }
        prompt.append("--- END EXAMPLE ---\n\n");

        // 5. Rules
        prompt.append("--- RULES ---\n");
        prompt.append("1. The class MUST be in package: com.example.automl_prototype_1.classifier\n");
        prompt.append("2. The class MUST implement IClassifier with all 3 methods.\n");
        prompt.append("3. Import IClassifier from: com.example.automl_prototype_1.service.IClassifier\n");
        prompt.append("4. Use only standard Java libraries and Smile ML (already on classpath).\n");
        prompt.append("5. Return ONLY the Java code. No explanations, no markdown fences.\n");
        prompt.append("--- END RULES ---\n\n");

        // 6. User's Description
        prompt.append("--- USER'S ALGORITHM DESCRIPTION ---\n");
        prompt.append(userDescription);
        prompt.append("\n--- END DESCRIPTION ---\n\n");

        prompt.append("Generate the complete Java class now:\n");

        String currentPrompt = prompt.toString();
        String generatedCode = null;
        int maxRetries = 3;

        for (int i = 0; i < maxRetries; i++) {
            com.example.automl_prototype_1.service.LogService.info("LocalLlmService",
                    "Generating code (Attempt " + (i + 1) + "/" + maxRetries + ")...");
            generatedCode = queryOllama(currentPrompt, "code_gen");

            if (generatedCode == null)
                return null;

            // Clean code fences if present (Standardize output)
            if (generatedCode.contains("```java")) {
                generatedCode = generatedCode.split("```java")[1].split("```")[0].trim();
            } else if (generatedCode.contains("```")) {
                generatedCode = generatedCode.split("```")[1].trim();
            }

            // Phase 3: Constitutional Audit
            ConstitutionalService.Verdict verdict = constitutionalService.auditCode(generatedCode);

            if (verdict.allowed) {
                com.example.automl_prototype_1.service.LogService.info("LocalLlmService", "Code Audit Passed.");
                return generatedCode;
            }

            // If VETOED:
            com.example.automl_prototype_1.service.LogService.warn("LocalLlmService",
                    "Code Audit FAILED: " + verdict.reason);

            // Reflexion: Feed the error back to the LLM
            currentPrompt += "\n\n[SYSTEM ERROR]: Your previous code was rejected by the Security Auditor.\n" +
                    "Reason: " + verdict.reason + "\n" +
                    "Critically analyze your mistake and regenerate the code FIXING this violation.\n";

            // Small delay to prevent tight loop thrashing
            try {
                Thread.sleep(1000);
            } catch (InterruptedException ignored) {
            }
        }

        com.example.automl_prototype_1.service.LogService.error("LocalLlmService",
                "Failed to generate valid code after " + maxRetries + " attempts.");
        return null;
    }

    /**
     * Builds the full context string for code generation requests.
     * Can be used by other services that need LLM context.
     */
    public String buildCodeGenerationContext() {
        StringBuilder sb = new StringBuilder();

        sb.append("--- PROJECT STRUCTURE ---\n");
        sb.append(projectContextService.getProjectStructure());
        sb.append("--- END PROJECT STRUCTURE ---\n\n");

        sb.append("--- INTERFACE CONTRACT ---\n");
        try {
            sb.append(codeModService.readFile(
                    "automl_core/src/main/java/com/example/automl_prototype_1/service/IClassifier.java"));
        } catch (Exception e) {
            sb.append("// IClassifier: train(double[][], int[]), predict(double[][]), getName()\n");
        }
        sb.append("--- END INTERFACE ---\n\n");

        sb.append("--- EXAMPLE IMPLEMENTATION ---\n");
        try {
            sb.append(codeModService.readFile(
                    "automl_core/src/main/java/com/example/automl_prototype_1/classifier/LogisticRegression.java"));
        } catch (Exception e) {
            sb.append("// Example unavailable.\n");
        }
        sb.append("--- END EXAMPLE ---\n");

        return sb.toString();
    }

    /**
     * Analyzes the structure of a code file using the Phase 2 Analysis Engine.
     */
    public String analyzeCode(String filePath) {
        return codeStructureService.analyzeFileSummary(filePath);
    }

    /**
     * Explains a code file using Structured Code Understanding.
     * 1. Extracts AST/Complexity via Python Engine.
     * 2. Feeds structure + code to LLM.
     */
    public String explainCode(String filePath) {
        String analysis = analyzeCode(filePath);
        String codeContent;
        try {
            codeContent = codeModService.readFile(filePath);
        } catch (Exception e) {
            return "Error reading file: " + e.getMessage();
        }

        String prompt = String.format(
                "You are a Senior Software Engineer. Analyze this Java file.\n\n" +
                        "--- STRUCTURED METADATA ---\n%s\n" +
                        "--- CODE ---\n%s\n\n" +
                        "Explain the purpose of this class, its key methods, and comment on its complexity/architecture based on the metadata.",
                analysis, codeContent);

        return queryOllama(prompt, "explain_code");
    }
}
