package com.example.automl_prototype_1.service;

import com.example.automl_prototype_1.service.agentic.DomainGuardService;

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
 * LLM client service: manages the Ollama process lifecycle (Start-on-Demand,
 * Stop-on-Idle), provides queryOllama() for all agentic services, and exposes
 * safety controls (emergency stop, interrupt, clear cache).
 */
public class OllamaService {

    private static final String OLLAMA_API_URL = "http://localhost:11434/api/generate";
    private static final int TIMEOUT_MS = 5000;

    // Domain boundary enforcement
    private final DomainGuardService domainGuardService;

    // Lifecycle Management
    private static Process ollamaProcess = null;
    private static final ScheduledExecutorService scheduler = Executors.newSingleThreadScheduledExecutor();
    private static final long IDLE_TIMEOUT_MS = 5 * 60 * 1000; // 5 Minutes
    private static final AtomicLong lastActivityTime = new AtomicLong(System.currentTimeMillis());
    private static boolean isMonitoring = false;

    private String cachedModel = null;

    private static OllamaService instance;

    public static synchronized OllamaService getInstance() {
        if (instance == null) {
            instance = new OllamaService();
        }
        return instance;
    }

    private OllamaService() {
        this.domainGuardService = new DomainGuardService();

        // Register Shutdown Hook
        Runtime.getRuntime().addShutdownHook(new Thread(() -> {
            com.example.automl_prototype_1.service.LogService.info("OllamaService", "Shutdown hook triggered.");
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
            synchronized (OllamaService.class) {
                if (ollamaProcess != null && ollamaProcess.isAlive()) {
                    long idleTime = System.currentTimeMillis() - lastActivityTime.get();
                    if (idleTime > IDLE_TIMEOUT_MS) {
                        com.example.automl_prototype_1.service.LogService.info("OllamaService",
                                "Ollama idle for 5m. Shutting down to save resources.");
                        stopOllamaProcess();
                    }
                }
            }
        }, 1, 1, TimeUnit.MINUTES);
    }

    private synchronized void stopOllamaProcess() {
        if (ollamaProcess != null) {
            ollamaProcess.destroyForcibly();
            ollamaProcess = null;
            com.example.automl_prototype_1.service.LogService.info("OllamaService", "Ollama process terminated.");
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
        com.example.automl_prototype_1.service.LogService.warn("OllamaService", "🚨 EMERGENCY STOP TRIGGERED 🚨");
        stopOllamaProcess();
    }

    /**
     * Interrupts the current generation loop (Soft Stop).
     */
    public void interrupt() {
        com.example.automl_prototype_1.service.LogService.info("OllamaService", "Interrupt signal received.");
    }

    /**
     * Clears LLM context/cache and restarts.
     */
    public void clearCache() {
        com.example.automl_prototype_1.service.LogService.info("OllamaService", "Clearing LLM Cache...");
        this.cachedModel = null;
        stopOllamaProcess();
        // It will auto-restart on next request
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
            com.example.automl_prototype_1.service.LogService.warn("OllamaService",
                    "Domain Boundary Rejection: Query outside of CS domain. Prompt: " +
                            (prompt.length() > 50 ? prompt.substring(0, 50) + "..." : prompt));
            return "[DOMAIN REJECTION] I can only assist with Computer Science related queries (Algorithms, ML, Systems, Coding, Security).";
        }

        lastActivityTime.set(System.currentTimeMillis()); // Beat heart

        // PRIORITY: Try Embedded SLM (TinyLlama) via GPU Server (Phase 8)
        // ONLY if reserved mode is NOT "DEEP_THOUGHT"
        if (!"DEEP_THOUGHT".equals(reserved)) {
            try {
                com.example.automl_prototype_1.gpu.PythonExecutionService pythonService = com.example.automl_prototype_1.gpu.PythonExecutionService
                        .getInstance();

                if (pythonService.isServerRunning()) {
                    // 1. RECALL MEMORIES (Phase 8)
                    // We construct a query context from the usage prompt
                    java.util.List<java.util.Map<String, Object>> memories = pythonService.queryMemory(prompt, 3);

                    StringBuilder memoryContext = new StringBuilder();
                    if (!memories.isEmpty()) {
                        memoryContext.append("\n[RECALLED MEMORIES]:\n");
                        for (java.util.Map<String, Object> mem : memories) {
                            memoryContext.append("- ").append(mem.get("text")).append("\n");
                        }
                    }

                    // 2. INJECT CONSTITUTION (Phase 8)
                    // We inject high-level rules if the prompt involves "strategy" or "code"
                    StringBuilder constitutionContext = new StringBuilder();
                    if (prompt.toLowerCase().contains("strategy") || prompt.toLowerCase().contains("code")
                            || prompt.toLowerCase().contains("recommend")) {
                        constitutionContext.append("\n[CONSTITUTIONAL RULES]:\n");
                        // Hardcoded high-level summary for now, or could fetch from service if we
                        // expose a "summarize" method
                        constitutionContext.append("- prioritize accuracy over speed unless specified\n");
                        constitutionContext.append("- avoid algorithms that are known to freeze on large datasets\n");
                        constitutionContext.append("- never generate malicious code patterns\n");
                    }

                    // 3. CONSTRUCT SYSTEM PROMPT
                    String systemPrompt = "You are an expert AutoML Assistant. " +
                            "Use the provided Memories and Constitution to guide your answer.\n" +
                            constitutionContext.toString() +
                            memoryContext.toString();

                    String response = pythonService.chatWithSLM(systemPrompt, prompt);

                    if (response != null && !response.startsWith("Error:")) {
                        com.example.automl_prototype_1.service.LogService.info("OllamaService",
                                "Served via Embedded SLM (TinyLlama) with RAG & Constitution");
                        return response;
                    }
                }
            } catch (Exception e) {
                com.example.automl_prototype_1.service.LogService.warn("OllamaService",
                        "Embedded SLM failed, falling back to Ollama: " + e.getMessage());
            }
        } else {
            com.example.automl_prototype_1.service.LogService.info("OllamaService",
                    "Bypassing SLM for Deep Thought (System 2)...");
        }

        // FALLBACK: External Ollama Instance
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
                com.example.automl_prototype_1.service.LogService.warn("OllamaService",
                        "Ollama API Error: " + status);
                return null;
            }

            com.fasterxml.jackson.databind.JsonNode responseNode = objectMapper.readTree(conn.getInputStream());
            if (responseNode.has("response")) {
                return responseNode.get("response").asText().trim();
            }
            return null;

        } catch (Exception e) {
            com.example.automl_prototype_1.service.LogService.warn("OllamaService",
                    "Query Failed: " + e.getMessage());
            return queryCloudFallback(prompt);
        }
    }

    private String queryCloudFallback(String prompt) {
        com.example.automl_prototype_1.service.LogService.info("OllamaService", "Attempting Cloud Fallback via Anthropic (Claude 3.7)...");
        AnthropicService anthropicService = AnthropicService.getInstance();
        if (anthropicService.isAvailable()) {
            return anthropicService.queryClaude(prompt);
        }
        
        com.example.automl_prototype_1.service.LogService.warn("OllamaService", "Cloud Fallback failed: AnthropicService unavailable.");
        return null;
    }

    /**
     * Queries the "Big Brain" (DeepSeek/Gemma) directly, bypassing the local SLM.
     * Used by AntigravityObserver for deep analysis.
     */
    public String queryDeep(String prompt) {
        return queryOllama(prompt, "DEEP_THOUGHT");
    }

    /**
     * Asynchronous chat method for UI Controllers.
     */
    public java.util.concurrent.CompletableFuture<String> chat(String message) {
        return java.util.concurrent.CompletableFuture.supplyAsync(() -> queryOllama(message, "chat"));
    }
}
