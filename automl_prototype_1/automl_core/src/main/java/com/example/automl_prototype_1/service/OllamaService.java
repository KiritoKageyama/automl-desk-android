package com.example.automl_prototype_1.service;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;

import java.util.concurrent.CompletableFuture;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.example.automl_prototype_1.service.agentic.ProjectContextService;
import com.example.automl_prototype_1.service.agentic.CodeModificationService;

public class OllamaService {

    private static OllamaService instance;
    private final HttpClient httpClient;
    private final ObjectMapper mapper;
    private final ProjectContextService projectContextService;
    private final CodeModificationService codeModService;

    private OllamaService() {
        this.httpClient = HttpClient.newBuilder()
                .connectTimeout(Duration.ofSeconds(10))
                .build();
        this.mapper = new ObjectMapper();
        this.projectContextService = new ProjectContextService();
        this.codeModService = new CodeModificationService();
    }

    public static synchronized OllamaService getInstance() {
        if (instance == null) {
            instance = new OllamaService();
        }
        return instance;
    }

    public CompletableFuture<String> chat(String userMessage) {
        AppStateService appState = AppStateService.getInstance();
        if (!appState.isOllamaEnabled()) {
            return CompletableFuture.completedFuture("AI Assistant is disabled in Settings.");
        }

        String model = appState.getOllamaModelName();
        String url = appState.getOllamaUrl() + "/api/generate"; // Use generate for simple prompt, or /api/chat

        // Build System Context
        String systemContext = buildSystemContext(appState);
        String fullPrompt = systemContext + "\n\nUser Question: " + userMessage + "\n\nAssistant Response:";

        // JSON Body
        ObjectNode json = mapper.createObjectNode();
        json.put("model", model);
        json.put("prompt", fullPrompt);
        json.put("stream", false);

        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(url))
                .header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(json.toString()))
                .build();

        return httpClient.sendAsync(request, HttpResponse.BodyHandlers.ofString())
                .thenApply(response -> {
                    if (response.statusCode() == 200) {
                        try {
                            com.fasterxml.jackson.databind.JsonNode resJson = mapper.readTree(response.body());
                            if (resJson.has("response")) {
                                return resJson.get("response").asText();
                            }
                            return "No response content.";
                        } catch (Exception e) {
                            return "Error parsing response: " + e.getMessage();
                        }
                    } else if (response.statusCode() == 404) {
                        return "Error: Model '" + model + "' not found. Check Settings or run 'ollama pull " + model
                                + "'.";
                    } else {
                        return "Error: Ollama returned status " + response.statusCode();
                    }
                })
                .exceptionally(e -> {
                    // Try to auto-start if connection fails
                    if (e.getMessage().contains("ConnectException") || e.getMessage().contains("Connection refused")) {
                        boolean started = tryStartOllamaServer();
                        if (started) {
                            return "Ollama server was not running. I have attempted to start it. Please wait a few seconds and try again.";
                        }
                    }
                    return "Error connecting to Ollama: " + e.getMessage() + ". Is it running?";
                });
    }

    private boolean tryStartOllamaServer() {
        try {
            System.out.println("Attempting to auto-start Ollama server...");
            // Use ProcessBuilder to start 'ollama serve'
            // We assume 'ollama' is in PATH or known location
            ProcessBuilder pb = new ProcessBuilder("ollama", "serve");
            pb.redirectErrorStream(true);
            pb.start();
            // Give it a moment to spin up
            Thread.sleep(2000);
            return true;
        } catch (Exception ex) {
            System.err.println("Failed to auto-start Ollama: " + ex.getMessage());
            // Try explicit path check as fallback?
            return false;
        }
    }

    private String buildSystemContext(AppStateService appState) {
        StringBuilder sb = new StringBuilder();
        sb.append("You are an expert AutoML Assistant integrated into a Java application.\n");
        sb.append("Current Context:\n");

        if (appState.getSelectedDatasetFile() != null) {
            sb.append("- Dataset: ").append(appState.getSelectedDatasetFile().getName()).append("\n");
        } else {
            sb.append("- No dataset selected.\n");
        }

        sb.append("- Selected Algorithms: ").append(appState.getSelectedAlgorithmNames()).append("\n");
        sb.append("- GPU Enabled: ").append(appState.isUseGpu()).append("\n");

        sb.append("Provide concise, helpful insights about machine learning, the dataset, or the app's features.\n\n");

        // === CONTEXT INJECTION: PROJECT STRUCTURE ===
        sb.append("--- PROJECT STRUCTURE ---\n");
        sb.append(projectContextService.getProjectStructure());
        sb.append("--- END PROJECT STRUCTURE ---\n\n");

        // === CONTEXT INJECTION: INTERFACE CONTRACT ===
        sb.append("--- INTERFACE CONTRACT (MANDATORY) ---\n");
        sb.append("Any classifier you generate MUST implement this Java interface exactly:\n");
        try {
            String iClassifierSource = codeModService.readFile(
                    "automl_core/src/main/java/com/example/automl_prototype_1/service/IClassifier.java");
            sb.append(iClassifierSource);
        } catch (Exception e) {
            sb.append(
                    "// IClassifier: void train(double[][] features, int[] labels); int[] predict(double[][] features); String getName();\n");
        }
        sb.append("\n--- END INTERFACE CONTRACT ---\n\n");

        // === CONTEXT INJECTION: EXAMPLE IMPLEMENTATION ===
        sb.append("--- EXAMPLE CLASSIFIER IMPLEMENTATION (Follow This Pattern) ---\n");
        try {
            String exampleClassifier = codeModService.readFile(
                    "automl_core/src/main/java/com/example/automl_prototype_1/classifier/LogisticRegression.java");
            sb.append(exampleClassifier);
        } catch (Exception e) {
            sb.append("// Example unavailable. Follow IClassifier contract above.\n");
        }
        sb.append("\n--- END EXAMPLE ---\n\n");

        // --- CODING CAPABILITY UPGRADE ---
        sb.append("--- CODING ASSISTANT CAPABILITY ENABLED ---\n");
        sb.append(
                "You have the ability to modify the project code. To request a code change, you MUST use the following XML format:\n");
        sb.append("<edit_request>\n");
        sb.append("  <file>relative/path/to/file.java</file>\n");
        sb.append("  <content>\n");
        sb.append("     // Entire new content of the file (or sufficient context)\n");
        sb.append("     public class ... \n");
        sb.append("  </content>\n");
        sb.append("</edit_request>\n");
        sb.append("To CREATE a new file, use:\n");
        sb.append("<create_file>\n");
        sb.append("  <path>relative/path/to/new_file.java</path>\n");
        sb.append("  <content>\n");
        sb.append("     // File content\n");
        sb.append("  </content>\n");
        sb.append("</create_file>\n");
        sb.append("To MODIFY the Constitution (Rules), use:\n");
        sb.append("<update_constitution>\n");
        sb.append("  <id>unique_rule_id</id>\n");
        sb.append("  <condition>meta.numInstances > 50000</condition>\n");
        sb.append("  <verdict>WARN</verdict>\n");
        sb.append("  <reason>Reason for warning...</reason>\n");
        sb.append("</update_constitution>\n");
        sb.append("To ENABLE/DISABLE Strict Mode (Absolute Rules), use:\n");
        sb.append("<set_strict_mode>true</set_strict_mode>\n");
        sb.append("IMPORTANT: You are PROHIBITED from modifying files in 'automl_core' or 'desktop-app'.\n");
        sb.append("If you need to change core logic (e.g., a Classifier), you MUST:\n");
        sb.append("1. Read the core file to understand it.\n");
        sb.append("2. Create a DUPLICATE in a new folder called 'custom_extensions'.\n");
        sb.append("3. Modify the duplicate.\n");
        sb.append("Use this sparingly and ONLY when explicitly asked to change code. Otherwise, just chat normally.");
        return sb.toString();
    }
}
