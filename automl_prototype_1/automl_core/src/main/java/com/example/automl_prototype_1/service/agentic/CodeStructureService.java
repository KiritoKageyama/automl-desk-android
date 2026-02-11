package com.example.automl_prototype_1.service.agentic;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.example.automl_prototype_1.service.LogService;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.concurrent.TimeUnit;

/**
 * Service that invokes the Python 'analysis_engine.py' to extract structured
 * code metadata.
 * Uses Tree-sitter for robust parsing.
 */
public class CodeStructureService {

    private static final String SCRIPT_PATH = "automl_core/src/main/resources/python/analysis_engine.py";
    private final ObjectMapper objectMapper;

    public CodeStructureService() {
        this.objectMapper = new ObjectMapper();
    }

    /**
     * Analyzes a file and returns structured metadata.
     * 
     * @param filePath Absolute or relative path to the file.
     * @return Formatted string summary of the analysis (for LLM context).
     */
    public String analyzeFileSummary(String filePath) {
        try {
            JsonNode result = runAnalysisEngine(filePath);
            if (result == null || result.has("error")) {
                return "// Structure Analysis Unavailable: "
                        + (result != null ? result.get("error").asText() : "Unknown error");
            }

            StringBuilder sb = new StringBuilder();
            sb.append("--- STRUCTURED ANALYSIS (AST) ---\n");

            if (result.has("classes") && result.get("classes").isArray()) {
                result.get("classes").forEach(c -> {
                    sb.append(String.format("Class: %s (Lines: %d-%d)\n",
                            c.get("name").asText(),
                            c.get("start_line").asInt(),
                            c.get("end_line").asInt()));
                });
            }

            if (result.has("methods") && result.get("methods").isArray()) {
                sb.append("Methods:\n");
                result.get("methods").forEach(m -> {
                    int complexity = m.get("complexity").asInt();
                    String complexityWarning = complexity > 10 ? " [HIGH_COMPLEXITY]" : "";
                    sb.append(String.format("  - %s (Complexity: %d)%s (Lines: %d-%d)\n",
                            m.get("name").asText(),
                            complexity,
                            complexityWarning,
                            m.get("start_line").asInt(),
                            m.get("end_line").asInt()));
                });
            }

            if (result.has("functions") && result.get("functions").isArray()) {
                sb.append("Functions:\n");
                result.get("functions").forEach(f -> {
                    int complexity = f.get("complexity").asInt();
                    String complexityWarning = complexity > 10 ? " [HIGH_COMPLEXITY]" : "";
                    sb.append(String.format("  - %s (Complexity: %d)%s (Lines: %d-%d)\n",
                            f.get("name").asText(),
                            complexity,
                            complexityWarning,
                            f.get("start_line").asInt(),
                            f.get("end_line").asInt()));
                });
            }
            sb.append("--- END ANALYSIS ---\n");

            return sb.toString();

        } catch (Exception e) {
            LogService.error("CodeStructureService", "Analysis failed for " + filePath + ": " + e.getMessage());
            return "// Structure Analysis Unavailable (Exception)";
        }
    }

    private JsonNode runAnalysisEngine(String targetPath) throws IOException, InterruptedException {
        File scriptFile = new File(SCRIPT_PATH);
        if (!scriptFile.exists()) {
            // Try absolute path if CWD is different
            scriptFile = new File(System.getProperty("user.dir"), SCRIPT_PATH);
        }

        ProcessBuilder pb = new ProcessBuilder("python", scriptFile.getAbsolutePath(), targetPath);
        pb.redirectErrorStream(true);

        Process process = pb.start();

        // Read output
        StringBuilder output = new StringBuilder();
        try (BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()))) {
            String line;
            while ((line = reader.readLine()) != null) {
                output.append(line);
            }
        }

        if (!process.waitFor(5, TimeUnit.SECONDS)) {
            process.destroy();
            throw new IOException("Analysis engine timed out.");
        }

        if (process.exitValue() != 0) {
            LogService.warn("CodeStructureService", "Analysis Engine Error: " + output.toString());
            return null;
        }

        try {
            return objectMapper.readTree(output.toString());
        } catch (Exception e) {
            LogService.warn("CodeStructureService", "Invalid JSON from engine: " + output.toString());
            return null;
        }
    }
}
