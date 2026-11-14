package com.example.automl_prototype_1.service;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;

import java.io.*;
import java.net.HttpURLConnection;
import java.net.URL;
import java.util.Properties;

/**
 * Service for interacting with Anthropic Claude API.
 * Provides high-precision reasoning and deep dataset analysis.
 */
public class AnthropicService {

    private static final String API_URL = "https://api.anthropic.com/v1/messages";
    private static final String ANTHROPIC_VERSION = "2023-06-01";
    private static final String DEFAULT_MODEL = "claude-3-7-sonnet-20250219";

    private String apiKey;
    private final ObjectMapper objectMapper = new ObjectMapper();
    private static AnthropicService instance;

    public static synchronized AnthropicService getInstance() {
        if (instance == null) {
            instance = new AnthropicService();
        }
        return instance;
    }

    private AnthropicService() {
        loadApiKey();
    }

    private void loadApiKey() {
        // 1. Try System Environment
        this.apiKey = System.getenv("ANTHROPIC_API_KEY");

        // 2. Try .env file in project root
        if (this.apiKey == null || this.apiKey.isEmpty()) {
            try {
                // Search up to 2 levels up for .env if not in current dir
                File envFile = new File(".env");
                if (!envFile.exists()) {
                    envFile = new File("../.env");
                }
                if (!envFile.exists()) {
                    envFile = new File("../../.env");
                }

                if (envFile.exists()) {
                    Properties props = new Properties();
                    try (FileInputStream fis = new FileInputStream(envFile)) {
                        props.load(fis);
                        this.apiKey = props.getProperty("ANTHROPIC_API_KEY");
                    }
                }
            } catch (IOException e) {
                LogService.warn("AnthropicService", "Failed to load .env file: " + e.getMessage());
            }
        }

        if (this.apiKey == null || this.apiKey.isEmpty()) {
            LogService.warn("AnthropicService", "No Anthropic API Key found in env or .env file.");
        } else {
             LogService.info("AnthropicService", "Anthropic API Key loaded successfully.");
        }
    }

    public boolean isAvailable() {
        return apiKey != null && !apiKey.isEmpty();
    }

    /**
     * Queries Claude 3.7 for high-precision tasks.
     * @param prompt The message to send to Claude.
     * @return The response text or null if failed.
     */
    public String queryClaude(String prompt) {
        if (!isAvailable()) {
            LogService.warn("AnthropicService", "Claude query skipped: API target key unavailable.");
            return null;
        }

        try {
            URL url = java.net.URI.create(API_URL).toURL();
            HttpURLConnection conn = (HttpURLConnection) url.openConnection();
            conn.setRequestMethod("POST");
            conn.setRequestProperty("x-api-key", apiKey);
            conn.setRequestProperty("anthropic-version", ANTHROPIC_VERSION);
            conn.setRequestProperty("Content-Type", "application/json");
            conn.setDoOutput(true);
            conn.setConnectTimeout(10000);
            conn.setReadTimeout(60000);

            ObjectNode rootNode = objectMapper.createObjectNode();
            rootNode.put("model", DEFAULT_MODEL);
            rootNode.put("max_tokens", 4096);

            ArrayNode messages = rootNode.putArray("messages");
            ObjectNode userMessage = messages.addObject();
            userMessage.put("role", "user");
            userMessage.put("content", prompt);

            try (OutputStream os = conn.getOutputStream()) {
                byte[] input = objectMapper.writeValueAsBytes(rootNode);
                os.write(input, 0, input.length);
            }

            int status = conn.getResponseCode();
            if (status != 200) {
                try (BufferedReader br = new BufferedReader(new InputStreamReader(conn.getErrorStream()))) {
                    StringBuilder error = new StringBuilder();
                    String line;
                    while ((line = br.readLine()) != null) error.append(line);
                    LogService.warn("AnthropicService", "Claude API Error (" + status + "): " + error.toString());
                }
                return null;
            }

            JsonNode responseNode = objectMapper.readTree(conn.getInputStream());
            if (responseNode.has("content") && responseNode.get("content").isArray()) {
                JsonNode firstContent = responseNode.get("content").get(0);
                if (firstContent.has("text")) {
                    return firstContent.get("text").asText().trim();
                }
            }
            return null;

        } catch (Exception e) {
            LogService.warn("AnthropicService", "Claude Query Failed: " + e.getMessage());
            return null;
        }
    }
}
