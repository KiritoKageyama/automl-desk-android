package com.example.automl_prototype_1.service;

import com.example.automl_prototype_1.model.Dataset;
import com.example.automl_prototype_1.service.agentic.LocalLlmService;
import java.net.HttpURLConnection;
import java.net.URL;
import java.net.URLEncoder;
import java.nio.charset.StandardCharsets;
import java.util.stream.Collectors;

/**
 * Intelligent SOTA Lookup Service.
 * Uses Local LLM to identify datasets from content (headers/rows) and
 * determines their
 * State-of-the-Art benchmarks.
 * 
 * Logic:
 * 1. Ask LLM to identify the dataset (Public vs Custom).
 * 2. If Public: Search PapersWithCode API for exact numbers.
 * 3. If Custom or API Fail: Ask LLM for a theoretical baseline estimate.
 */
public class SotaLookupService {

    private static final String API_URL = "https://paperswithcode.com/api/v1/datasets/?q=";
    private final com.fasterxml.jackson.databind.ObjectMapper objectMapper;
    private final LocalLlmService llmService;

    public SotaLookupService() {
        this.objectMapper = new com.fasterxml.jackson.databind.ObjectMapper();
        this.llmService = new LocalLlmService();
    }

    /**
     * Intelligent SOTA search.
     * Analyzes the legitimate dataset content to find benchmarks.
     */
    public String searchSota(Dataset dataset) {
        if (!llmService.isAvailable()) {
            return "AI Service Unavailable. Cannot analyze SOTA.";
        }

        // 1. Context Extraction
        String headers = String.join(", ", dataset.getHeaders().subList(0, Math.min(dataset.getHeaders().size(), 10)));
        int rows = dataset.getRecordCount();
        int cols = dataset.getFeatureCount();

        // 2. LLM Identification (The "Brain")
        String identificationPrompt = String.format(
                "Analyze this dataset metadata:\n" +
                        "- Dimensions: %d rows x %d columns\n" +
                        "- Headers: [%s]\n" +
                        "Question: Is this a well-known public machine learning dataset (e.g., Iris, Titanic, WDBC, Wine, MNIST)? "
                        +
                        "If yes, output ONLY the canonical name (e.g. 'Iris Dataset'). " +
                        "If it looks like a custom/private dataset, output 'CUSTOM'.",
                rows, cols, headers);

        String predictedName = llmService.queryOllama(identificationPrompt, "active");
        if (predictedName == null)
            predictedName = "CUSTOM";
        predictedName = predictedName.trim().replace("\"", "").replace("'", "");

        LogService.info("SotaLookup", "AI Identified Dataset as: " + predictedName);

        // 3. Branching Logic
        if (predictedName.equalsIgnoreCase("CUSTOM") || predictedName.contains("CUSTOM")) {
            return handleCustomDataset(dataset);
        } else {
            return handlePublicDataset(predictedName);
        }
    }

    private String handlePublicDataset(String canonicalName) {
        // Try Internet Search First (The "Truth")
        try {
            String apiResult = searchLivePapersWithCode(canonicalName);
            if (apiResult != null) {
                return "Benchmarks for " + canonicalName + ": " + apiResult + " (Source: PapersWithCode)";
            }
        } catch (Exception e) {
            LogService.warn("SotaLookup", "API Search failed for " + canonicalName);
        }

        // Fallback to LLM Knowledge
        String fallbackPrompt = "What is the generally accepted State-of-the-Art (SOTA) test accuracy for the '"
                + canonicalName + "' dataset? Give a single short sentence with the percentage.";
        return llmService.queryOllama(fallbackPrompt, "active") + " (Source: LLM Knowledge)";
    }

    private String handleCustomDataset(Dataset dataset) {
        // For custom data, we can't look up a "result". We must estimate difficulty.
        int rows = dataset.getRecordCount();
        int cols = dataset.getFeatureCount();

        String prompt = String.format(
                "I have a custom dataset with %d rows and %d features. The headers are: %s. " +
                        "Based on general ML principles, what is a \"good\" target accuracy for a dataset of this size and domain? "
                        +
                        "Is it likely noisy/sparse? Give a realistic baseline estimate (e.g. 'Expect 85-90%'). Be concise.",
                rows, cols, dataset.getHeaders().stream().limit(5).collect(Collectors.joining(", ")));

        return "Custom Dataset: " + llmService.queryOllama(prompt, "active");
    }

    private String searchLivePapersWithCode(String query) throws Exception {
        String encodedQuery = URLEncoder.encode(query, StandardCharsets.UTF_8.toString());
        URL url = java.net.URI.create(API_URL + encodedQuery).toURL();
        HttpURLConnection connection = (HttpURLConnection) url.openConnection();
        connection.setRequestMethod("GET");
        connection.setRequestProperty("Accept", "application/json");
        connection.setRequestProperty("User-Agent", "AutoML-Bot/1.0");
        connection.setConnectTimeout(5000);
        connection.setReadTimeout(5000);

        if (connection.getResponseCode() == 200) {
            try (java.util.Scanner s = new java.util.Scanner(connection.getInputStream()).useDelimiter("\\A")) {
                String response = s.hasNext() ? s.next() : "";

                com.fasterxml.jackson.databind.JsonNode root = objectMapper.readTree(response);
                com.fasterxml.jackson.databind.JsonNode results = root.path("results");

                if (results.isArray() && results.size() > 0) {
                    // Return the description of the first match
                    return "Found Record: " + results.get(0).path("name").asText();
                }
            }
        }
        return null;
    }

    // Maintain backward compatibility for older callers (though they should
    // upgrade)
    public String searchSota(String datasetName, int r, int c) {
        return "Legacy SOTA Lookup Deprecated. Please use the AI-enabled searchSota(Dataset) method.";
    }
}
