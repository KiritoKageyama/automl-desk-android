package com.example.automl_prototype_1.service.agentic;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.net.HttpURLConnection;
import java.net.URL;
import java.net.URLEncoder;
import java.nio.charset.StandardCharsets;
import java.util.HashMap;
import java.util.Map;

/**
 * Simulates an external search service to find State-of-the-Art (SOTA) results
 * for known datasets.
 * In a real-world scenario, this would query an external API (e.g., Papers With
 * Code, Google Search).
 */
public class SotaLookupService {

    private final Map<String, String> sotaDatabase;
    private static final String API_URL = "https://paperswithcode.com/api/v1/datasets/?q=";
    private final com.fasterxml.jackson.databind.ObjectMapper objectMapper;

    public SotaLookupService() {
        sotaDatabase = new HashMap<>();
        // Seed with some common datasets often used in testing (Fallback)
        sotaDatabase.put("wdbc", "99.0% (Neural Networks / Ensembles)");
        sotaDatabase.put("breast_cancer", "99.0% (Neural Networks / Ensembles)");
        sotaDatabase.put("iris", "100% (Various Algorithms)");
        sotaDatabase.put("wine", "98.0% (Regularized Discriminant Analysis)");
        sotaDatabase.put("diabetes", "77.0% (Ensemble Classifiers)");
        sotaDatabase.put("sonar", "90.0% (Random Forest / SVM)");
        sotaDatabase.put("glass", "75.0% (Random Forest / XGBoost)");
        sotaDatabase.put("ionosphere", "94.0% (Backpropagation / SVM)");

        this.objectMapper = new com.fasterxml.jackson.databind.ObjectMapper();
    }

    /**
     * Searches for SOTA results using filename or dataset dimensions.
     */
    public String searchSota(String datasetName, int rowCount, int featureCount) {
        String cleanName = (datasetName != null) ? datasetName.toLowerCase().replace(".csv", "").trim() : "";

        // 1. Try identifying by dimensions if name is generic
        if (cleanName.contains("data") || cleanName.contains("train") || cleanName.contains("test")
                || cleanName.length() < 3) {
            String inferredName = fingerprintDataset(rowCount, featureCount);
            if (inferredName != null) {
                System.out.println("SotaLookupService: Inferred dataset '" + inferredName + "' from dimensions ("
                        + rowCount + "x" + featureCount + ")");
                cleanName = inferredName;
            }
        }

        // 2. Try Live Search
        if (!cleanName.equals("data") && !cleanName.equals("dataset") && cleanName.length() > 3) {
            try {
                System.out.println("SotaLookupService: Live searching for query: '" + cleanName + "'");
                String liveResult = searchLive(cleanName);
                if (liveResult != null) {
                    return liveResult + " (Source: Papers With Code)";
                }
            } catch (Exception e) {
                System.err.println("Live SOTA search failed for '" + cleanName + "': " + e.getMessage());
            }
        } else {
            System.out.println("SotaLookupService: Skipping live search for generic query: '" + cleanName + "'");
        }

        // 3. Fallback to Internal Database
        for (Map.Entry<String, String> entry : sotaDatabase.entrySet()) {
            if (cleanName.contains(entry.getKey())) {
                return entry.getValue() + " (Source: Internal Knowledge Base)";
            }
        }

        return "No specific SOTA record found.";
    }

    public String searchSota(String datasetName) {
        return searchSota(datasetName, -1, -1);
    }

    private String fingerprintDataset(int rows, int cols) {
        // Known Dimensions (Allowing slight variance in columns for ID/Class/Trailing)
        if (rows == 150 && cols >= 4 && cols <= 6)
            return "iris";
        if (rows == 178 && cols >= 13 && cols <= 15)
            return "wine";
        if (rows == 569 && cols >= 30 && cols <= 35)
            return "wdbc"; // Breast Cancer WDBC
        if (rows == 208 && cols >= 60 && cols <= 62)
            return "sonar";
        if (rows == 214 && cols >= 9 && cols <= 11)
            return "glass";
        if (rows == 768 && cols >= 8 && cols <= 10)
            return "diabetes";
        return null; // Unknown dimensions
    }

    private String searchLive(String query) throws Exception {
        String encodedQuery = URLEncoder.encode(query, StandardCharsets.UTF_8.toString());
        URL url = java.net.URI.create(API_URL + encodedQuery).toURL();
        HttpURLConnection connection = (HttpURLConnection) url.openConnection();
        connection.setRequestMethod("GET");
        connection.setRequestProperty("Accept", "application/json");
        connection.setRequestProperty("User-Agent",
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36");
        connection.setConnectTimeout(10000); // 10 seconds
        connection.setReadTimeout(10000);

        int responseCode = connection.getResponseCode();
        if (responseCode == 200) {
            StringBuilder response = new StringBuilder();
            try (BufferedReader in = new BufferedReader(new InputStreamReader(connection.getInputStream()))) {
                String inputLine;
                while ((inputLine = in.readLine()) != null) {
                    response.append(inputLine);
                }
            }

            com.fasterxml.jackson.databind.JsonNode root = objectMapper.readTree(response.toString());
            com.fasterxml.jackson.databind.JsonNode results = root.path("results");

            if (results.isArray() && results.size() > 0) {
                // Get the first result
                com.fasterxml.jackson.databind.JsonNode firstMatch = results.get(0);
                String name = firstMatch.path("name").asText();
                return "Found '" + name + "' on PapersWithCode. Check online for exact benchmarks.";
            }
        }
        return null;
    }
}
