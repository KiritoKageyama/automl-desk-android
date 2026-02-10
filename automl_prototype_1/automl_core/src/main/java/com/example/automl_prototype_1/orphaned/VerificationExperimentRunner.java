package com.example.automl_prototype_1.experiment;

import com.example.automl_prototype_1.algorithm.GeneticAlgorithm;
import com.example.automl_prototype_1.classifier.RandomForestClassifier;
import com.example.automl_prototype_1.dataprovider.CsvDataProvider;
import com.example.automl_prototype_1.model.Dataset;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.StringReader;
import java.net.URL;
import java.util.*;

public class VerificationExperimentRunner {

    public static void main(String[] args) {
        System.out.println("Starting Verification Experiments on Real-World Datasets...");

        // Define datasets
        List<ExperimentConfig> configs = new ArrayList<>();

        // Iris: 4 features, class at end
        // Rename 'class' to 'Class' for compatibility
        configs.add(new ExperimentConfig(
                "Iris",
                "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",
                "sepal_length,sepal_width,petal_length,petal_width,Class",
                "Class",
                Collections.emptyList()));

        // Wine: Class at start, 13 features
        configs.add(new ExperimentConfig(
                "Wine",
                "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data",
                "Class,alcohol,malic_acid,ash,alcalinity_of_ash,magnesium,total_phenols,flavanoids,nonflavanoid_phenols,proanthocyanins,color_intensity,hue,od280_od315_of_diluted_wines,proline",
                "Class",
                Collections.emptyList()));

        // Sonar: 60 features, class at end. No ID.
        StringBuilder sonarHeader = new StringBuilder();
        for (int i = 1; i <= 60; i++)
            sonarHeader.append("f").append(i).append(",");
        sonarHeader.append("Class");

        configs.add(new ExperimentConfig(
                "Sonar",
                "https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data",
                sonarHeader.toString(),
                "Class",
                Collections.emptyList()));

        // Breast Cancer WDBC: ID, Diagnosis, 30 features
        // We must IGNORE 'id'
        StringBuilder wdbcHeader = new StringBuilder("id,Class,");
        for (int i = 1; i <= 30; i++)
            wdbcHeader.append("f").append(i).append(",");
        wdbcHeader.setLength(wdbcHeader.length() - 1); // remove last comma

        configs.add(new ExperimentConfig(
                "Breast Cancer",
                "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data",
                wdbcHeader.toString(),
                "Class",
                Collections.singletonList("id")));

        CsvDataProvider dataProvider = new CsvDataProvider();

        // Enable Balanced Optimization (Accuracy + F1)
        com.example.automl_prototype_1.service.AppStateService.getInstance().setEnableBalancedOptimization(true);
        com.example.automl_prototype_1.service.AppStateService.getInstance().setUseGpu(true); // Ensure GPU is tried if
                                                                                              // available

        for (ExperimentConfig config : configs) {
            System.out.println("\n--------------------------------------------------");
            System.out.println("Processing Dataset: " + config.name);
            try {
                // 1. Fetch and Prepare Data
                String csvContent = fetchAndPrepareCsv(config);

                // 2. Load Raw Dataset
                Dataset raw = dataProvider.loadDataset(new StringReader(csvContent));

                // 3. Clean Dataset (Remove Ignored Cols, Keep Class in Features)
                Dataset cleaned = cleanDataset(raw, config);

                System.out.println("Data Loaded: " + cleaned.getRecordCount() + " instances, "
                        + cleaned.getFeatureCount() + " features (including Class).");

                // 4. Run Genetic Algorithm (Proposed)
                // Using 50 pop, 10 iter for speed in verification
                GeneticAlgorithm ga = new GeneticAlgorithm(50, 10);

                Map<String, Object> configMap = new HashMap<>();
                configMap.put("classifier", new RandomForestClassifier());

                com.example.automl_prototype_1.model.ExecutionResult result = ga.execute(cleaned, configMap);

                // 5. Evaluate Final Performance
                System.out.println("Result for " + config.name + ":");
                System.out.printf("Accuracy: %.4f%n", result.getAccuracy());
                System.out.printf("F1-Score: %.4f%n", result.getF1Score());
                // Note: getOriginalFeatures returns 0 if constructor didn't set it (which
                // AbstractFeatureSelectionAlgorithm might not propagate optimally unless
                // specifically handled)
                // Actually AbstractFeatureSelectionAlgorithm passes numFeatures to
                // ExecutionResult.
                System.out.println(
                        "Selected Features: " + result.getSelectedFeatures() + " / " + result.getOriginalFeatures());

            } catch (Exception e) {
                System.err.println("Failed to process " + config.name + ": " + e.getMessage());
                com.example.automl_prototype_1.service.LogService.error("VerificationRunner",
                        "Failed to process " + config.name + ": " + e.getMessage());
            }
        }
    }

    private static String fetchAndPrepareCsv(ExperimentConfig config) throws IOException {
        System.out.println("Fetching URL: " + config.url);
        URL url = java.net.URI.create(config.url).toURL();
        StringBuilder content = new StringBuilder();

        // Add Header
        content.append(config.header).append("\n");

        try (BufferedReader reader = new BufferedReader(new InputStreamReader(url.openStream()))) {
            String line;
            while ((line = reader.readLine()) != null) {
                if (line.trim().isEmpty())
                    continue;
                content.append(line).append("\n");
            }
        }
        return content.toString();
    }

    private static Dataset cleanDataset(Dataset raw, ExperimentConfig config) {
        List<String> rawHeaders = raw.getHeaders();
        List<Integer> ignoredIndices = new ArrayList<>();
        for (String ig : config.ignoreCols) {
            int idx = rawHeaders.indexOf(ig);
            if (idx != -1)
                ignoredIndices.add(idx);
        }

        int numRows = raw.getRecordCount();
        int numRawCols = raw.getFeatureCount();
        int numKeepCols = numRawCols - ignoredIndices.size();

        double[][] X = new double[numRows][numKeepCols];
        double[][] rawMatrix = raw.getFeatureMatrix();

        List<String> newHeaders = new ArrayList<>();

        // Build headers
        for (int j = 0; j < numRawCols; j++) {
            if (ignoredIndices.contains(j))
                continue;
            newHeaders.add(rawHeaders.get(j));
        }

        for (int i = 0; i < numRows; i++) {
            int k = 0;
            for (int j = 0; j < numRawCols; j++) {
                if (!ignoredIndices.contains(j)) {
                    X[i][k++] = rawMatrix[i][j];
                }
            }
        }

        // Return Dataset with labels=null, targetColumnName=Class (so algorithm can
        // extract it)
        return new Dataset(null, newHeaders, X, null, config.targetCol);
    }

    static class ExperimentConfig {
        String name;
        String url;
        String header;
        String targetCol;
        List<String> ignoreCols;

        public ExperimentConfig(String name, String url, String header, String targetCol, List<String> ignoreCols) {
            this.name = name;
            this.url = url;
            this.header = header;
            this.targetCol = targetCol;
            this.ignoreCols = ignoreCols;
        }
    }
}
