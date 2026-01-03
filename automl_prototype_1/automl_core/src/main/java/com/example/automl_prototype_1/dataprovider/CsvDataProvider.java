package com.example.automl_prototype_1.dataprovider;

import com.example.automl_prototype_1.model.Dataset;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVRecord;

import java.io.IOException;
import java.io.Reader;
import java.util.ArrayList;
import java.util.List;

/**
 * Provides functionality to load datasets from CSV sources efficiently.
 * Loads data directly into primitive arrays to minimize memory usage.
 */
public class CsvDataProvider {

    /**
     * Loads a Dataset from the given Reader.
     * Assumes the CSV data has headers in the first row.
     *
     * @param dataReader The Reader providing the CSV data.
     * @return A Dataset object containing the parsed data.
     * @throws IOException If reading or parsing fails.
     */
    public Dataset loadDataset(Reader dataReader) throws IOException {
        if (dataReader == null) {
            throw new IOException("Invalid Reader provided (null). Cannot load dataset.");
        }

        System.out.println("CsvDataProvider: Loading dataset from Reader (Efficient Mode)...");
        List<double[]> tempRows = new ArrayList<>();
        List<String> headers;
        boolean hasHeaders = false;

        // First pass: Read first line to detect if it contains headers or data
        CSVFormat detectFormat = CSVFormat.DEFAULT.builder()
                .setTrim(true)
                .setIgnoreEmptyLines(true)
                .build();

        try (CSVParser detectParser = detectFormat.parse(dataReader)) {
            List<CSVRecord> records = detectParser.getRecords();
            if (records.isEmpty()) {
                throw new IOException("CSV file is empty.");
            }

            CSVRecord firstRecord = records.get(0);
            int numCols = firstRecord.size();

            // Check if first row is all numeric (means no headers)
            boolean allNumeric = true;
            for (int i = 0; i < firstRecord.size(); i++) {
                String val = firstRecord.get(i).trim();
                if (!val.isEmpty()) {
                    try {
                        Double.parseDouble(val);
                    } catch (NumberFormatException e) {
                        // Non-numeric found - could be header or categorical value
                        // If it's in position 0 or 1, might be ID or Class label
                        if (i == 0 || i == 1) {
                            // Could be data (ID or categorical class)
                            continue;
                        } else {
                            // Non-numeric in feature columns suggests headers
                            allNumeric = false;
                            break;
                        }
                    }
                }
            }

            // If all numeric (except maybe first 2 columns), assume no headers
            if (allNumeric || firstRecord.get(1).matches("[MmBb]")) {
                // No headers - generate standard names
                System.out.println("CsvDataProvider: No header row detected. Generating standard column names...");
                hasHeaders = false;
                headers = new ArrayList<>();
                for (int i = 0; i < numCols; i++) {
                    // Heuristic: If it's the last column, call it "Target_Class" to be helpful,
                    // otherwise Column_X
                    if (i == numCols - 1) {
                        headers.add("Target_Class");
                    } else {
                        headers.add("Column_" + i);
                    }
                }
                System.out.println("CsvDataProvider: Generated headers: " + headers);
            } else {
                // Has headers - use them
                System.out.println("CsvDataProvider: Header row detected.");
                hasHeaders = true;
                headers = new ArrayList<>();
                for (int i = 0; i < firstRecord.size(); i++) {
                    headers.add(firstRecord.get(i).trim());
                }
                System.out.println("CsvDataProvider: Headers found: " + headers);
            }

            int numCols2 = headers.size();

            // Maps column index to a map of (String value -> Double code)
            java.util.Map<Integer, java.util.Map<String, Double>> columnEncoders = new java.util.HashMap<>();
            // Maps column index to the next available code
            java.util.Map<Integer, Double> nextCodes = new java.util.HashMap<>();

            // Process all records (skip first if it was headers)
            int startIdx = hasHeaders ? 1 : 0;
            for (int recordIdx = startIdx; recordIdx < records.size(); recordIdx++) {
                CSVRecord csvRecord = records.get(recordIdx);
                double[] rowData = new double[numCols2];
                for (int i = 0; i < numCols2; i++) {
                    // Handle potential missing columns in a row
                    if (i < csvRecord.size()) {
                        String val = csvRecord.get(i).trim();
                        try {
                            rowData[i] = Double.parseDouble(val);
                        } catch (NumberFormatException e) {
                            // It's a string (categorical). Encode it.
                            columnEncoders.putIfAbsent(i, new java.util.HashMap<>());
                            java.util.Map<String, Double> encoder = columnEncoders.get(i);

                            if (!encoder.containsKey(val)) {
                                double code = nextCodes.getOrDefault(i, 0.0);
                                encoder.put(val, code);
                                nextCodes.put(i, code + 1.0);
                            }
                            rowData[i] = encoder.get(val);
                        }
                    } else {
                        rowData[i] = 0.0; // Missing value
                    }
                }
                tempRows.add(rowData);
            }
        }

        if (tempRows.isEmpty() && !headers.isEmpty()) {
            System.out.println("CsvDataProvider: Warning - Loaded headers but no data records found.");
        } else {
            System.out.println(
                    "CsvDataProvider: Loaded " + tempRows.size() + " records with " + headers.size() + " features.");
        }

        // Convert List<double[]> to double[][]
        double[][] features = tempRows.toArray(new double[0][]);

        // Create Dataset with all data in 'features', labels null, target null
        // The target will be separated later using dataset.getFeatureMatrix(targetName)
        return new Dataset(null, headers, features, null, null);
    }
}