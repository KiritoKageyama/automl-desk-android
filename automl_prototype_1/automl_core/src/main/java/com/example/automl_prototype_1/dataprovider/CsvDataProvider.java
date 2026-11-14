package com.example.automl_prototype_1.dataprovider;

import com.example.automl_prototype_1.model.Dataset;

import java.io.IOException;
import java.io.Reader;
import java.io.BufferedReader;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.HashMap;

/**
 * Provides functionality to load datasets from CSV sources efficiently.
 * Loads data directly into primitive arrays to minimize memory usage.
 * Now uses FastCsvParser (Zero-Dependency) for parsing.
 */
public class CsvDataProvider {

    /**
     * Loads a Dataset from the given Reader using FastCsvParser.
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

        System.out.println("CsvDataProvider: Loading dataset using FastCsvParser (NIO)...");
        List<double[]> tempRows = new ArrayList<>();
        List<String> headers = null;

        FastCsvParser parser = new FastCsvParser();

        // We will read line by line to keep memory strictly low, rather than parseAll
        try (BufferedReader br = (dataReader instanceof BufferedReader) ? (BufferedReader) dataReader
                : new BufferedReader(dataReader)) {
            String line;
            boolean isFirstRow = true;
            int numCols = 0;

            Map<Integer, Map<String, Double>> columnEncoders = new HashMap<>();
            Map<Integer, Double> nextCodes = new HashMap<>();

            while ((line = br.readLine()) != null) {
                if (line.trim().isEmpty())
                    continue;

                List<String> record = parser.parseLine(line);
                if (record.isEmpty())
                    continue;

                if (isFirstRow) {
                    numCols = record.size();

                    // Heuristic A: Check for non-numeric types in first row
                    boolean hasNonNumeric = false;
                    for (int i = 0; i < numCols; i++) {
                        String val = record.get(i);
                        if (val.isEmpty())
                            continue;
                        try {
                            Double.parseDouble(val);
                        } catch (NumberFormatException e) {
                            hasNonNumeric = true;
                            break;
                        }
                    }

                    // Heuristic B: Specific domain check
                    boolean isBreastCancerId = (numCols > 1 && record.get(1).matches("[MmBb]"));

                    if (hasNonNumeric && !isBreastCancerId) {
                        System.out.println("CsvDataProvider: Header row detected.");
                        headers = new ArrayList<>(record);
                    } else {
                        System.out.println(
                                "CsvDataProvider: No header row detected. Generating standard column names...");
                        headers = new ArrayList<>();
                        for (int i = 0; i < numCols; i++) {
                            headers.add((i == numCols - 1) ? "Target_Class" : "Column_" + i);
                        }
                        // Process first row as data since it wasn't a header
                        processRecord(record, tempRows, numCols, columnEncoders, nextCodes);
                    }
                    isFirstRow = false;
                } else {
                    processRecord(record, tempRows, numCols, columnEncoders, nextCodes);
                }
            }

            if (tempRows.isEmpty()) {
                System.out.println("CsvDataProvider: Warning - Processed CSV but found no valid data rows.");
            } else {
                System.out.println("CsvDataProvider: Loaded " + tempRows.size() + " records.");
            }
        }

        double[][] features = tempRows.toArray(new double[0][]);
        return new Dataset(null, headers, features, null, null);
    }

    private void processRecord(List<String> record, List<double[]> rows, int expectedCols,
            Map<Integer, Map<String, Double>> encoders,
            Map<Integer, Double> nextCodes) {

        double[] rowData = new double[expectedCols];
        for (int i = 0; i < expectedCols; i++) {
            if (i < record.size()) {
                String val = record.get(i);
                try {
                    if (val.isEmpty()) {
                        rowData[i] = 0.0;
                    } else {
                        rowData[i] = Double.parseDouble(val);
                    }
                } catch (NumberFormatException e) {
                    // Categorical encoding
                    encoders.putIfAbsent(i, new HashMap<>());
                    Map<String, Double> encoder = encoders.get(i);

                    if (!encoder.containsKey(val)) {
                        double code = nextCodes.getOrDefault(i, 0.0);
                        encoder.put(val, code);
                        nextCodes.put(i, code + 1.0);
                    }
                    rowData[i] = encoder.get(val);
                }
            } else {
                rowData[i] = 0.0; // Missing column padding
            }
        }
        rows.add(rowData);
    }
}