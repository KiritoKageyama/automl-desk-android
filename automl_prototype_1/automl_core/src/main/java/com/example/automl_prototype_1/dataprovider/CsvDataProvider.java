package com.example.automl_prototype_1.dataprovider;

import com.example.automl_prototype_1.model.Dataset;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVRecord;

import java.io.IOException;
import java.io.Reader;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.HashMap;

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

        System.out.println("CsvDataProvider: Loading dataset from Reader (Truly Efficient Mode)...");
        List<double[]> tempRows = new ArrayList<>();
        List<String> headers = null;
        // boolean hasHeaders = false; // Removed unused variable

        // Build format with robust settings
        CSVFormat format = CSVFormat.DEFAULT.builder()
                .setTrim(true)
                .setIgnoreEmptyLines(true)
                .build();

        // Use Iterator to stream records instead of loading all into memory
        try (CSVParser parser = format.parse(dataReader)) {
            Iterator<CSVRecord> iterator = parser.iterator();

            if (!iterator.hasNext()) {
                throw new IOException("CSV file is empty.");
            }

            // 1. Analyze First Record
            CSVRecord firstRecord = iterator.next();
            int numCols = firstRecord.size();
            boolean firstRowIsHeader = false;

            // Heuristic A: Check for non-numeric types in first row
            boolean hasNonNumeric = false;
            for (int i = 0; i < numCols; i++) {
                String val = firstRecord.get(i).trim();
                if (val.isEmpty())
                    continue;
                try {
                    Double.parseDouble(val);
                } catch (NumberFormatException e) {
                    hasNonNumeric = true; // Found a string, likely a header
                    break;
                }
            }

            // Heuristic B: Specific domain check (kept for backward compat but guarded)
            boolean isBreastCancerId = (numCols > 1 && firstRecord.get(1).matches("[MmBb]"));

            if (hasNonNumeric && !isBreastCancerId) {
                firstRowIsHeader = true;
            }

            // 2. Setup Headers
            if (firstRowIsHeader) {
                System.out.println("CsvDataProvider: Header row detected.");
                headers = new ArrayList<>();
                for (String h : firstRecord) {
                    headers.add(h.trim());
                }
            } else {
                System.out.println("CsvDataProvider: No header row detected. Generating standard column names...");
                headers = new ArrayList<>();
                for (int i = 0; i < numCols; i++) {
                    if (i == numCols - 1) {
                        headers.add("Target_Class");
                    } else {
                        headers.add("Column_" + i);
                    }
                }
            }

            // 3. Prepare Encoders
            Map<Integer, Map<String, Double>> columnEncoders = new HashMap<>();
            Map<Integer, Double> nextCodes = new HashMap<>();

            // 4. Process Records (Streamed)
            // If the first row was NOT a header, we must process it as data
            if (!firstRowIsHeader) {
                processRecord(firstRecord, tempRows, numCols, columnEncoders, nextCodes);
            }

            // Iterate the rest
            while (iterator.hasNext()) {
                processRecord(iterator.next(), tempRows, numCols, columnEncoders, nextCodes);
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

    private void processRecord(CSVRecord record, List<double[]> rows, int expectedCols,
            Map<Integer, Map<String, Double>> encoders,
            Map<Integer, Double> nextCodes) {

        double[] rowData = new double[expectedCols];
        for (int i = 0; i < expectedCols; i++) {
            if (i < record.size()) {
                String val = record.get(i).trim();
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