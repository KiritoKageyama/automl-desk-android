package com.example.automl_prototype_1.dataprovider;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.Reader;
import java.util.ArrayList;
import java.util.List;

/**
 * A lightweight, high-performance CSV parser using Java standard I/O.
 * Designed to replace heavy dependencies like Apache Commons CSV.
 * Supports basic CSV features:
 * - Specific delimiters (default: ,)
 * - Quoted values (double quotes escape quotes)
 * - Trimming whitespace
 */
public class FastCsvParser {

    private final char delimiter;
    private final char quoteChar;

    public FastCsvParser() {
        this(',', '"');
    }

    public FastCsvParser(char delimiter, char quoteChar) {
        this.delimiter = delimiter;
        this.quoteChar = quoteChar;
    }

    /**
     * Parses a single line of CSV.
     * Handles quoted fields and escaped quotes within fields.
     */
    public List<String> parseLine(String line) {
        List<String> tokens = new ArrayList<>();
        if (line == null || line.isEmpty()) {
            return tokens;
        }

        StringBuilder sb = new StringBuilder();
        boolean inQuotes = false;

        for (int i = 0; i < line.length(); i++) {
            char c = line.charAt(i);

            if (c == quoteChar) {
                // Check for escaped quote ("")
                if (inQuotes && i + 1 < line.length() && line.charAt(i + 1) == quoteChar) {
                    sb.append(quoteChar);
                    i++; // Skip the next quote
                } else {
                    inQuotes = !inQuotes; // Toggle quote state
                }
            } else if (c == delimiter && !inQuotes) {
                tokens.add(sb.toString().trim());
                sb.setLength(0); // Reset buffer
            } else {
                sb.append(c);
            }
        }

        tokens.add(sb.toString().trim()); // Add last token
        return tokens;
    }

    /**
     * Reads all lines from a Reader and parses them.
     */
    public List<List<String>> parse(Reader reader) throws IOException {
        List<List<String>> rows = new ArrayList<>();
        try (BufferedReader br = (reader instanceof BufferedReader) ? (BufferedReader) reader
                : new BufferedReader(reader)) {
            String line;
            while ((line = br.readLine()) != null) {
                if (line.trim().isEmpty())
                    continue; // Skip empty lines
                rows.add(parseLine(line));
            }
        }
        return rows;
    }
}
