package com.example.automl_prototype_1.service;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import javafx.beans.property.IntegerProperty;
import javafx.beans.property.SimpleIntegerProperty;

/**
 * Service to manage real-time data streams.
 * Buffer incoming data, notify listeners, and save to CSV.
 */
public class RealTimeDataService {

    private static RealTimeDataService instance;
    private final List<double[]> dataBuffer = new ArrayList<>();
    private final IntegerProperty bufferSizeProperty = new SimpleIntegerProperty(0);
    private int expectedFeatureCount = -1;

    private RealTimeDataService() {
        LogService.info("RealTimeService", "Initialized.");
    }

    public static synchronized RealTimeDataService getInstance() {
        if (instance == null) {
            instance = new RealTimeDataService();
        }
        return instance;
    }

    /**
     * Push a new data point to the stream.
     * 
     * @param features Array of feature values.
     */
    public synchronized void pushDataPoint(double[] features) {
        if (expectedFeatureCount == -1) {
            expectedFeatureCount = features.length;
        } else if (features.length != expectedFeatureCount) {
            LogService.error("RealTimeService",
                    "Feature count mismatch! Expected " + expectedFeatureCount + ", got " + features.length);
            return;
        }

        dataBuffer.add(features);
        bufferSizeProperty.set(dataBuffer.size());

        // Notify listeners (UI updates) - simplistic approach
        // Ideally use an EventBus or ObservableList, but property is enough for count.
    }

    public synchronized void clearBuffer() {
        dataBuffer.clear();
        bufferSizeProperty.set(0);
        expectedFeatureCount = -1;
        LogService.info("RealTimeService", "Buffer cleared.");
    }

    public synchronized boolean saveBufferToCsv(File destination) {
        if (dataBuffer.isEmpty()) {
            LogService.warn("RealTimeService", "Buffer empty, nothing to save.");
            return false;
        }

        try (BufferedWriter writer = new BufferedWriter(new FileWriter(destination))) {
            // Write Header? We don't have header names here.
            // Generate generic headers: Feat0, Feat1...
            StringBuilder header = new StringBuilder();
            for (int i = 0; i < expectedFeatureCount; i++) {
                header.append("Feat").append(i);
                if (i < expectedFeatureCount - 1)
                    header.append(",");
            }
            writer.write(header.toString());
            writer.newLine();

            // Write Data
            for (double[] row : dataBuffer) {
                StringBuilder line = new StringBuilder();
                for (int i = 0; i < row.length; i++) {
                    line.append(row[i]);
                    if (i < row.length - 1)
                        line.append(",");
                }
                writer.write(line.toString());
                writer.newLine();
            }
            LogService.info("RealTimeService", "Saved " + dataBuffer.size() + " records to " + destination.getName());
            return true;
        } catch (IOException e) {
            LogService.error("RealTimeService", "Failed to save CSV: " + e.getMessage());
            return false;
        }
    }

    public IntegerProperty bufferSizeProperty() {
        return bufferSizeProperty;
    }

    public synchronized List<double[]> getSnapshot() {
        return new ArrayList<>(dataBuffer);
    }
}
