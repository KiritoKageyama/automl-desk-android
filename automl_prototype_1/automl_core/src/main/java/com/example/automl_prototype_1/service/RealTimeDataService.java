package com.example.automl_prototype_1.service;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ConcurrentLinkedQueue;

/**
 * Service for handling real-time data streams.
 * Acts as a central hub for incoming data points (e.g., from sensors, API, or
 * simulation).
 */
public class RealTimeDataService {
    private static final RealTimeDataService instance = new RealTimeDataService();
    private final ConcurrentLinkedQueue<DataPoint> buffer = new ConcurrentLinkedQueue<>();
    private final List<DataListener> listeners = new ArrayList<>();

    private RealTimeDataService() {
    }

    public static RealTimeDataService getInstance() {
        return instance;
    }

    public record DataPoint(long timestamp, double[] values) {
    }

    public interface DataListener {
        void onData(DataPoint point);
    }

    public void addListener(DataListener listener) {
        synchronized (listeners) {
            listeners.add(listener);
        }
    }

    public void pushData(double[] values) {
        DataPoint point = new DataPoint(System.currentTimeMillis(), values);
        buffer.add(point);

        // Notify listeners
        synchronized (listeners) {
            for (DataListener listener : listeners) {
                listener.onData(point);
            }
        }

        // Basic buffer management (keep last 10000 points)
        if (buffer.size() > 10000) {
            buffer.poll();
        }
    }

    public List<DataPoint> getLatest(int n) {
        List<DataPoint> list = new ArrayList<>(buffer);
        int start = Math.max(0, list.size() - n);
        return list.subList(start, list.size());
    }

    public void clear() {
        buffer.clear();
    }

    // --- Compatibility Methods for DataVisualizationViewController ---

    public javafx.beans.property.IntegerProperty bufferSizeProperty() {
        // Simple wrapper, updates might be manually triggered or bound to listener
        return new javafx.beans.property.SimpleIntegerProperty(buffer.size());
    }

    public List<DataPoint> getSnapshot() {
        return new ArrayList<>(buffer);
    }

    public void saveBufferToCsv(java.io.File file) {
        try (java.io.PrintWriter writer = new java.io.PrintWriter(file)) {
            writer.println("Timestamp,Value");
            for (DataPoint p : buffer) {
                StringBuilder sb = new StringBuilder();
                sb.append(p.timestamp());
                for (double v : p.values()) {
                    sb.append(",").append(v);
                }
                writer.println(sb.toString());
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public void clearBuffer() {
        clear();
    }

    public void pushDataPoint(double[] values) {
        pushData(values);
    }
}
