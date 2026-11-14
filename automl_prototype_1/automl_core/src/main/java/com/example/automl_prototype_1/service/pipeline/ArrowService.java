package com.example.automl_prototype_1.service.pipeline;

import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.Float8Vector;
import org.apache.arrow.vector.VarCharVector;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.types.FloatingPointPrecision;
import org.apache.arrow.vector.types.pojo.ArrowType;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.arrow.vector.ipc.ArrowStreamWriter;
import org.apache.arrow.vector.ipc.ArrowFileWriter;
import org.apache.arrow.vector.types.pojo.FieldType;
import org.apache.arrow.vector.types.pojo.Schema;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.UUID;
import java.nio.channels.Channels;

import com.example.automl_prototype_1.service.LogService;

/**
 * Service to serialize training data into Apache Arrow IPC Streams.
 * This enables zero-copy (or at least valid in-memory) transfer to the Python
 * backend, avoiding slow CSV I/O.
 */
public class ArrowService {

    private static ArrowService instance;
    private final BufferAllocator allocator;

    private ArrowService() {
        this.allocator = new RootAllocator(Long.MAX_VALUE);
    }

    public static synchronized ArrowService getInstance() {
        if (instance == null) {
            instance = new ArrowService();
        }
        return instance;
    }

    /**
     * Serializes features and labels into an Arrow IPC Stream (Batch).
     *
     * @param features 2D array of features [rows][cols]
     * @param labels   Array of labels
     * @return Byte array containing the Arrow IPC stream
     */
    public byte[] serializeTrainData(double[][] features, int[] labels) throws IOException {
        if (features == null || features.length == 0) {
            throw new IllegalArgumentException("Features cannot be empty");
        }

        int rowCount = features.length;
        int colCount = features[0].length;

        // 1. Define Schema
        List<Field> fields = new ArrayList<>();
        // Features f0...fn
        for (int i = 0; i < colCount; i++) {
            fields.add(new Field("f" + i,
                    FieldType.nullable(new ArrowType.FloatingPoint(FloatingPointPrecision.DOUBLE)), null));
        }
        // Target (Labels) - Convert int to double for uniformity in this prototype,
        // or usage of Int vector. Server expects "target".
        // Let's use Double for "target" to support regression in future without
        // changing schema logic much,
        // or Int if we want strictness. The Python server reads "target" and converts
        // to numpy.
        // Let's use Float8 (Double) to be safe with NaN support etc, though Int is fine
        // for classification.
        // Actually, let's use Float8 to match the Python server's generic "float"
        // assumption if simplest,
        // but `int[] labels` suggests classification.
        // Let's stick to Float8 for simplicity of the vector loop.
        fields.add(new Field("target", FieldType.nullable(new ArrowType.FloatingPoint(FloatingPointPrecision.DOUBLE)),
                null));

        Schema schema = new Schema(fields);

        // 2. Create VectorSchemaRoot
        try (VectorSchemaRoot root = VectorSchemaRoot.create(schema, allocator)) {
            root.allocateNew();

            // 3. Populate Vectors
            for (int i = 0; i < colCount; i++) {
                Float8Vector vector = (Float8Vector) root.getVector("f" + i);
                for (int row = 0; row < rowCount; row++) {
                    vector.setSafe(row, features[row][i]);
                }
                vector.setValueCount(rowCount);
            }

            Float8Vector targetVector = (Float8Vector) root.getVector("target");
            for (int row = 0; row < rowCount; row++) {
                targetVector.setSafe(row, (double) labels[row]);
            }
            targetVector.setValueCount(rowCount);

            root.setRowCount(rowCount);

            // 4. Write to Stream
            try (ByteArrayOutputStream out = new ByteArrayOutputStream();
                    ArrowStreamWriter writer = new ArrowStreamWriter(root, null, Channels.newChannel(out))) {
                writer.start();
                writer.writeBatch();
                writer.end();
                return out.toByteArray();
            }
        } catch (Exception e) {
            LogService.error("ArrowService", "Serialization failed: " + e.getMessage());
            throw new IOException(e);
        }
    }

    /**
     * Writes tabular training data to a memory-mapped Arrow IPC file.
     * The OS keeps the file in Page Cache (RAM), so Python can read it
     * via pa.memory_map() without any data copying.
     *
     * @param features 2D array of features [rows][cols]
     * @param labels   Array of labels
     * @return Absolute path to the Arrow IPC file (for Python to mmap)
     */
    public String writeToMmap(double[][] features, int[] labels) throws IOException {
        if (features == null || features.length == 0) {
            throw new IllegalArgumentException("Features cannot be empty for MMAP write");
        }

        int rowCount = features.length;
        int colCount = features[0].length;

        // 1. Define Schema (same as serializeTrainData)
        List<Field> fields = new ArrayList<>();
        for (int i = 0; i < colCount; i++) {
            fields.add(new Field("f" + i,
                    FieldType.nullable(new ArrowType.FloatingPoint(FloatingPointPrecision.DOUBLE)), null));
        }
        fields.add(new Field("target",
                FieldType.nullable(new ArrowType.FloatingPoint(FloatingPointPrecision.DOUBLE)), null));

        Schema schema = new Schema(fields);

        // 2. Create temp file
        File tempDir = new File(System.getProperty("java.io.tmpdir"), "automl_mmap");
        tempDir.mkdirs();
        File mmapFile = new File(tempDir, "arrow_data_" + UUID.randomUUID() + ".arrow");
        mmapFile.deleteOnExit(); // Cleanup when JVM exits

        // 3. Write Arrow IPC to file
        try (VectorSchemaRoot root = VectorSchemaRoot.create(schema, allocator)) {
            root.allocateNew();

            // Populate feature vectors
            for (int i = 0; i < colCount; i++) {
                Float8Vector vector = (Float8Vector) root.getVector("f" + i);
                for (int row = 0; row < rowCount; row++) {
                    vector.setSafe(row, features[row][i]);
                }
                vector.setValueCount(rowCount);
            }

            // Populate target vector
            Float8Vector targetVector = (Float8Vector) root.getVector("target");
            for (int row = 0; row < rowCount; row++) {
                targetVector.setSafe(row, (double) labels[row]);
            }
            targetVector.setValueCount(rowCount);
            root.setRowCount(rowCount);

            // Write to file (ArrowFileWriter for random access compatibility)
            try (FileOutputStream fos = new FileOutputStream(mmapFile);
                    ArrowFileWriter writer = new ArrowFileWriter(root, null, fos.getChannel())) {
                writer.start();
                writer.writeBatch();
                writer.end();
            }
        } catch (Exception e) {
            LogService.error("ArrowService", "MMAP write failed: " + e.getMessage());
            throw new IOException(e);
        }

        long fileSizeMB = mmapFile.length() / (1024 * 1024);
        LogService.info("ArrowService",
                "MMAP file written: " + mmapFile.getAbsolutePath() + " (" + fileSizeMB + " MB)");

        return mmapFile.getAbsolutePath();
    }

    /**
     * Writes multimodal data (file paths or text content) to a memory-mapped
     * Arrow IPC file with a String column. Supports Image paths, audio paths,
     * and raw text for GenAI/NLP.
     *
     * @param data    List of file paths or text content strings
     * @param labels  Array of labels
     * @param colName Column name ("image_path", "text_content", etc.)
     * @return Absolute path to the Arrow IPC file
     */
    public String writeToMmap(List<String> data, int[] labels, String colName) throws IOException {
        if (data == null || data.isEmpty()) {
            throw new IllegalArgumentException("Data list cannot be empty for MMAP write");
        }

        int rowCount = data.size();

        // 1. Define Schema: one string column + target
        List<Field> fields = new ArrayList<>();
        fields.add(new Field(colName, FieldType.nullable(new ArrowType.Utf8()), null));
        fields.add(new Field("target",
                FieldType.nullable(new ArrowType.FloatingPoint(FloatingPointPrecision.DOUBLE)), null));

        Schema schema = new Schema(fields);

        // 2. Create temp file
        File tempDir = new File(System.getProperty("java.io.tmpdir"), "automl_mmap");
        tempDir.mkdirs();
        File mmapFile = new File(tempDir, "arrow_data_" + UUID.randomUUID() + ".arrow");
        mmapFile.deleteOnExit();

        // 3. Write Arrow IPC to file
        try (VectorSchemaRoot root = VectorSchemaRoot.create(schema, allocator)) {
            root.allocateNew();

            // Populate string column
            VarCharVector strVector = (VarCharVector) root.getVector(colName);
            for (int i = 0; i < rowCount; i++) {
                byte[] bytes = data.get(i).getBytes(StandardCharsets.UTF_8);
                strVector.setSafe(i, bytes, 0, bytes.length);
            }
            strVector.setValueCount(rowCount);

            // Populate target vector
            Float8Vector targetVector = (Float8Vector) root.getVector("target");
            for (int i = 0; i < rowCount; i++) {
                targetVector.setSafe(i, labels != null && i < labels.length ? (double) labels[i] : 0.0);
            }
            targetVector.setValueCount(rowCount);
            root.setRowCount(rowCount);

            // Write to file
            try (FileOutputStream fos = new FileOutputStream(mmapFile);
                    ArrowFileWriter writer = new ArrowFileWriter(root, null, fos.getChannel())) {
                writer.start();
                writer.writeBatch();
                writer.end();
            }
        } catch (Exception e) {
            LogService.error("ArrowService", "MMAP (multimodal) write failed: " + e.getMessage());
            throw new IOException(e);
        }

        LogService.info("ArrowService",
                "MMAP file written (multimodal/" + colName + "): " + mmapFile.getAbsolutePath());

        return mmapFile.getAbsolutePath();
    }
}
