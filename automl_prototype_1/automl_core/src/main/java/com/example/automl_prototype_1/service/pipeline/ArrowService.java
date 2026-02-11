package com.example.automl_prototype_1.service.pipeline;

import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.Float8Vector;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.types.FloatingPointPrecision;
import org.apache.arrow.vector.types.pojo.ArrowType;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.arrow.vector.ipc.ArrowStreamWriter;
import org.apache.arrow.vector.types.pojo.FieldType;
import org.apache.arrow.vector.types.pojo.Schema;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
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
}
