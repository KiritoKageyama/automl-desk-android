package com.example.automl_prototype_1.gpu;

import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.*;
import org.apache.arrow.vector.types.FloatingPointPrecision;
import org.apache.arrow.vector.types.pojo.ArrowType;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.arrow.vector.types.pojo.FieldType;
import org.apache.arrow.vector.types.pojo.Schema;

import java.io.ByteArrayOutputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * Apache Arrow serialization for high-performance training data transfer.
 * Phase 6 Step 4.
 */
public class ArrowTrainingDataSerializer implements ITrainingDataSerializer {

    @Override
    public SerializedData serializeTrainingRequest(double[][] features, int[] labels, Map<String, Object> params)
            throws Exception {
        if (features == null || features.length == 0) {
            throw new IllegalArgumentException("Features array cannot be empty");
        }

        int rows = features.length;
        int cols = features[0].length;

        try (RootAllocator allocator = new RootAllocator()) {
            List<Field> fields = new ArrayList<>();
            List<FieldVector> vectors = new ArrayList<>();

            // 1. Define Schema
            // Features: f0, f1, ... fN (Float64)
            for (int i = 0; i < cols; i++) {
                fields.add(new Field("f" + i,
                        FieldType.nullable(new ArrowType.FloatingPoint(FloatingPointPrecision.DOUBLE)), null));
            }
            // Labels: target (Int32)
            fields.add(new Field("target", FieldType.nullable(new ArrowType.Int(32, true)), null));

            Schema schema = new Schema(fields);

            try (VectorSchemaRoot root = VectorSchemaRoot.create(schema, allocator)) {
                // 2. Allocate Memory
                root.allocateNew();

                // 3. Populate Vectors
                // This loop is CPU intensive but much faster than JSON string concatenation
                for (int i = 0; i < cols; i++) {
                    Float8Vector featureVector = (Float8Vector) root.getVector("f" + i);
                    featureVector.allocateNew(rows);
                    for (int r = 0; r < rows; r++) {
                        featureVector.setSafe(r, features[r][i]);
                    }
                    featureVector.setValueCount(rows);
                    vectors.add(featureVector);
                }

                IntVector labelVector = (IntVector) root.getVector("target");
                labelVector.allocateNew(rows);
                for (int r = 0; r < rows; r++) {
                    labelVector.setSafe(r, labels[r]);
                }
                labelVector.setValueCount(rows);
                vectors.add(labelVector);

                root.setRowCount(rows);

                // 4. Serialize to Stream (Bytes)
                // Optimization: Pre-allocate buffer to avoid resizing churn
                // Estimate: (Rows * Cols * 8) + (Rows * 4) + Overhead
                int estimatedSize = (rows * cols * 8) + (rows * 4) + 4096;
                ByteArrayOutputStream out = new ByteArrayOutputStream(estimatedSize);
                try (org.apache.arrow.vector.ipc.ArrowStreamWriter writer = new org.apache.arrow.vector.ipc.ArrowStreamWriter(
                        root, null, out)) {
                    writer.start();
                    writer.writeBatch();
                    writer.end();
                }

                // 5. Construct Result
                // NOTE: Parameters (params) are not easily put into Arrow Stream unless we use
                // metadata.
                // For now, we return the stream. Params handled by caller/headers.

                return SerializedData.fromBinary(out.toByteArray(), "application/vnd.apache.arrow.stream");
            }
        }
    }

    @Override
    public String getProtocolName() {
        return "ARROW";
    }
}
