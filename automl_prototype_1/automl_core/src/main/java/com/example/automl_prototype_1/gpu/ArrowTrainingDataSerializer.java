package com.example.automl_prototype_1.gpu;

import java.util.Map;

/**
 * Apache Arrow serialization for high-performance training data transfer.
 * Delegates to ArrowService (single source of truth for Arrow serialization).
 */
public class ArrowTrainingDataSerializer implements ITrainingDataSerializer {

    @Override
    public SerializedData serializeTrainingRequest(double[][] features, int[] labels, Map<String, Object> params)
            throws Exception {
        byte[] arrowBytes = com.example.automl_prototype_1.service.pipeline.ArrowService
                .getInstance().serializeTrainData(features, labels);
        return SerializedData.fromBinary(arrowBytes, "application/vnd.apache.arrow.stream");
    }

    @Override
    public String getProtocolName() {
        return "ARROW";
    }
}
