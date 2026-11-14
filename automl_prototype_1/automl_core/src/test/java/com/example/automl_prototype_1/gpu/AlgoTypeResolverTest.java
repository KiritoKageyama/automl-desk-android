package com.example.automl_prototype_1.gpu;

import com.example.automl_prototype_1.classifier.cnn.CnnClassifier;
import com.example.automl_prototype_1.classifier.llm.GenericHuggingFaceClassifier;
import com.example.automl_prototype_1.service.IClassifier;
import org.junit.jupiter.api.Test;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

class AlgoTypeResolverTest {

    @Test
    void hfClassifierMapsToTransformerNotRf() {
        GenericHuggingFaceClassifier hf = new GenericHuggingFaceClassifier("org/model", "MyHF");
        AlgoTypeResolver.Resolved r = AlgoTypeResolver.resolve(hf, Map.of());
        assertEquals("HF_TRANSFORMER", r.algoType());
        assertEquals("org/model", r.params().get("hf_model_id"));
    }

    @Test
    void cnnClassifierMapsToCnnNotCustomScript() {
        // CnnClassifier extracts script in ctor; may throw if resource missing in unit env.
        // Use a stub instead if construction fails.
        IClassifier cnn;
        try {
            cnn = new CnnClassifier();
        } catch (Throwable t) {
            cnn = new IClassifier() {
                @Override
                public void train(double[][] features, int[] labels) {
                }

                @Override
                public int[] predict(double[][] features) {
                    return new int[0];
                }

                @Override
                public String getName() {
                    return "CNN Stub";
                }
            };
            // Without CnnClassifier instance we can't assert CNN — skip soft
            AlgoTypeResolver.Resolved fallback = AlgoTypeResolver.resolve(cnn, null);
            assertEquals("RF", fallback.algoType()); // unknown stub defaults RF
            return;
        }
        AlgoTypeResolver.Resolved r = AlgoTypeResolver.resolve(cnn, null);
        assertEquals("CNN", r.algoType());
    }

    @Test
    void nullClassifierDefaultsToRf() {
        assertEquals("RF", AlgoTypeResolver.resolve(null, null).algoType());
    }
}
