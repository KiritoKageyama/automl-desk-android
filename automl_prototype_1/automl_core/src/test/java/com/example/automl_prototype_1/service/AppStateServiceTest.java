package com.example.automl_prototype_1.service;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.io.File;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for AppStateService — the central application state singleton.
 * Validates singleton pattern, default values, setters, and boundary
 * conditions.
 */
class AppStateServiceTest {

    private AppStateService state;

    @BeforeEach
    void setUp() {
        state = AppStateService.getInstance();
    }

    // ── Singleton ──────────────────────────────────────────────────────────────

    @Test
    void getInstance_returnsSameInstance() {
        assertSame(AppStateService.getInstance(), AppStateService.getInstance());
    }

    @Test
    void hasInstance_isTrueAfterGetInstance() {
        AppStateService.getInstance(); // Ensure created
        assertTrue(AppStateService.hasInstance());
    }

    // ── Default values ────────────────────────────────────────────────────────

    @Test
    void defaults_kFolds_isFive() {
        assertEquals(5, state.getKFolds());
    }

    @Test
    void defaults_scalingStrategy_isStandardization() {
        assertEquals("Standardization", state.getScalingStrategy());
    }

    @Test
    void defaults_tuningEnabled_isFalse() {
        assertFalse(state.isTuningEnabled());
    }

    @Test
    void defaults_standardizationEnabled_isTrue() {
        assertTrue(state.isStandardizationEnabled());
    }

    @Test
    void defaults_balancedOptimizationEnabled_isTrue() {
        assertTrue(state.isBalancedOptimizationEnabled());
    }

    @Test
    void defaults_useGpu_isTrue() {
        assertTrue(state.isUseGpu());
    }

    @Test
    void defaults_loggingEnabled_isTrue() {
        assertTrue(state.isLoggingEnabled());
    }

    @Test
    void defaults_ollamaEnabled_isTrue() {
        assertTrue(state.isOllamaEnabled());
    }

    @Test
    void defaults_dataType_isNumerical() {
        assertEquals(AppStateService.DataType.NUMERICAL, state.getSelectedDataType());
    }

    @Test
    void defaults_trainingMode_isMLAlgorithms() {
        assertEquals(AppStateService.TrainingMode.ML_ALGORITHMS, state.getSelectedTrainingMode());
    }

    // ── Setters and getters ───────────────────────────────────────────────────

    @Test
    void setAndGet_kFolds() {
        state.setKFolds(10);
        assertEquals(10, state.getKFolds());
        state.setKFolds(5); // Reset
    }

    @Test
    void setAndGet_selectedDatasetFile() {
        File f = new File("test.csv");
        state.setSelectedDatasetFile(f);
        assertEquals(f, state.getSelectedDatasetFile());
        state.setSelectedDatasetFile(null); // Reset
    }

    @Test
    void setAndGet_targetColumnName() {
        state.setTargetColumnName("label");
        assertEquals("label", state.getTargetColumnName());
        state.setTargetColumnName(null); // Reset
    }

    @Test
    void setAndGet_selectedAlgorithmNames() {
        List<String> algos = List.of("GA", "BPSO");
        state.setSelectedAlgorithmNames(algos);
        assertEquals(algos, state.getSelectedAlgorithmNames());
        state.setSelectedAlgorithmNames(null); // Reset
    }

    @Test
    void setAndGet_ollamaSettings() {
        state.setOllamaEnabled(false);
        assertFalse(state.isOllamaEnabled());
        state.setOllamaEnabled(true); // Reset

        state.setOllamaModelName("llama3:8b");
        assertEquals("llama3:8b", state.getOllamaModelName());
        state.setOllamaModelName("deepseek-r1:8b"); // Reset
    }

    // ── Boundary: CPU offload ratio clamping ──────────────────────────────────

    @Test
    void setCpuOffloadRatio_clampsToZero() {
        state.setCpuOffloadRatio(-0.5);
        assertEquals(0.0, state.getCpuOffloadRatio(), 0.001);
    }

    @Test
    void setCpuOffloadRatio_clampsToOne() {
        state.setCpuOffloadRatio(1.5);
        assertEquals(1.0, state.getCpuOffloadRatio(), 0.001);
    }

    @Test
    void setCpuOffloadRatio_acceptsValidValue() {
        state.setCpuOffloadRatio(0.3);
        assertEquals(0.3, state.getCpuOffloadRatio(), 0.001);
        state.setCpuOffloadRatio(0.1); // Reset
    }

    // ── Hyperparameters ───────────────────────────────────────────────────────

    @Test
    void algorithmHyperparameters_defaultGA_exists() {
        Map<String, Map<String, Double>> params = state.getAlgorithmHyperparameters();
        assertTrue(params.containsKey("GA"), "Default GA params should exist");
        assertEquals(0.8, params.get("GA").get("crossoverRate"), 0.001);
    }

    @Test
    void setAlgorithmHyperparameter_createsNewEntry() {
        state.setAlgorithmHyperparameter("TEST_ALGO", "testParam", 42.0);
        assertEquals(42.0, state.getAlgorithmHyperparameter("TEST_ALGO", "testParam"), 0.001);
        // Cleanup
        state.getAlgorithmHyperparameters().remove("TEST_ALGO");
    }

    @Test
    void getAlgorithmHyperparameter_unknownAlgo_returnsNull() {
        assertNull(state.getAlgorithmHyperparameter("NONEXISTENT", "param"));
    }

    // ── Enums ─────────────────────────────────────────────────────────────────

    @Test
    void dataType_enum_hasExpectedValues() {
        assertEquals(4, AppStateService.DataType.values().length);
        assertNotNull(AppStateService.DataType.valueOf("NUMERICAL"));
        assertNotNull(AppStateService.DataType.valueOf("IMAGE"));
        assertNotNull(AppStateService.DataType.valueOf("TEXT"));
    }

    @Test
    void trainingMode_enum_hasExpectedValues() {
        assertTrue(AppStateService.TrainingMode.values().length >= 10,
                "TrainingMode should have at least 10 modes");
    }

    // ── Dynamic Models ────────────────────────────────────────────────────────

    @Test
    void registerDynamicModel_addsToMap() {
        state.registerDynamicModel("test-model", "org/test-model");
        Map<String, String> models = state.getDynamicModels();
        assertEquals("org/test-model", models.get("test-model"));
        models.remove("test-model"); // Cleanup
    }
}
