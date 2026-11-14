package com.example.automl_prototype_1.registry;

import com.example.automl_prototype_1.algorithm.*;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for AlgorithmRegistry — the single source of truth for algorithm
 * name→instance resolution.
 */
class AlgorithmRegistryTest {

    private final AlgorithmRegistry registry = AlgorithmRegistry.getInstance();

    // ── Singleton ──────────────────────────────────────────────────────────────

    @Test
    void getInstance_returnsSameInstance() {
        assertSame(AlgorithmRegistry.getInstance(), AlgorithmRegistry.getInstance());
    }

    // ── getAlgorithm: valid short codes ────────────────────────────────────────

    @Test
    void getAlgorithm_GA_returnsGeneticAlgorithm() {
        OptimizationAlgorithm algo = registry.getAlgorithm("GA");
        assertNotNull(algo);
        assertInstanceOf(GeneticAlgorithm.class, algo);
    }

    @Test
    void getAlgorithm_IGPSO_returnsIGPSO() {
        assertInstanceOf(IGPSO.class, registry.getAlgorithm("IGPSO"));
    }

    @Test
    void getAlgorithm_WWO_returnsWaterWave() {
        assertInstanceOf(WaterWaveOptimization.class, registry.getAlgorithm("WWO"));
    }

    @Test
    void getAlgorithm_BPSO_returnsBinaryPSO() {
        assertInstanceOf(BinaryParticleSwarmOptimization.class, registry.getAlgorithm("BPSO"));
    }

    @Test
    void getAlgorithm_ASO_returnsAtomSearch() {
        assertInstanceOf(AtomSearchOptimization.class, registry.getAlgorithm("ASO"));
    }

    @Test
    void getAlgorithm_PLS_returnsParallelLocalSearch() {
        assertInstanceOf(ParallelLocalSearchAlgorithm.class, registry.getAlgorithm("PLS"));
    }

    @Test
    void getAlgorithm_SA_returnsSimulatedAnnealing() {
        assertInstanceOf(SimulatedAnnealing.class, registry.getAlgorithm("SA"));
    }

    @Test
    void getAlgorithm_GWO_returnsGreyWolf() {
        assertInstanceOf(GreyWolfOptimizer.class, registry.getAlgorithm("GWO"));
    }

    @Test
    void getAlgorithm_DE_returnsDifferentialEvolution() {
        assertInstanceOf(DifferentialEvolution.class, registry.getAlgorithm("DE"));
    }

    // ── Case insensitivity ────────────────────────────────────────────────────

    @ParameterizedTest
    @ValueSource(strings = { "ga", "Ga", "gA", "GA" })
    void getAlgorithm_isCaseInsensitive(String name) {
        assertNotNull(registry.getAlgorithm(name), "Should resolve '" + name + "' case-insensitively");
    }

    // ── Long name aliases ─────────────────────────────────────────────────────

    @Test
    void getAlgorithm_longName_GeneticAlgorithm() {
        assertInstanceOf(GeneticAlgorithm.class, registry.getAlgorithm("Genetic Algorithm"));
    }

    @Test
    void getAlgorithm_longName_WaterWaveOptimization() {
        assertInstanceOf(WaterWaveOptimization.class, registry.getAlgorithm("Water Wave Optimization"));
    }

    @Test
    void getAlgorithm_longName_ParallelLocalSearch() {
        assertInstanceOf(ParallelLocalSearchAlgorithm.class, registry.getAlgorithm("Parallel Local Search"));
    }

    // ── Non-optimization pass-throughs ────────────────────────────────────────

    @Test
    void getAlgorithm_ManualSelection() {
        assertInstanceOf(ManualSelectionAlgorithm.class, registry.getAlgorithm("Manual Selection"));
    }

    @Test
    void getAlgorithm_NoOptimization() {
        assertInstanceOf(NoOptimizationAlgorithm.class, registry.getAlgorithm("No Optimization"));
    }

    // ── Edge cases ────────────────────────────────────────────────────────────

    @Test
    void getAlgorithm_null_returnsNull() {
        assertNull(registry.getAlgorithm(null));
    }

    @Test
    void getAlgorithm_unknown_returnsNull() {
        assertNull(registry.getAlgorithm("NonexistentAlgorithm"));
    }

    @Test
    void getAlgorithm_empty_returnsNull() {
        assertNull(registry.getAlgorithm(""));
    }

    // ── Fresh instances ───────────────────────────────────────────────────────

    @Test
    void getAlgorithm_returnsFreshInstanceEachTime() {
        OptimizationAlgorithm a1 = registry.getAlgorithm("GA");
        OptimizationAlgorithm a2 = registry.getAlgorithm("GA");
        assertNotSame(a1, a2, "Each call should return a new instance");
    }

    // ── createRandomized ──────────────────────────────────────────────────────

    @Test
    void createRandomized_GA_returnsNonNull() {
        assertNotNull(registry.createRandomized("GA"));
    }

    @Test
    void createRandomized_null_returnsNull() {
        assertNull(registry.createRandomized(null));
    }

    @Test
    void createRandomized_unknown_returnsNull() {
        assertNull(registry.createRandomized("FakeAlgo"));
    }

    // ── getAlgorithmNames ─────────────────────────────────────────────────────

    @Test
    void getAlgorithmNames_containsAllShortCodes() {
        List<String> names = registry.getAlgorithmNames();
        assertTrue(names.contains("GA"));
        assertTrue(names.contains("IGPSO"));
        assertTrue(names.contains("WWO"));
        assertTrue(names.contains("BPSO"));
        assertTrue(names.contains("ASO"));
        assertTrue(names.contains("PLS"));
        assertTrue(names.contains("SA"));
        assertTrue(names.contains("GWO"));
        assertTrue(names.contains("DE"));
    }

    @Test
    void getAlgorithmNames_isUnmodifiable() {
        List<String> names = registry.getAlgorithmNames();
        assertThrows(UnsupportedOperationException.class, () -> names.add("TEST"));
    }

    @Test
    void getAlgorithmNames_everyNameResolvesToInstance() {
        for (String name : registry.getAlgorithmNames()) {
            assertNotNull(registry.getAlgorithm(name),
                    "Algorithm name '" + name + "' should resolve to an instance");
        }
    }
}
