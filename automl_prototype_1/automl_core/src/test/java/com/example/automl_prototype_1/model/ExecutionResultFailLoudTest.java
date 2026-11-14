package com.example.automl_prototype_1.model;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class ExecutionResultFailLoudTest {

    @Test
    void failedResultIsNotSuccessAndHasNaNMetrics() {
        ExecutionResult r = ExecutionResult.failed("GA", "feature selection exploded");
        assertFalse(r.isSuccess());
        assertEquals("feature selection exploded", r.getErrorMessage());
        assertEquals("GA", r.getAlgorithmName());
        assertTrue(Double.isNaN(r.getAccuracy()));
        assertTrue(Double.isNaN(r.getPrecision()));
    }

    @Test
    void defaultConstructorIsSuccess() {
        assertTrue(new ExecutionResult().isSuccess());
    }
}
