package com.example.automl_prototype_1.service.agentic;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for DomainGuardService — the prompt classifier that ensures
 * LLM queries stay within the AutoML domain.
 */
class DomainGuardServiceTest {

    private DomainGuardService guard;

    @BeforeEach
    void setUp() {
        guard = new DomainGuardService();
    }

    // ── On-domain prompts ─────────────────────────────────────────────────────

    @Test
    void classify_mlPrompt_hasPositiveScore() {
        DomainGuardService.ClassificationResult result = guard
                .classify("Which algorithm is best for classification with high-dimensional data?");
        assertTrue(result.score > 0, "ML classification prompt should score positively");
    }

    @Test
    void classify_featureSelectionPrompt_hasPositiveScore() {
        DomainGuardService.ClassificationResult result = guard
                .classify("How does genetic algorithm feature selection work?");
        assertTrue(result.score > 0, "Feature selection prompt should score positively");
    }

    @Test
    void classify_datasetPrompt_hasPositiveScore() {
        DomainGuardService.ClassificationResult result = guard
                .classify("Analyze the dataset and recommend preprocessing steps");
        assertTrue(result.score > 0, "Dataset analysis prompt should score positively");
    }

    // ── Off-domain / negative prompts ─────────────────────────────────────────

    @Test
    void classify_completelyOffTopic_hasZeroOrNegativeScore() {
        DomainGuardService.ClassificationResult result = guard
                .classify("What is the capital of France and how do I bake bread");
        assertTrue(result.score <= 0, "Off-topic prompt should score zero or negative");
    }

    @Test
    void classify_emptyPrompt_hasZeroScore() {
        DomainGuardService.ClassificationResult result = guard.classify("");
        assertEquals(0.0, result.score, "Empty prompt should have zero score");
    }

    // ── Domain identification ─────────────────────────────────────────────────

    @Test
    void classify_mlPrompt_identifiesDomain() {
        DomainGuardService.ClassificationResult result = guard
                .classify("Explain how random forest classification works for feature selection");
        assertNotEquals("UNKNOWN", result.domain,
                "ML prompt should be classified to a known domain, not UNKNOWN");
    }

    // ── Structure ─────────────────────────────────────────────────────────────

    @Test
    void classificationResult_hasAllFields() {
        DomainGuardService.ClassificationResult result = guard.classify("test");
        assertNotNull(result.domain, "Result should always have a domain");
        // score can be any double, just check it's a real number
        assertFalse(Double.isNaN(result.score), "Score should not be NaN");
    }
}
