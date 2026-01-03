package com.example.automl_prototype_1.model;

import com.example.automl_prototype_1.algorithm.OptimizationAlgorithm;
import com.example.automl_prototype_1.service.IClassifier;

/**
 * Represents a recommended pairing of an optimization algorithm and a base
 * classifier.
 *
 * @param optimizer  The recommended optimization algorithm instance.
 * @param classifier The recommended base classifier instance.
 * @param reason     A human-readable string explaining why this pair was
 *                   recommended.
 * @param score      A numerical score indicating the strength of the
 *                   recommendation (higher is better).
 */
public record AlgorithmRecommendation(
                OptimizationAlgorithm optimizer,
                IClassifier classifier,
                String reason,
                double score) {
}