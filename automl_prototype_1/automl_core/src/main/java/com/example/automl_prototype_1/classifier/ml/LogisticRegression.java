package com.example.automl_prototype_1.classifier.ml;

import com.example.automl_prototype_1.service.IClassifier;
import smile.data.DataFrame;
import smile.data.formula.Formula;

/**
 * Standard CPU-based Logistic Regression Classifier using Smile.
 * Provides a fallback/baseline comparison for the GPU version.
 */
public class LogisticRegression implements IClassifier {

    private smile.classification.LogisticRegression model;

    @Override
    public String getName() {
        return "Logistic Regression (CPU)";
    }

    @Override
    public void train(double[][] features, int[] labels) {
        if (features.length == 0 || features[0].length == 0)
            return;

        // Prepare data for Smile
        int[][] labelMatrix = new int[labels.length][1];
        for (int i = 0; i < labels.length; i++) {
            labelMatrix[i][0] = labels[i];
        }

        DataFrame dataFrame = DataFrame.of(features).merge(DataFrame.of(labelMatrix, "Class"));
        Formula formula = Formula.of("Class", ".");

        // Smile's LogisticRegression.fit uses BFGS or L-BFGS by default
        this.model = smile.classification.LogisticRegression.fit(formula, dataFrame);
    }

    @Override
    public int[] predict(double[][] features) {
        if (this.model == null)
            return new int[features.length];

        int[] predictions = new int[features.length];
        // Predict row by row
        for (int i = 0; i < features.length; i++) {
            // Check if model expects Tuple or double[]. usually fits on Formula expect
            // Tuple?
            // Actually smile's LR from fit(formula, df) typically returns a LinearModel
            // or similar that might expect Tuple.
            // However, usually pure double[] works if schema matches?
            // Let's assume standard array input if the underlying model supports it.
            // If it fails, we might need to construct Tuple.
            // Given the error 'predict(double[])' existing, it seems it EXPECTS double[].
            predictions[i] = this.model.predict(features[i]);
        }
        return predictions;
    }
}
