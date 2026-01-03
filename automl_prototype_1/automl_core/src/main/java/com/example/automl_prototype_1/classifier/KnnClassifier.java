package com.example.automl_prototype_1.classifier;

import com.example.automl_prototype_1.service.IClassifier;
import smile.classification.KNN;

public class KnnClassifier implements IClassifier {
    private final int k;
    private KNN<double[]> model;

    public KnnClassifier(int k) {
        if (k <= 0)
            throw new IllegalArgumentException("K must be a positive integer.");
        this.k = k;
    }

    public KnnClassifier() {
        this(5);
    }

    @Override
    public String getName() {
        return "K-Nearest Neighbors (KNN)";
    }

    @Override
    public void train(double[][] features, int[] labels) {
        if (features.length == 0) {
            this.model = null;
            return;
        }
        int effectiveK = Math.min(this.k, features.length);
        if (effectiveK == 0)
            effectiveK = 1;
        this.model = KNN.fit(features, labels, effectiveK);
    }

    @Override
    public int[] predict(double[][] features) {
        if (this.model == null)
            return new int[features.length];
        return this.model.predict(features);
    }
}