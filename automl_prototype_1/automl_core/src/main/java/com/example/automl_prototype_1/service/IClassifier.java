package com.example.automl_prototype_1.service;

public interface IClassifier {
    void train(double[][] features, int[] labels);

    int[] predict(double[][] features);

    String getName();
}