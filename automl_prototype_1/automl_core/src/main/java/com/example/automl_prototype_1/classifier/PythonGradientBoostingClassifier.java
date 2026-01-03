package com.example.automl_prototype_1.classifier;

public class PythonGradientBoostingClassifier extends PythonAdaBoostClassifier {

    public PythonGradientBoostingClassifier() {
        super(100, 0.1, 3);
        // We reuse the parent Logic but override name and train call
    }

    @Override
    public String getName() {
        return "Gradient Boosting (Python)";
    }

    @Override
    public void train(double[][] features, int[] labels) {
        trainGeneric("gbm", features, labels);
    }
}
