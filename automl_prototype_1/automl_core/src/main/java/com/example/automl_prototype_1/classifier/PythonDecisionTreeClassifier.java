package com.example.automl_prototype_1.classifier;

public class PythonDecisionTreeClassifier extends PythonAdaBoostClassifier {

    public PythonDecisionTreeClassifier() {
        super(1, 1.0, 0); // 0 depth -> None in script, n_est ignored
    }

    @Override
    public String getName() {
        return "Decision Tree (Python)";
    }

    @Override
    public void train(double[][] features, int[] labels) {
        trainGeneric("dt", features, labels); // 'dt' maps to DecisionTree in script
    }
}
