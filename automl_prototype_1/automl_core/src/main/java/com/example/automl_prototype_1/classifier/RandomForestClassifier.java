package com.example.automl_prototype_1.classifier;

import com.example.automl_prototype_1.service.IClassifier;
import smile.classification.RandomForest;
import smile.data.DataFrame;
import smile.data.formula.Formula;
import java.util.Properties;

public class RandomForestClassifier implements IClassifier {
    private final int nTrees;
    private final ThreadLocal<RandomForest> model = new ThreadLocal<>();

    public RandomForestClassifier(int nTrees) {
        if (nTrees <= 0)
            throw new IllegalArgumentException("Number of trees must be positive.");
        this.nTrees = nTrees;
    }

    public RandomForestClassifier() {
        this(100);
    }

    @Override
    public String getName() {
        return "Random Forest (RF)";
    }

    @Override
    public void train(double[][] features, int[] labels) {
        if (features.length == 0 || features[0].length == 0) {
            this.model.remove();
            return;
        }

        // 1. Convert 1D labels array to 2D array for DataFrame.
        int[][] labelMatrix = new int[labels.length][1];
        for (int i = 0; i < labels.length; i++) {
            labelMatrix[i][0] = labels[i];
        }

        // 2. Create the DataFrame.
        DataFrame dataFrame = DataFrame.of(features).merge(DataFrame.of(labelMatrix, "Class"));

        // 3. Create the Formula.
        Formula formula = Formula.of("Class", ".");

        // 4. Call the fit method using Properties.
        Properties props = new Properties();
        props.setProperty("smile.random.forest.trees", String.valueOf(this.nTrees));
        props.setProperty("trees", String.valueOf(this.nTrees)); // Fallback

        this.model.set(RandomForest.fit(formula, dataFrame, props));
    }

    @Override
    public int[] predict(double[][] features) {
        RandomForest m = this.model.get();
        if (m == null)
            return new int[features.length];
        DataFrame predictionFrame = DataFrame.of(features);
        return m.predict(predictionFrame);
    }
}