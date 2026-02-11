package com.example.automl_prototype_1.classifier.cpu;

import com.example.automl_prototype_1.service.IClassifier;
import smile.classification.GradientTreeBoost;
import smile.data.DataFrame;
import smile.data.formula.Formula;
import java.util.Properties;

public class XGBoostClassifier implements IClassifier {
    private final int ntrees;
    private final int maxDepth;
    private GradientTreeBoost model;

    public XGBoostClassifier() {
        this(200, 8); // Default: 200 trees, depth 8
    }

    public XGBoostClassifier(int ntrees) {
        this(ntrees, 8);
    }

    public XGBoostClassifier(int ntrees, int maxDepth) {
        if (ntrees <= 0)
            throw new IllegalArgumentException("Number of trees must be positive.");
        if (maxDepth <= 0)
            throw new IllegalArgumentException("Max depth must be positive.");
        this.ntrees = ntrees;
        this.maxDepth = maxDepth;
    }

    @Override
    public String getName() {
        return "XGBoost";
    }

    @Override
    public void train(double[][] features, int[] labels) {
        if (features.length == 0 || features[0].length == 0) {
            this.model = null;
            return;
        }

        // STEP 1: Convert 1D labels array to 2D array for DataFrame.
        int[][] labelMatrix = new int[labels.length][1];
        for (int i = 0; i < labels.length; i++) {
            labelMatrix[i][0] = labels[i];
        }

        // STEP 2: Create the DataFrame.
        DataFrame dataFrame = DataFrame.of(features).merge(DataFrame.of(labelMatrix, "Class"));

        // STEP 3: Create the Formula.
        Formula formula = Formula.of("Class", ".");

        // STEP 4: Call the fit method using Properties.
        Properties props = new Properties();
        props.setProperty("smile.gradient.boost.trees", String.valueOf(this.ntrees));
        props.setProperty("trees", String.valueOf(this.ntrees)); // Fallback/Alternative key
        props.setProperty("smile.gradient.boost.max.depth", String.valueOf(this.maxDepth));
        props.setProperty("max.depth", String.valueOf(this.maxDepth)); // Fallback

        this.model = GradientTreeBoost.fit(formula, dataFrame, props);
    }

    @Override
    public int[] predict(double[][] features) {
        if (this.model == null) {
            return new int[features.length];
        }
        DataFrame predictionFrame = DataFrame.of(features);
        return this.model.predict(predictionFrame);
    }
}