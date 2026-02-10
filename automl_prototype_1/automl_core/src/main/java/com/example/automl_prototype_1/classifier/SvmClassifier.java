package com.example.automl_prototype_1.classifier;

import com.example.automl_prototype_1.service.IClassifier;
import smile.classification.SVM;
import smile.math.kernel.GaussianKernel;
import smile.math.kernel.LinearKernel;

public class SvmClassifier implements IClassifier {
    private final String kernelType;
    private final double C;
    private final double tol;
    private SVM<double[]> model;

    // Public constructor for creating specific SVMs (used for our fast search
    // version)
    public SvmClassifier(String kernelType, double C, double tol) {
        this.kernelType = kernelType;
        this.C = C;
        this.tol = tol;
    }

    // Public constructor for the high-precision Gaussian SVM (used by UI)
    public SvmClassifier() {
        this("gaussian", 1.0, 1e-3); // High precision
    }

    @Override
    public String getName() {
        return "Support Vector Machine (SVM)";
    }

    @Override
    public void train(double[][] features, int[] labels) {
        if (features.length == 0) {
            this.model = null;
            return;
        }

        // Fix for Smile SVM: It requires/outputs labels -1 and 1.
        // Our system uses 0 and 1. We must map.
        int[] svmLabels = new int[labels.length];
        for (int i = 0; i < labels.length; i++) {
            // Map 0 -> -1, 1 -> 1
            svmLabels[i] = (labels[i] == 0) ? -1 : 1;
        }

        if ("linear".equals(this.kernelType)) {
            this.model = SVM.fit(features, svmLabels, new LinearKernel(), this.C, this.tol);
        } else {
            this.model = SVM.fit(features, svmLabels, new GaussianKernel(1.0), this.C, this.tol);
        }
    }

    @Override
    public int[] predict(double[][] features) {
        if (this.model == null)
            return new int[features.length];

        int[] svmPreds = this.model.predict(features);
        int[] systemPreds = new int[svmPreds.length];

        for (int i = 0; i < svmPreds.length; i++) {
            // Map -1 -> 0, 1 -> 1
            systemPreds[i] = (svmPreds[i] == -1) ? 0 : 1;
        }
        return systemPreds;
    }

    // --- NEW METHOD to create a fast search version ---
    public IClassifier createFastSearchVersion() {
        // Create a Linear SVM with a MUCH lower iteration cap for speed.
        return new SvmClassifier("linear", 1.0, 0.1); // Low precision, faster convergence
    }
}