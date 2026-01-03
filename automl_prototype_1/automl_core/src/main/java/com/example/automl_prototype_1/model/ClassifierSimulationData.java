package com.example.automl_prototype_1.model;

public class ClassifierSimulationData {
    private String name;
    private String decisionFunction;
    private String lossFormula;
    private String sampleOutput;

    public ClassifierSimulationData(String name, String decisionFunction, String lossFormula, String sampleOutput) {
        this.name = name;
        this.decisionFunction = decisionFunction;
        this.lossFormula = lossFormula;
        this.sampleOutput = sampleOutput;
    }

    public String getName() {
        return name;
    }

    public String getDecisionFunction() {
        return decisionFunction;
    }

    public String getLossFormula() {
        return lossFormula;
    }

    public String getSampleOutput() {
        return sampleOutput;
    }
}
