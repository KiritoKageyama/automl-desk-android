package com.example.automl_prototype_1.model;

public class TuningSimulationData {
    private String strategyName;
    private String parameterDistribution;
    private String optimizationGoal;
    private String sampleOutput;

    public TuningSimulationData() {
    }

    public TuningSimulationData(String strategyName, String parameterDistribution, String optimizationGoal,
            String sampleOutput) {
        this.strategyName = strategyName;
        this.parameterDistribution = parameterDistribution;
        this.optimizationGoal = optimizationGoal;
        this.sampleOutput = sampleOutput;
    }

    public String getStrategyName() {
        return strategyName;
    }

    public String getParameterDistribution() {
        return parameterDistribution;
    }

    public String getOptimizationGoal() {
        return optimizationGoal;
    }

    public String getSampleOutput() {
        return sampleOutput;
    }
}
