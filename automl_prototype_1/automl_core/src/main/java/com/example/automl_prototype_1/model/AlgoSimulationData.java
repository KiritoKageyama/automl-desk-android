package com.example.automl_prototype_1.model;

public class AlgoSimulationData {
    private String name;
    private String coreFormula;
    private String processDescription;
    private String sampleOutput;

    public AlgoSimulationData(String name, String coreFormula, String processDescription, String sampleOutput) {
        this.name = name;
        this.coreFormula = coreFormula;
        this.processDescription = processDescription;
        this.sampleOutput = sampleOutput;
    }

    public String getName() {
        return name;
    }

    public String getCoreFormula() {
        return coreFormula;
    }

    public String getProcessDescription() {
        return processDescription;
    }

    public String getSampleOutput() {
        return sampleOutput;
    }
}
