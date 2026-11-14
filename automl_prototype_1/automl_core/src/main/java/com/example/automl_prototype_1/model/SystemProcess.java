package com.example.automl_prototype_1.model;

/**
 * Represents a running Windows process detected by the SystemOptimizerService.
 */
public class SystemProcess {
    private String name;
    private String pid;
    private String memoryUsageStr;
    private long memoryUsageMb;
    private String aiAnalysis; // To be populated by Agentic AI
    private boolean isSelectedForKill;

    public SystemProcess(String name, String pid, String memoryUsageStr, long memoryUsageMb) {
        this.name = name;
        this.pid = pid;
        this.memoryUsageStr = memoryUsageStr;
        this.memoryUsageMb = memoryUsageMb;
        this.isSelectedForKill = false;
        this.aiAnalysis = "Waiting for AI analysis...";
    }

    public String getName() {
        return name;
    }

    public String getPid() {
        return pid;
    }

    public String getMemoryUsageStr() {
        return memoryUsageStr;
    }

    public long getMemoryUsageMb() {
        return memoryUsageMb;
    }

    public String getAiAnalysis() {
        return aiAnalysis;
    }

    public void setAiAnalysis(String aiAnalysis) {
        this.aiAnalysis = aiAnalysis;
    }

    public boolean isSelectedForKill() {
        return isSelectedForKill;
    }

    public void setSelectedForKill(boolean selectedForKill) {
        isSelectedForKill = selectedForKill;
    }

    @Override
    public String toString() {
        return name + " (PID: " + pid + ") - " + memoryUsageMb + " MB";
    }
}
