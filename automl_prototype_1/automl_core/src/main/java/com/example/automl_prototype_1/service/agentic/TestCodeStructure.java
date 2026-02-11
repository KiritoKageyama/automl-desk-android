package com.example.automl_prototype_1.service.agentic;

public class TestCodeStructure {
    public static void main(String[] args) {
        System.out.println("--- Testing Code Structure Service ---");
        CodeStructureService service = new CodeStructureService();

        // Test on DomainGuardService.java
        String target = "automl_core/src/main/java/com/example/automl_prototype_1/service/agentic/DomainGuardService.java";

        long start = System.currentTimeMillis();
        String summary = service.analyzeFileSummary(target);
        long duration = System.currentTimeMillis() - start;

        System.out.println("Analysis Time: " + duration + "ms");
        System.out.println(summary);

        if (summary.contains("Class: DomainGuardService") && summary.contains("classify")) {
            System.out.println("[SUCCESS] Structure extracted correctly.");
        } else {
            System.out.println("[FAILURE] metadata missing.");
        }
    }
}
