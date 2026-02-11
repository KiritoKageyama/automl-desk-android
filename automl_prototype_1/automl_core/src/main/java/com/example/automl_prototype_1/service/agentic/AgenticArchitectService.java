package com.example.automl_prototype_1.service.agentic;

import com.example.automl_prototype_1.service.LogService;
import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;

/**
 * The "Architect" of the system.
 * Responsible for generating new Code Components (Classes, Envs) via GenAI.
 * 
 * UPGRADE: Now features a Self-Correction Loop.
 * Generate -> Compile -> Check Errors -> Fix -> Retry.
 */
public class AgenticArchitectService {

    private final LocalLlmService llmService;
    private static final int MAX_RETRIES = 3;

    public AgenticArchitectService() {
        this.llmService = new LocalLlmService();
    }

    /**
     * Generates a new RL Environment class, verifying it compiles.
     */
    public String generateEnvironmentCode(String description, String className) {
        if (!llmService.isAvailable()) {
            LogService.error("Architect", "LLM Service not available.");
            return "// Error: LLM Unavailable";
        }

        // 1. Initial Prompt
        String currentCode = initialGeneration(description, className);
        if (!saveClassFile(currentCode, className))
            return "// Error: File Save Failed";

        // 2. Compilation & Correction Loop
        for (int attempt = 1; attempt <= MAX_RETRIES; attempt++) {
            LogService.info("Architect", "Verifying compilation (Attempt " + attempt + "/" + MAX_RETRIES + ")...");

            CompileResult result = runMavenCompile();

            if (result.success) {
                LogService.info("Architect", "✅ Compilation Successful! " + className + " is ready.");
                return currentCode;
            } else {
                LogService.warn("Architect", "❌ Compilation Failed:\n" + result.errorLog);

                if (attempt < MAX_RETRIES) {
                    LogService.info("Architect", "Asking LLM to fix errors...");
                    currentCode = fixCode(currentCode, result.errorLog, className);
                    saveClassFile(currentCode, className); // Overwrite with fix
                } else {
                    LogService.error("Architect", "Failed to fix code after " + MAX_RETRIES + " attempts.");
                    // Optional: Delete broken file to not break project
                    cleanupFile(className);
                    return "// Error: Compilation Failed after " + MAX_RETRIES + " attempts.\n/*\n" + result.errorLog
                            + "\n*/";
                }
            }
        }
        return currentCode;
    }

    private String initialGeneration(String description, String className) {
        String prompt = "JOB: Create a Java class named '" + className
                + "' that implements the 'IEnvironment' interface.\n" +
                "CONTEXT: The app is an AutoML/RL framework. IEnvironment is defined as:\n" +
                "public interface IEnvironment {\n" +
                "    double[] reset();\n" +
                "    StepResult step(int action);\n" +
                "    String render();\n" +
                "    int getObservationSpaceSize();\n" +
                "    int getActionSpaceSize();\n" +
                "    String getName();\n" +
                "    class StepResult { public final double[] state; public final double reward; public final boolean done; public final String info; ...Constructor... }\n"
                +
                "}\n" +
                "TASK: Implement '" + className + "' based on this description: " + description + ".\n" +
                "RULES:\n" +
                "1. MUST implement IEnvironment.\n" +
                "2. MUST be valid Java 11+ code.\n" +
                "3. MUST include package 'com.example.automl_prototype_1.rl.extensions';\n" +
                "4. Provide ONLY the code. No markdown formatting blocks (```), just the code.\n" +
                "5. Ensure robust logic.";

        LogService.info("Architect", "Generating initial code for " + className + "...");
        return cleanLlmOutput(llmService.queryOllama(prompt, "active"));
    }

    private String fixCode(String brokenCode, String errorLog, String className) {
        // Truncate error log if too long to fit in context
        String truncatedError = errorLog.length() > 2000 ? errorLog.substring(0, 2000) + "...[truncated]" : errorLog;

        String prompt = "JOB: Fix this Java code which failed to compile.\n" +
                "CLASS NAME: " + className + "\n" +
                "BROKEN CODE:\n" + brokenCode + "\n\n" +
                "COMPILER ERROR:\n" + truncatedError + "\n\n" +
                "TASK: Return the FIXED code. Complete class. No markdown.";

        return cleanLlmOutput(llmService.queryOllama(prompt, "active"));
    }

    private String cleanLlmOutput(String output) {
        if (output == null)
            return "";
        return output.replace("```java", "").replace("```", "").trim();
    }

    private CompileResult runMavenCompile() {
        try {
            // Find pom.xml (assuming we are in desktop-app/run dir, core is ../automl_core)
            String projectRoot = System.getProperty("user.dir");
            File pomFile = new File(projectRoot, "../automl_core/pom.xml");
            if (!pomFile.exists()) {
                // Try absolute path fallback if relative fails (e.g. running from root)
                pomFile = new File("automl_prototype_1/automl_core/pom.xml");
                if (!pomFile.exists()) {
                    return new CompileResult(false, "Could not find pom.xml at " + pomFile.getAbsolutePath());
                }
            }

            ProcessBuilder pb = new ProcessBuilder("mvn", "compile", "-f", pomFile.getAbsolutePath());
            // pb.inheritIO(); // Don't inherit, we want to capture output
            Process process = pb.start();

            // Capture output for error feedback
            String output;
            try (BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()))) {
                output = reader.lines().collect(Collectors.joining("\n"));
            }
            boolean finished = process.waitFor(30, TimeUnit.SECONDS);

            if (!finished) {
                process.destroyForcibly();
                return new CompileResult(false, "Timeout waiting for maven compile.");
            }

            if (process.exitValue() == 0) {
                return new CompileResult(true, "Success");
            } else {
                return new CompileResult(false, output); // Return full log for analysis
            }

        } catch (Exception e) {
            return new CompileResult(false, "Exception running maven: " + e.getMessage());
        }
    }

    private static class CompileResult {
        boolean success;
        String errorLog;

        public CompileResult(boolean s, String e) {
            success = s;
            errorLog = e;
        }
    }

    public boolean saveClassFile(String code, String className) {
        try {
            String projectRoot = System.getProperty("user.dir");
            // Logic to find correct source folder
            Path path = Path.of(projectRoot,
                    "../automl_core/src/main/java/com/example/automl_prototype_1/rl/extensions/" + className + ".java");

            // Fallback for different run dirs
            if (!path.getParent().toFile().exists()) {
                path = Path
                        .of("automl_prototype_1/automl_core/src/main/java/com/example/automl_prototype_1/rl/extensions/"
                                + className + ".java");
            }

            Files.createDirectories(path.getParent());
            Files.writeString(path, code);
            return true;
        } catch (IOException e) {
            LogService.error("Architect", "Failed to save class: " + e.getMessage());
            return false;
        }
    }

    /**
     * Generates a new Classifier class (IClassifier).
     * Enforces usage of PythonExecutionService for ML tasks if needed.
     */
    public String generateClassifierCode(String description, String className) {
        if (!llmService.isAvailable()) {
            return "// Error: LLM Unavailable";
        }

        // 1. Initial Prompt
        String currentCode = initialClassifierGeneration(description, className);
        if (!saveClassFile(currentCode, className))
            return "// Error: File Save Failed";

        // 2. Compilation & Correction Loop (Reusing existing logic)
        for (int attempt = 1; attempt <= MAX_RETRIES; attempt++) {
            LogService.info("Architect", "Verifying output compilation (Attempt " + attempt + ")...");
            CompileResult result = runMavenCompile();
            if (result.success) {
                return currentCode;
            } else {
                if (attempt < MAX_RETRIES) {
                    currentCode = fixCode(currentCode, result.errorLog, className);
                    saveClassFile(currentCode, className);
                } else {
                    cleanupFile(className);
                    return "// Error: Compilation Failed.\n/*\n" + result.errorLog + "\n*/";
                }
            }
        }
        return currentCode;
    }

    private String initialClassifierGeneration(String description, String className) {
        String prompt = "JOB: Create a Java class named '" + className
                + "' that implements the 'com.example.automl_prototype_1.service.IClassifier' interface.\n" +
                "CONTEXT: The app is an AutoML framework. IClassifier is defined as:\n" +
                "public interface IClassifier {\n" +
                "    String getName();\n" +
                "    void train(double[][] features, int[] labels);\n" +
                "    int[] predict(double[][] features);\n" +
                "}\n" +
                "AVAILABLE SERVICES:\n" +
                "- 'com.example.automl_prototype_1.gpu.PythonExecutionService': Use this for complex ML. Method: trainModel(this, features, labels, paramsMap) and predict(this, features).\n"
                +
                "- 'com.example.automl_prototype_1.gpu.HardwareAcceleratorService': Use for raw OpenCL if needed.\n" +
                "TASK: Implement '" + className + "' based on: " + description + ".\n" +
                "RULES:\n" +
                "1. MUST implement IClassifier and java.io.Serializable.\n" +
                "2. MUST include package 'com.example.automl_prototype_1.rl.extensions'; (or similar extension package)\n"
                +
                "3. If the user asks for a Python-based algo (like LightGBM, CatBoost variant), utilize PythonExecutionService.\n"
                +
                "4. If the user asks for simple logic, implement in Java.\n" +
                "5. Provide ONLY the code. No markdown.";

        LogService.info("Architect", "Generating classifier code for " + className + "...");
        return cleanLlmOutput(llmService.queryOllama(prompt, "active"));
    }

    private String cleanupFile(String className) {
        // ... (existing cleanup)
        try {
            String projectRoot = System.getProperty("user.dir");
            Path path = Path.of(projectRoot,
                    "../automl_core/src/main/java/com/example/automl_prototype_1/rl/extensions/" + className + ".java");
            Files.deleteIfExists(path);
            return "Cleaned up";
        } catch (Exception e) {
            return "Error cleaning";
        }
    }
}
