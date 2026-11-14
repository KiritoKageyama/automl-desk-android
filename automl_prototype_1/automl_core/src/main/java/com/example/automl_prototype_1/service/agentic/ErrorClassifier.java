package com.example.automl_prototype_1.service.agentic;

/**
 * ErrorClassifier — Phase 9 Self-Repair Enhancement.
 *
 * Classifies Python/Java error strings into structured error types
 * so that SelfCorrectionLoopService can apply targeted remediation
 * strategies instead of blanket algorithm banning.
 *
 * Classification hierarchy:
 * CUDA_OOM — GPU out of memory (reduce batch size)
 * MISSING_DEP — ImportError / ModuleNotFoundError (fallback to CPU)
 * DATA_ERROR — bad input shape, NaN, infinite values
 * TIMEOUT — execution exceeded time limit
 * COMPILE_ERR — Java compile failure in generated classifiers
 * CONVERGENCE — algorithm diverged or produced NaN fitness
 * UNKNOWN — anything else
 */
public class ErrorClassifier {

    public enum ErrorType {
        CUDA_OOM,
        MISSING_DEP,
        DATA_ERROR,
        TIMEOUT,
        COMPILE_ERR,
        CONVERGENCE,
        UNKNOWN
    }

    // -------------------------------------------------------------------------
    // Single shared instance (stateless helper — no need for full singleton)
    // -------------------------------------------------------------------------
    private static final ErrorClassifier INSTANCE = new ErrorClassifier();

    public static ErrorClassifier getInstance() {
        return INSTANCE;
    }

    private ErrorClassifier() {
    }

    // -------------------------------------------------------------------------
    // Classification
    // -------------------------------------------------------------------------

    /**
     * Classify an error string into a structured ErrorType.
     *
     * @param errorText raw stderr / exception message
     * @return best-match ErrorType
     */
    public ErrorType classify(String errorText) {
        if (errorText == null || errorText.isBlank())
            return ErrorType.UNKNOWN;
        String lower = errorText.toLowerCase();

        // GPU / CUDA memory
        if (lower.contains("cuda out of memory") || lower.contains("oom") ||
                lower.contains("outofmemoryerror") || lower.contains("cudaoutofmemory")) {
            return ErrorType.CUDA_OOM;
        }

        // Missing Python dependency
        if (lower.contains("importerror") || lower.contains("modulenotfounderror") ||
                lower.contains("no module named") || lower.contains("cannot import")) {
            return ErrorType.MISSING_DEP;
        }

        // Data / shape / NaN issues
        if (lower.contains("nan") || lower.contains("infinity") ||
                lower.contains("valueerror") || lower.contains("shape mismatch") ||
                lower.contains("invalid value") || lower.contains("dtype") ||
                lower.contains("index out of bounds") || lower.contains("arrayindexoutofboundsexception")) {
            return ErrorType.DATA_ERROR;
        }

        // Timeout
        if (lower.contains("timeout") || lower.contains("timed out") ||
                lower.contains("timeoutexception") || lower.contains("interruptedexception") ||
                lower.contains("killed") || lower.contains("signal 9")) {
            return ErrorType.TIMEOUT;
        }

        // Java compile errors (from AgenticArchitectService generated code)
        if (lower.contains("compilationexception") || lower.contains("javac") ||
                lower.contains("compilation failed") || lower.contains("cannot find symbol") ||
                lower.contains("error: ") && lower.contains(".java:")) {
            return ErrorType.COMPILE_ERR;
        }

        // Algorithm convergence / NaN fitness
        if (lower.contains("diverged") || lower.contains("nan fitness") ||
                lower.contains("fitness is nan") || lower.contains("no improvement") ||
                lower.contains("convergence failed") || lower.contains("arithmetic")) {
            return ErrorType.CONVERGENCE;
        }

        return ErrorType.UNKNOWN;
    }

    /**
     * Returns a short human-readable label for each type.
     */
    public String describe(ErrorType type) {
        return switch (type) {
            case CUDA_OOM -> "GPU Out-of-Memory";
            case MISSING_DEP -> "Missing Python Dependency";
            case DATA_ERROR -> "Data / Shape Error";
            case TIMEOUT -> "Execution Timeout";
            case COMPILE_ERR -> "Java Compile Error";
            case CONVERGENCE -> "Algorithm Divergence";
            case UNKNOWN -> "Unknown Error";
        };
    }

    /**
     * Returns a remediation suggestion for the error type.
     * Used by SelfCorrectionLoopService to decide next action.
     */
    public String remediation(ErrorType type) {
        return switch (type) {
            case CUDA_OOM -> "Reduce batch size or switch to CPU fallback.";
            case MISSING_DEP -> "Install missing library or disable GPU path.";
            case DATA_ERROR -> "Check for NaN/Inf in dataset; verify feature counts.";
            case TIMEOUT -> "Reduce algorithm iterations or increase timeout budget.";
            case COMPILE_ERR -> "Check generated Java code for syntax errors; retry with smaller template.";
            case CONVERGENCE -> "Reduce learning rate or population size; try a different algorithm.";
            case UNKNOWN -> "Check full logs for root cause; consider algorithm ban.";
        };
    }
}
