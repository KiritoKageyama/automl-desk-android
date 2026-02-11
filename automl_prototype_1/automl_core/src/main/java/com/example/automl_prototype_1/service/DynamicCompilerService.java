package com.example.automl_prototype_1.service;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import javax.tools.JavaCompiler;
import javax.tools.ToolProvider;
import javax.tools.StandardJavaFileManager;
import javax.tools.JavaFileObject;

/**
 * Service for compiling Java source files at runtime.
 * Essential for the "Meta-Architect" phase where the app extends itself.
 */
public class DynamicCompilerService {

    public static class CompilationResult {
        public final boolean success;
        public final String messages;

        public CompilationResult(boolean success, String messages) {
            this.success = success;
            this.messages = messages;
        }
    }

    /**
     * Compiles a specific Java source file.
     * 
     * @param sourceFile The .java file to compile.
     * @return CompilationResult
     */
    public CompilationResult compile(File sourceFile) {
        JavaCompiler compiler = ToolProvider.getSystemJavaCompiler();
        if (compiler == null) {
            return new CompilationResult(false,
                    "No Java Compiler available. Ensure you are running with a JDK, not just JRE.");
        }

        StandardJavaFileManager fileManager = compiler.getStandardFileManager(null, null, null);
        Iterable<? extends JavaFileObject> compilationUnits = fileManager
                .getJavaFileObjectsFromFiles(Arrays.asList(sourceFile));

        // Build Classpath
        // We need to include the current running classpath so the new class can find
        // Core interfaces
        String classpath = System.getProperty("java.class.path");
        List<String> options = Arrays.asList("-classpath", classpath, "-d", "target/classes"); // Output to current
                                                                                               // target/classes

        // Run Compilation
        // We capture output to a stream if needed, but for now we trust the return code
        // and maybe capture diagnostics if we implement a DiagnosticListener

        JavaCompiler.CompilationTask task = compiler.getTask(
                null, // Writer (null = System.err)
                fileManager,
                null, // DiagnosticListener
                options,
                null,
                compilationUnits);

        boolean success = task.call();

        try {
            fileManager.close();
        } catch (IOException e) {
            e.printStackTrace();
        }

        return new CompilationResult(success, success ? "Compilation Successful" : "Compilation Failed (Check Logs)");
    }
}
