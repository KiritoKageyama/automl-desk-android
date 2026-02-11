package com.example.automl_prototype_1.service;

import com.example.automl_prototype_1.gpu.PythonExecutionService;
import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CompletableFuture;

public class LibraryManagerService {

    private static LibraryManagerService instance;

    private LibraryManagerService() {
    }

    public static synchronized LibraryManagerService getInstance() {
        if (instance == null) {
            instance = new LibraryManagerService();
        }
        return instance;
    }

    public CompletableFuture<String> installLibrary(String packageName) {
        return CompletableFuture.supplyAsync(() -> runPipCommand("install", packageName));
    }

    public CompletableFuture<String> uninstallLibrary(String packageName) {
        return CompletableFuture.supplyAsync(() -> runPipCommand("uninstall", "-y", packageName));
    }

    public CompletableFuture<List<String>> getInstalledLibraries() {
        return CompletableFuture.supplyAsync(() -> {
            String output = runPipCommand("list");
            List<String> libraries = new ArrayList<>();
            for (String line : output.split("\\n")) {
                if (!line.trim().isEmpty() && !line.startsWith("Package")) { // Skip header
                    libraries.add(line.trim());
                }
            }
            return libraries;
        });
    }

    private String runPipCommand(String... args) {
        try {
            // We need to find the python executable used by PythonExecutionService
            // For now, we assume 'python' or 'pip' is in path, OR we use the same python
            // path logic
            // PythonExecutionService doesn't expose pythonPath yet, let's assume 'python -m
            // pip'

            String pythonPath = "python"; // Default
            // Ideally we get this from PythonExecutionService if it has a custom path

            List<String> command = new ArrayList<>();
            command.add(pythonPath);
            command.add("-m");
            command.add("pip");
            for (String arg : args) {
                command.add(arg);
            }

            ProcessBuilder pb = new ProcessBuilder(command);
            pb.redirectErrorStream(true);
            Process process = pb.start();

            BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()));
            StringBuilder output = new StringBuilder();
            String line;
            while ((line = reader.readLine()) != null) {
                output.append(line).append("\n");
            }

            int exitCode = process.waitFor();
            if (exitCode != 0) {
                return "Error (Code " + exitCode + "):\n" + output.toString();
            }
            return output.toString();

        } catch (Exception e) {
            e.printStackTrace();
            return "Execution Failed: " + e.getMessage();
        }
    }
}
