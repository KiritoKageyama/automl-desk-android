package com.example.automl_prototype_1.service.agentic;

import com.example.automl_prototype_1.service.LogService;

import java.io.IOException;
import java.nio.file.*;

import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * Service responsible for performing SAFE code modifications requested by the
 * AI.
 * Enforces strict directory constraints and creates backups.
 */
public class CodeModificationService {

    private static final String PROJECT_ROOT = System.getProperty("user.dir");

    private static final List<String> PROTECTED_PATHS = List.of(
            "automl_core",
            "desktop-app",
            "pom.xml" // Protect build config too
    );

    public CodeModificationService() {
        LogService.info("CodeModService", "Initialized. Root Constraint: " + PROJECT_ROOT);
    }

    /**
     * Reads a file reliably.
     */
    public String readFile(String relativePath) throws IOException {
        Path path = validatePath(relativePath);
        if (!Files.exists(path)) {
            throw new IOException("File not found: " + relativePath);
        }
        return Files.readString(path);
    }

    /**
     * Writes to a file safely.
     * 1. Validates path is within Project Root.
     * 2. ENFORCES CORE PROTECTION: Blocks writes to core modules.
     * 3. Creates a .bak backup of the existing file.
     * 4. Overwrites/Creates with new content.
     */
    public void writeFile(String relativePath, String content) throws IOException {
        Path targetPath = validatePath(relativePath);
        String normalPath = relativePath.replace("\\", "/");

        // --- ENFORCE CORE PROTECTION ---
        for (String protectedDir : PROTECTED_PATHS) {
            if (normalPath.startsWith(protectedDir) || normalPath.contains("/" + protectedDir + "/")) {
                throw new SecurityException("Security Policy Violation: AI is not allowed to modify Core Files in '"
                        + protectedDir + "'. Please duplicate the file to 'custom_extensions/' and modify it there.");
            }
        }

        // Safety: Check if write lock exists (prevent AI from overwriting while user is
        // editing?)
        // For now, simpler safety: Just backup.

        if (Files.exists(targetPath)) {
            // Create Backup
            Path backupPath = Paths.get(targetPath.toString() + ".bak");
            Files.copy(targetPath, backupPath, StandardCopyOption.REPLACE_EXISTING);
            LogService.info("CodeModService", "Backup created: " + backupPath.getFileName());
        } else {
            // Ensure parent dirs exist
            Files.createDirectories(targetPath.getParent());
        }

        Files.writeString(targetPath, content, StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING);
        LogService.info("CodeModService", "Wrote file: " + relativePath);
    }

    /**
     * Lists files in the project to help AI understand structure.
     * Excludes hidden folders (.git, .idea, target, node_modules).
     */
    public List<String> listProjectFiles() throws IOException {
        // Limit depth to avoid overload
        try (Stream<Path> stream = Files.walk(Paths.get(PROJECT_ROOT), 5)) {
            return stream
                    .filter(p -> !Files.isDirectory(p))
                    .map(p -> Paths.get(PROJECT_ROOT).relativize(p).toString())
                    .filter(s -> !s.startsWith(".") && !s.contains("target") && !s.contains(".git")
                            && !s.contains("node_modules"))
                    .collect(Collectors.toList());
        }
    }

    /**
     * Ensures the path is inside the project root to prevent accessing C:/Windows
     * etc.
     */
    private Path validatePath(String relativePath) throws SecurityException {
        // Normalizing to prevent .. attacks
        Path resolved = Paths.get(PROJECT_ROOT, relativePath).normalize();
        if (!resolved.startsWith(Paths.get(PROJECT_ROOT).normalize())) {
            throw new SecurityException("Access Denied: Path is outside project root. " + resolved);
        }
        return resolved;
    }
}
