package com.example.automl_prototype_1.gpu;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Locale;

/**
 * Restricts CUSTOM_SCRIPT execution to allowlisted directories.
 */
public final class ScriptAllowlist {

    private ScriptAllowlist() {
    }

    public static boolean isAllowed(String scriptPath) {
        if (scriptPath == null || scriptPath.isBlank()) {
            return false;
        }
        try {
            Path target = Paths.get(scriptPath).toAbsolutePath().normalize();
            String name = target.getFileName() != null
                    ? target.getFileName().toString().toLowerCase(Locale.ROOT)
                    : "";
            if (!name.endsWith(".py")) {
                return false;
            }

            Path userDir = Paths.get(System.getProperty("user.dir")).toAbsolutePath().normalize();
            Path home = Paths.get(System.getProperty("user.home")).toAbsolutePath().normalize();

            Path[] roots = new Path[] {
                    userDir.resolve("automl_core").resolve("src").resolve("main").resolve("resources").resolve("python"),
                    userDir.resolve("..").resolve("automl_core").resolve("src").resolve("main").resolve("resources")
                            .resolve("python").normalize(),
                    userDir.resolve("wsl_temp"),
                    home.resolve(".automl").resolve("scripts")
            };
            for (Path root : roots) {
                Path abs = root.toAbsolutePath().normalize();
                if (target.startsWith(abs)) {
                    return true;
                }
            }
            // Also allow scripts extracted under java.io.tmpdir/automl*
            Path tmp = Paths.get(System.getProperty("java.io.tmpdir")).toAbsolutePath().normalize();
            if (target.startsWith(tmp) && target.toString().toLowerCase(Locale.ROOT).contains("automl")) {
                return true;
            }
            return false;
        } catch (Exception e) {
            return false;
        }
    }
}
