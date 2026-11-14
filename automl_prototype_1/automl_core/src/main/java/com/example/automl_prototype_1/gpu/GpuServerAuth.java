package com.example.automl_prototype_1.gpu;

import com.example.automl_prototype_1.service.LogService;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.SecureRandom;
import java.util.HexFormat;

/**
 * Shared token for Java ↔ GPU server (:5006). Token file: ~/.automl_gpu_token
 * Disable with AUTOML_GPU_AUTH=false (localhost-only still recommended).
 */
public final class GpuServerAuth {

    public static final String HEADER = "X-AutoML-Gpu-Token";
    private static final GpuServerAuth INSTANCE = new GpuServerAuth();

    private volatile String token = "";
    private volatile boolean enabled = true;
    private volatile boolean resolved;

    private GpuServerAuth() {
    }

    public static GpuServerAuth getInstance() {
        return INSTANCE;
    }

    public synchronized void ensureResolved() {
        if (resolved) {
            return;
        }
        enabled = !"false".equalsIgnoreCase(System.getenv().getOrDefault("AUTOML_GPU_AUTH", "true"));
        if (!enabled) {
            token = "";
            resolved = true;
            LogService.info("GpuAuth", "GPU token auth DISABLED (AUTOML_GPU_AUTH=false)");
            return;
        }
        String env = System.getenv("AUTOML_GPU_TOKEN");
        if (env != null && !env.isBlank()) {
            token = env.trim();
        } else {
            token = loadOrCreateTokenFile();
        }
        resolved = true;
        LogService.info("GpuAuth", "GPU token auth ON → " + tokenFile());
    }

    public boolean isEnabled() {
        ensureResolved();
        return enabled;
    }

    public String getToken() {
        ensureResolved();
        return token;
    }

    public Path tokenFile() {
        return Path.of(System.getProperty("user.home"), ".automl_gpu_token");
    }

    private String loadOrCreateTokenFile() {
        try {
            Path file = tokenFile();
            if (Files.isRegularFile(file)) {
                String existing = Files.readString(file, StandardCharsets.UTF_8).trim();
                if (!existing.isBlank()) {
                    return existing;
                }
            }
            byte[] bytes = new byte[16];
            new SecureRandom().nextBytes(bytes);
            String generated = HexFormat.of().formatHex(bytes);
            Files.writeString(file, generated, StandardCharsets.UTF_8);
            return generated;
        } catch (Exception e) {
            LogService.warn("GpuAuth", "Could not persist GPU token: " + e.getMessage());
            return Long.toHexString(System.nanoTime());
        }
    }

    /** Package-visible for tests. */
    synchronized void resetForTests() {
        resolved = false;
        token = "";
        enabled = true;
    }
}
