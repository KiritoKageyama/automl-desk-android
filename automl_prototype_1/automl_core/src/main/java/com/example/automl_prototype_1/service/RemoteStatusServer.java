package com.example.automl_prototype_1.service;

import com.example.automl_prototype_1.gpu.PythonExecutionService;
import com.example.automl_prototype_1.model.progress.AlgoProgress;
import com.example.automl_prototype_1.model.progress.OrchestratorState;
import com.example.automl_prototype_1.service.progress.ProgressTrackingService;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.sun.net.httpserver.Headers;
import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpServer;

import java.io.IOException;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.InetSocketAddress;
import java.net.URI;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.SecureRandom;
import java.util.ArrayList;
import java.util.HexFormat;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicReference;

/**
 * Lightweight LAN HTTP server so the Android companion can poll desktop job status.
 * Default: http://0.0.0.0:8088/api/status
 * Auth: require header {@code X-AutoML-Token} (or {@code ?token=}) matching the desktop token.
 * Disable with env {@code AUTOML_REMOTE_AUTH=false} (not recommended on LAN).
 */
public class RemoteStatusServer {

    public static final int DEFAULT_PORT = 8088;
    public static final String TOKEN_HEADER = "X-AutoML-Token";

    private static RemoteStatusServer instance;

    private final ObjectMapper mapper = new ObjectMapper();
    private final AtomicReference<Map<String, Object>> cachedGpuHealth = new AtomicReference<>(Map.of());
    private volatile long lastGpuProbeMs = 0;

    private HttpServer server;
    private int port = DEFAULT_PORT;
    private String authToken = "";
    private boolean authEnabled = true;

    private RemoteStatusServer() {
    }

    public static synchronized RemoteStatusServer getInstance() {
        if (instance == null) {
            instance = new RemoteStatusServer();
        }
        return instance;
    }

    public synchronized void start() {
        start(DEFAULT_PORT);
    }

    public synchronized void start(int port) {
        if (server != null) {
            return;
        }
        this.port = port;
        this.authEnabled = !"false".equalsIgnoreCase(System.getenv().getOrDefault("AUTOML_REMOTE_AUTH", "true"));
        this.authToken = resolveOrCreateToken();
        try {
            String bind = System.getenv().getOrDefault("AUTOML_REMOTE_BIND", "0.0.0.0");
            server = HttpServer.create(new InetSocketAddress(bind, port), 0);
            server.createContext("/api/status", this::handleStatus);
            server.createContext("/api/health", this::handleHealth);
            server.createContext("/", this::handleRoot);
            server.setExecutor(Executors.newCachedThreadPool(r -> {
                Thread t = new Thread(r, "RemoteStatusServer");
                t.setDaemon(true);
                return t;
            }));
            server.start();
            LogService.info("RemoteStatus", "Listening on " + bind + ":" + port + " (GET /api/status)"
                    + (authEnabled ? " [token auth ON]" : " [token auth OFF]"));
            System.out.println("[RemoteStatus] Android monitor URL: http://<this-pc-lan-ip>:" + port + "/api/status");
            if (authEnabled) {
                System.out.println("[RemoteStatus] Token (header X-AutoML-Token): " + authToken);
                System.out.println("[RemoteStatus] Token file: " + tokenFilePath());
            }
        } catch (IOException e) {
            LogService.error("RemoteStatus", "Failed to start on port " + port + ": " + e.getMessage());
            server = null;
        }
    }

    public synchronized void stop() {
        if (server != null) {
            server.stop(0);
            server = null;
            LogService.info("RemoteStatus", "Stopped.");
        }
    }

    public boolean isRunning() {
        return server != null;
    }

    public int getPort() {
        return port;
    }

    public String getAuthToken() {
        return authToken;
    }

    public boolean isAuthEnabled() {
        return authEnabled;
    }

    private Path tokenFilePath() {
        return Path.of(System.getProperty("user.home"), ".automl_remote_token");
    }

    private String resolveOrCreateToken() {
        String env = System.getenv("AUTOML_REMOTE_TOKEN");
        if (env != null && !env.isBlank()) {
            return env.trim();
        }
        try {
            Path file = tokenFilePath();
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
            LogService.warn("RemoteStatus", "Could not persist token file: " + e.getMessage());
            return Long.toHexString(System.nanoTime()) + Long.toHexString(System.currentTimeMillis());
        }
    }

    private boolean authorize(HttpExchange exchange) throws IOException {
        if (!authEnabled) {
            return true;
        }
        String header = exchange.getRequestHeaders().getFirst(TOKEN_HEADER);
        if (header == null || header.isBlank()) {
            header = exchange.getRequestHeaders().getFirst("Authorization");
            if (header != null && header.regionMatches(true, 0, "Bearer ", 0, 7)) {
                header = header.substring(7).trim();
            }
        }
        String query = exchange.getRequestURI().getRawQuery();
        String queryToken = null;
        if (query != null) {
            for (String part : query.split("&")) {
                if (part.startsWith("token=")) {
                    queryToken = java.net.URLDecoder.decode(part.substring(6), StandardCharsets.UTF_8);
                    break;
                }
            }
        }
        String provided = (header != null && !header.isBlank()) ? header.trim() : queryToken;
        if (provided != null && provided.equals(authToken)) {
            return true;
        }
        writeJson(exchange, 401, Map.of(
                "error", "Unauthorized",
                "hint", "Send header X-AutoML-Token with the desktop token (see ~/.automl_remote_token)"));
        return false;
    }

    private void handleRoot(HttpExchange exchange) throws IOException {
        if ("OPTIONS".equalsIgnoreCase(exchange.getRequestMethod())) {
            writeCorsPreflight(exchange);
            return;
        }
        Map<String, Object> body = new LinkedHashMap<>();
        body.put("service", "AutoML Remote Status");
        body.put("endpoints", List.of("/api/status", "/api/health"));
        body.put("port", port);
        body.put("authRequired", authEnabled);
        writeJson(exchange, 200, body);
    }

    private void handleHealth(HttpExchange exchange) throws IOException {
        if ("OPTIONS".equalsIgnoreCase(exchange.getRequestMethod())) {
            writeCorsPreflight(exchange);
            return;
        }
        // Health stays open so Android can discover the host before entering a token.
        Map<String, Object> body = new LinkedHashMap<>();
        body.put("ok", true);
        body.put("service", "RemoteStatusServer");
        body.put("port", port);
        body.put("authRequired", authEnabled);
        writeJson(exchange, 200, body);
    }

    private void handleStatus(HttpExchange exchange) throws IOException {
        if ("OPTIONS".equalsIgnoreCase(exchange.getRequestMethod())) {
            writeCorsPreflight(exchange);
            return;
        }
        if (!"GET".equalsIgnoreCase(exchange.getRequestMethod())
                && !"HEAD".equalsIgnoreCase(exchange.getRequestMethod())) {
            exchange.getResponseHeaders().add("Allow", "GET, HEAD");
            writeJson(exchange, 405, Map.of("error", "Method not allowed"));
            return;
        }
        if (!authorize(exchange)) {
            return;
        }

        OrchestratorState snap = ProgressTrackingService.getInstance().snapshot();
        AppStateService app = AppStateService.getInstance();

        AlgoProgress current = snap.currentRun();
        String runningAlgo = current != null ? current.algoId() : null;
        String phase = current != null && current.phase() != null ? current.phase().name() : "IDLE";

        double iterPct = 0.0;
        double trialPct = 0.0;
        if (current != null) {
            if (current.totalIters() > 0) {
                iterPct = 100.0 * current.currentIter() / current.totalIters();
            }
            if (current.totalTrials() > 0) {
                trialPct = 100.0 * current.currentTrial() / current.totalTrials();
            }
        }
        double progressPct = current != null ? Math.min(100.0, Math.max(iterPct, trialPct * 0.25)) : 0.0;

        String datasetName = null;
        try {
            if (app.getSelectedDatasetFile() != null) {
                datasetName = app.getSelectedDatasetFile().getName();
            }
        } catch (Exception ignored) {
        }

        List<String> queue = new ArrayList<>();
        if (snap.queue() != null) {
            for (AlgoProgress q : snap.queue()) {
                queue.add(q.algoId());
            }
        }

        Map<String, Object> currentRun = null;
        if (current != null) {
            currentRun = new LinkedHashMap<>();
            currentRun.put("algoId", current.algoId());
            currentRun.put("phase", phase);
            currentRun.put("currentIter", current.currentIter());
            currentRun.put("totalIters", current.totalIters());
            currentRun.put("currentTrial", current.currentTrial());
            currentRun.put("totalTrials", current.totalTrials());
            currentRun.put("bestScore", sanitizeScore(current.bestScore()));
        }

        boolean gpuUp = false;
        try {
            gpuUp = PythonExecutionService.getInstance().isServerRunning();
        } catch (Exception ignored) {
        }

        Map<String, Object> body = new LinkedHashMap<>();
        body.put("timestamp", System.currentTimeMillis());
        body.put("datasetName", datasetName);
        body.put("domain", app.getCurrentDomain() != null ? app.getCurrentDomain().getDisplayName() : null);
        body.put("runningAlgo", runningAlgo);
        body.put("phase", phase);
        body.put("progressPct", Math.round(progressPct * 10.0) / 10.0);
        body.put("accuracySoFar", current != null ? sanitizeScore(current.bestScore()) : null);
        body.put("globalBestAlgo", snap.globalBestAlgo());
        body.put("globalBestScore", sanitizeScore(snap.globalBestScore()));
        body.put("globalBestTrial", snap.globalBestTrial());
        body.put("globalBestIter", snap.globalBestIter());
        body.put("queue", queue);
        body.put("currentRun", currentRun);
        body.put("useGpu", app.isUseGpu());
        body.put("gpuName", app.getSelectedGpuName());
        body.put("gpuServerUp", gpuUp);
        body.put("gpuHealth", probeGpuHealth(gpuUp));
        body.put("watchingJob",
                com.example.automl_prototype_1.service.agentic.TrainingWatchdogService.getInstance().getCurrentJob());

        writeJson(exchange, 200, body);
    }

    private static Double sanitizeScore(double score) {
        if (Double.isInfinite(score) || Double.isNaN(score) || score < -1e100) {
            return null;
        }
        return Math.round(score * 10000.0) / 10000.0;
    }

    private Map<String, Object> probeGpuHealth(boolean gpuUp) {
        if (!gpuUp) {
            return Map.of("status", "down");
        }
        long now = System.currentTimeMillis();
        if (now - lastGpuProbeMs < 5000 && !cachedGpuHealth.get().isEmpty()) {
            return cachedGpuHealth.get();
        }
        lastGpuProbeMs = now;
        try {
            HttpURLConnection con = (HttpURLConnection) URI.create("http://127.0.0.1:5006/health").toURL()
                    .openConnection();
            con.setConnectTimeout(800);
            con.setReadTimeout(1500);
            con.setRequestMethod("GET");
            if (con.getResponseCode() == 200) {
                @SuppressWarnings("unchecked")
                Map<String, Object> health = mapper.readValue(con.getInputStream(), Map.class);
                cachedGpuHealth.set(health);
                return health;
            }
        } catch (Exception e) {
            Map<String, Object> fallback = Map.of("status", "unreachable", "error", e.getMessage());
            cachedGpuHealth.set(fallback);
            return fallback;
        }
        return Map.of("status", "error");
    }

    private void writeCorsPreflight(HttpExchange exchange) throws IOException {
        Headers headers = exchange.getResponseHeaders();
        headers.set("Access-Control-Allow-Origin", "*");
        headers.set("Access-Control-Allow-Headers", "Content-Type, " + TOKEN_HEADER + ", Authorization");
        headers.set("Access-Control-Allow-Methods", "GET, HEAD, OPTIONS");
        exchange.sendResponseHeaders(204, -1);
        exchange.close();
    }

    private void writeJson(HttpExchange exchange, int code, Object body) throws IOException {
        byte[] bytes = mapper.writeValueAsBytes(body);
        Headers headers = exchange.getResponseHeaders();
        headers.set("Content-Type", "application/json; charset=utf-8");
        headers.set("Access-Control-Allow-Origin", "*");
        headers.set("Access-Control-Allow-Headers", "Content-Type, " + TOKEN_HEADER + ", Authorization");
        headers.set("Cache-Control", "no-store");
        exchange.sendResponseHeaders(code, bytes.length);
        if (!"HEAD".equalsIgnoreCase(exchange.getRequestMethod())) {
            try (OutputStream os = exchange.getResponseBody()) {
                os.write(bytes);
            }
        } else {
            exchange.close();
        }
    }
}
