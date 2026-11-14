package com.example.automl_prototype_1.lca;

import com.example.automl_prototype_1.lca.actor.ActionDispatcher;
import com.example.automl_prototype_1.lca.perception.PerceptionService;
import com.example.automl_prototype_1.service.LogService;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.util.Map;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * The "Brain Stem" of the Agent.
 * Runs the Fast Loop: See -> Think (Python) -> Act.
 */
public class LcaService {

    private static LcaService instance;
    private final ScheduledExecutorService fastLoopExecutor;
    private final AtomicBoolean isRunning = new AtomicBoolean(false);
    private final ObjectMapper mapper = new ObjectMapper();

    // Loop frequency: 10Hz = 100ms
    private static final int LOOP_DELAY_MS = 1000; // Slower for prototype: 1Hz

    private LcaService() {
        this.fastLoopExecutor = Executors.newSingleThreadScheduledExecutor();
    }

    public static synchronized LcaService getInstance() {
        if (instance == null) {
            instance = new LcaService();
        }
        return instance;
    }

    public void startAgent() {
        if (isRunning.getAndSet(true))
            return;

        LogService.info("LCA-Core", "Agent Started.");
        fastLoopExecutor.scheduleWithFixedDelay(this::agentLoopStep, 1000, LOOP_DELAY_MS, TimeUnit.MILLISECONDS);
    }

    public void stopAgent() {
        if (!isRunning.getAndSet(false))
            return;
        LogService.info("LCA-Core", "Agent Paused.");
        // We don't shutdown executor, just stop logic?
        // With scheduleWithFixedDelay, we can't easily cancel next without future.
        // We use the atomic flag to skip execution.
    }

    private void agentLoopStep() {
        if (!isRunning.get())
            return;

        try {
            // 1. Perception
            Map<String, Object> snapshot = PerceptionService.getInstance().captureSnapshot();

            // 2. Think (Call Python)
            Map<String, Object> action = callPythonAgent(snapshot);

            // 3. Act
            if (action != null) {
                ActionDispatcher.getInstance().dispatch(action);

                // 4. Reflexion (Slow Loop trigger would happen here or implicitly in Python)
                // For now, if we had a reward signal, we'd send it back.

                // Notify UI for Mind's Eye visualization
                String base64Image = (String) snapshot.get("screenshot_base64");
                notifyUi(base64Image, action);
            }

        } catch (Exception e) {
            LogService.error("LCA-Core", "Loop Error: " + e.getMessage());
        }
    }

    @SuppressWarnings("unchecked")
    private Map<String, Object> callPythonAgent(Map<String, Object> snapshot) throws Exception {
        URL url = new URL("http://127.0.0.1:5006/"); // GPU Server
        HttpURLConnection con = (HttpURLConnection) url.openConnection();
        con.setRequestMethod("POST");
        con.setRequestProperty("Content-Type", "application/json");
        com.example.automl_prototype_1.gpu.GpuServerAuth auth =
                com.example.automl_prototype_1.gpu.GpuServerAuth.getInstance();
        if (auth.isEnabled() && auth.getToken() != null && !auth.getToken().isBlank()) {
            con.setRequestProperty(com.example.automl_prototype_1.gpu.GpuServerAuth.HEADER, auth.getToken());
        }
        con.setDoOutput(true);

        // Prepare Request
        Map<String, Object> request = new java.util.HashMap<>();
        request.put("action", "AGENT_STEP");
        request.put("snapshot", snapshot);

        String json = mapper.writeValueAsString(request);

        try (OutputStream os = con.getOutputStream()) {
            byte[] input = json.getBytes(StandardCharsets.UTF_8);
            os.write(input, 0, input.length);
        }

        if (con.getResponseCode() == 200) {
            try (BufferedReader br = new BufferedReader(
                    new InputStreamReader(con.getInputStream(), StandardCharsets.UTF_8))) {
                StringBuilder response = new StringBuilder();
                String line;
                while ((line = br.readLine()) != null)
                    response.append(line.trim());

                Map<String, Object> respMap = mapper.readValue(response.toString(), Map.class);
                if (Boolean.TRUE.equals(respMap.get("success"))) {
                    return (Map<String, Object>) respMap.get("action");
                }
            }
        }
        return null;
    }

    // --- UI Observability ---
    public static class AgentState {
        public String screenshotBase64;
        public String thoughtProcess; // JSON string of action/confidence
        public long timestamp;
    }

    private java.util.function.Consumer<AgentState> uiCallback;

    public void setUiCallback(java.util.function.Consumer<AgentState> callback) {
        this.uiCallback = callback;
    }

    private void notifyUi(String base64, Map<String, Object> action) {
        if (uiCallback != null) {
            try {
                AgentState state = new AgentState();
                state.screenshotBase64 = base64;
                state.thoughtProcess = action != null
                        ? mapper.writerWithDefaultPrettyPrinter().writeValueAsString(action)
                        : "thinking...";
                state.timestamp = System.currentTimeMillis();
                uiCallback.accept(state);
            } catch (Exception e) {
                // Ignore UI update errors
            }
        }
    }
}
