package com.example.automl_prototype_1.gpu;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;

import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.TimeUnit;
import com.example.automl_prototype_1.service.LogService;
import com.example.automl_prototype_1.service.AppStateService;
import com.example.automl_prototype_1.service.TelemetryService;

/**
 * Manages micro-batching of GPU tasks to the persistent Python server.
 */
public class GpuBatchService {

    private static GpuBatchService instance;
    private final BlockingQueue<GpuJob> jobQueue;
    private final Thread dispatcherThread;
    private volatile boolean running = true;
    private final ObjectMapper mapper = new ObjectMapper();

    // Batch Settings
    // Batch Settings
    private static final int MAX_BATCH_SIZE = 500; // Keep large max batch
    private static final long BATCH_WINDOW_MS = 10; // Reduced to 10ms to avoid latency penalty for small batches
    private static final String SERVER_URL = "http://127.0.0.1:5006";
    private static final long BYPASS_JSON_THRESHOLD_BYTES = 1024L; // 1KB - Force Arrow for almost everything

    private final ITrainingDataSerializer arrowSerializer = new ArrowTrainingDataSerializer();

    private GpuBatchService() {
        this.jobQueue = new LinkedBlockingQueue<>();
        this.dispatcherThread = new Thread(this::dispatchLoop, "GpuBatchDispatcher");
        this.dispatcherThread.setDaemon(true);
        this.dispatcherThread.start();
        LogService.info("GpuBatchService", "Started. Waiting for jobs...");
    }

    public static synchronized GpuBatchService getInstance() {
        if (instance == null) {
            instance = new GpuBatchService();
            // Async check capabilities on startup
            new Thread(instance::checkCapabilities).start();
        }
        return instance;
    }

    public boolean isQueueEmpty() {
        return jobQueue.isEmpty();
    }

    private void checkCapabilities() {
        // Delegate to PythonExecutionService for server management
        if (!PythonExecutionService.getInstance().isServerRunning()) {
            // Try to start it if we need it
            LogService.info("GpuBatchService", "GPU Server not running. Requesting start...");
            PythonExecutionService.getInstance().startGpuServer();

            // Wait for it to spin up
            try {
                Thread.sleep(2000);
            } catch (Exception e) {
            }
        }

        try {
            // Construct minimalist PING
            ObjectNode request = mapper.createObjectNode();
            request.put("action", "PING");

            String responseStr = sendPost(SERVER_URL, request.toString());
            com.fasterxml.jackson.databind.JsonNode node = mapper.readTree(responseStr);

            if (node.has("backends")) {
                com.fasterxml.jackson.databind.JsonNode backends = node.get("backends");
                String rf = backends.get("rf").asText();
                String svm = backends.get("svm").asText();
                String xgb = backends.get("xgb").asText();

                LogService.info("GpuStatus", "=== GPU SERVER CAPABILITIES ===");
                LogService.info("GpuStatus", "Random Forest Backend: " + rf.toUpperCase()
                        + (rf.equals("cuml") ? " (GPU)" : " (CPU Bridge)"));
                LogService.info("GpuStatus",
                        "SVM Backend: " + svm.toUpperCase() + (svm.equals("cuml") ? " (GPU)" : " (CPU Bridge)"));
                LogService.info("GpuStatus", "XGBoost Backend: " + xgb.toUpperCase()
                        + (xgb.equals("installed") ? " (GPU Enabled)" : " (Not Found/CPU)"));
                LogService.info("GpuStatus", "===============================");

                if (rf.equals("sklearn")) {
                    LogService.info("GpuStatus", "NOTICE: cuML not found. Random Forest is running on CPU via Python.");
                }
            }
        } catch (Exception e) {
            LogService.warn("GpuStatus",
                    "Detailed GPU capabilities check failed (Server might be busy): " + e.getMessage());
        }
    }

    public CompletableFuture<Object> submitTrain(String algoType, double[][] features, int[] labels,
            java.util.Map<String, Object> params) {
        PythonExecutionService.getInstance().recordActivity(); // Keep alive
        GpuJob job = new GpuJob(GpuJob.Type.TRAIN);
        job.algoType = algoType;
        job.features = features;
        job.labels = labels;

        // Force GPU Logic
        boolean forceGpu = AppStateService.getInstance().isForceGpu();
        if (forceGpu) {
            java.util.Map<String, Object> mutableParams = (params != null) ? new java.util.HashMap<>(params)
                    : new java.util.HashMap<>();
            mutableParams.put("force_gpu", true);
            job.params = mutableParams;
            if (forceGpu)
                LogService.info("GpuBatchService", "Forcing GPU for job: " + algoType);
        } else {
            job.params = params;
        }

        jobQueue.offer(job);

        // Record Telemetry
        if (features != null && features.length > 0) {
            TelemetryService.getInstance().recordTransaction(algoType, features.length, features[0].length);
        }

        return job.future;
    }

    public CompletableFuture<Object> submitPredict(String modelId, double[][] features) {
        GpuJob job = new GpuJob(GpuJob.Type.PREDICT);
        job.modelId = modelId;
        job.features = features;
        jobQueue.offer(job);
        return job.future;
    }

    public CompletableFuture<Object> submitCleanup(String modelId) {
        GpuJob job = new GpuJob(GpuJob.Type.CLEANUP);
        job.modelId = modelId;
        jobQueue.offer(job);
        return job.future;
    }

    public CompletableFuture<Object> submitUploadDataset(double[][] features, int[] labels) {
        GpuJob job = new GpuJob(GpuJob.Type.UPLOAD);
        job.features = features;
        job.labels = labels;
        // Priority offer?
        jobQueue.offer(job);
        return job.future;
    }

    public CompletableFuture<Object> submitTrainCached(String algoType, String datasetId, int[] featureIndices,
            int[] rowIndices,
            java.util.Map<String, Object> params) {
        GpuJob job = new GpuJob(GpuJob.Type.TRAIN);
        job.algoType = algoType;
        job.datasetId = datasetId;
        job.featureIndices = featureIndices;
        job.rowIndices = rowIndices;

        // Force GPU Logic
        boolean forceGpu = AppStateService.getInstance().isForceGpu();
        if (forceGpu) {
            java.util.Map<String, Object> mutableParams = (params != null) ? new java.util.HashMap<>(params)
                    : new java.util.HashMap<>();
            mutableParams.put("force_gpu", true);
            job.params = mutableParams;
        } else {
            job.params = params;
        }

        jobQueue.offer(job);
        return job.future;
    }

    public CompletableFuture<Object> submitClearCache() {
        GpuJob job = new GpuJob(GpuJob.Type.CLEAR_CACHE);
        // Priority offer?
        jobQueue.offer(job);
        return job.future;
    }

    public CompletableFuture<Object> submitPredictCached(String modelId, String datasetId, int[] featureIndices,
            int[] rowIndices) {
        GpuJob job = new GpuJob(GpuJob.Type.PREDICT);
        job.modelId = modelId;
        job.datasetId = datasetId;
        job.featureIndices = featureIndices;
        job.rowIndices = rowIndices;
        jobQueue.offer(job);
        return job.future;
    }

    private void dispatchLoop() {
        while (running) {
            try {
                // 1. Collect Batch
                List<GpuJob> batch = new ArrayList<>();
                GpuJob first = jobQueue.poll(1000, TimeUnit.MILLISECONDS); // Wait a bit if empty
                if (first == null)
                    continue;

                batch.add(first);
                long deadline = System.currentTimeMillis() + BATCH_WINDOW_MS;

                while (batch.size() < MAX_BATCH_SIZE && System.currentTimeMillis() < deadline) {
                    GpuJob next = jobQueue.poll(5, TimeUnit.MILLISECONDS); // Quick poll
                    if (next != null) {
                        batch.add(next);
                    } else {
                        // If queue empty temporarily, wait just a tiny bit or break?
                        // break to send what we have for latency
                        break;
                    }
                }

                // 2. Process Batch
                if (!batch.isEmpty()) {
                    // LogService.info("GpuBatchService", "Processing Batch of Size: " +
                    // batch.size());
                    processBatch(batch);
                }

            } catch (InterruptedException e) {
                if (!running)
                    break;
            } catch (Exception e) {
                LogService.error("GpuBatchService", "Dispatcher Error: " + e.getMessage());
                e.printStackTrace();
            }
        }
    }

    private void processBatch(List<GpuJob> batch) {
        PythonExecutionService.getInstance().recordActivity(); // Keep alive during processing
        // Simple strategy: Split batch into Trains and Predicts.

        List<GpuJob> trainJobs = batch.stream().filter(j -> j.type == GpuJob.Type.TRAIN)
                .collect(java.util.stream.Collectors.toList());
        List<GpuJob> predictJobs = batch.stream().filter(j -> j.type == GpuJob.Type.PREDICT).toList();
        List<GpuJob> cleanupJobs = batch.stream().filter(j -> j.type == GpuJob.Type.CLEANUP).toList();
        List<GpuJob> uploadJobs = batch.stream().filter(j -> j.type == GpuJob.Type.UPLOAD)
                .collect(java.util.stream.Collectors.toList());

        // --- Optimized High-Volume Path (Arrow) ---
        // Inspect trainJobs for LARGE payloads
        List<GpuJob> arrowJobs = new ArrayList<>();
        if (!trainJobs.isEmpty()) {
            // Need mutable iterator to remove
            java.util.Iterator<GpuJob> it = trainJobs.iterator();
            while (it.hasNext()) {
                GpuJob job = it.next();
                if (job.features != null && job.features.length > 0) {
                    long sizeBytes = (long) job.features.length * job.features[0].length * 8;
                    if (sizeBytes > BYPASS_JSON_THRESHOLD_BYTES) {
                        arrowJobs.add(job);
                        it.remove(); // Remove from JSON batch
                    }
                }
            }
        }

        // Inspect uploadJobs for LARGE payloads (Arrow)
        List<GpuJob> arrowUploads = new ArrayList<>();
        if (!uploadJobs.isEmpty()) {
            java.util.Iterator<GpuJob> it = uploadJobs.iterator();
            while (it.hasNext()) {
                GpuJob job = it.next();
                if (job.features != null && job.features.length > 0) {
                    long sizeBytes = (long) job.features.length * job.features[0].length * 8;
                    if (sizeBytes > BYPASS_JSON_THRESHOLD_BYTES) {
                        arrowUploads.add(job);
                        it.remove();
                    }
                }
            }
        }

        // Process Arrow Jobs Individually
        for (GpuJob job : arrowJobs) {
            processArrowJob(job);
        }

        // Process Arrow Uploads Individually
        for (GpuJob job : arrowUploads) {
            processArrowUpload(job);
        }

        if (!trainJobs.isEmpty())
            sendBatchRequest(trainJobs, "TRAIN_BATCH");
        if (!predictJobs.isEmpty())
            sendBatchRequest(predictJobs, "PREDICT_BATCH");
        if (!cleanupJobs.isEmpty())
            sendBatchRequest(cleanupJobs, "CLEANUP_BATCH");

        // Uploads must be processed individually to match server expectation of single
        // upload per request logic
        for (GpuJob uploadJob : uploadJobs) {
            sendBatchRequest(java.util.Collections.singletonList(uploadJob), "UPLOAD_DATASET");
        }

        // Clear Cache jobs
        List<GpuJob> clearJobs = batch.stream().filter(j -> j.type == GpuJob.Type.CLEAR_CACHE).toList();
        if (!clearJobs.isEmpty()) {
            sendBatchRequest(clearJobs, "CLEAR_CACHE");
        }
    }

    private void processArrowJob(GpuJob job) {
        try {
            LogService.info("GpuBatchService", "Bypassing JSON for Large Job " + job.id + ". Using Arrow.");
            LogService.info("GpuBatchService", "Bypassing JSON for Large Job " + job.id + ". Using Arrow.");
            SerializedData data = arrowSerializer
                    .serializeTrainingRequest(job.features, job.labels, job.params);

            // Send Headers
            java.util.Map<String, String> headers = new java.util.HashMap<>();
            headers.put("X-Algo-Type", job.algoType);
            if (job.params != null) {
                headers.put("X-Params", mapper.writeValueAsString(job.params));
            }

            // Post to /train_arrow
            String responseStr = sendRequest(SERVER_URL + "/train_arrow", data.getBody(), data.getContentType(),
                    headers);

            // Response should be {"results": [{"id":..., "model_id":..., "success": true}]}
            // Or our server returned that format to match handle_train_batch

            ObjectNode response = (ObjectNode) mapper.readTree(responseStr);
            if (response.has("results")) {
                ArrayNode results = (ArrayNode) response.get("results");
                if (results.size() > 0) {
                    com.fasterxml.jackson.databind.JsonNode rNode = results.get(0);
                    boolean success = rNode.get("success").asBoolean();
                    if (success) {
                        String mId = rNode.get("model_id").asText();
                        if (rNode.has("onnx_path") && !rNode.get("onnx_path").isNull()) {
                            mId += "|" + rNode.get("onnx_path").asText();
                        }
                        job.future.complete(mId);
                    } else {
                        job.future.completeExceptionally(new RuntimeException(
                                rNode.has("error") ? rNode.get("error").asText() : "Unknown Error"));
                    }
                } else {
                    job.future.completeExceptionally(new RuntimeException("Empty results from Arrow Train"));
                }
            } else {
                // Fallback if structure different
                job.future.completeExceptionally(
                        new RuntimeException("Invalid response from Arrow Train: " + responseStr));
            }

        } catch (Exception e) {
            LogService.error("GpuBatchService", "Arrow Train Failed: " + e.getMessage());
            TelemetryService.getInstance().recordError("ArrowTrain");
            job.future.completeExceptionally(e);
        }
    }

    private void processArrowUpload(GpuJob job) {
        try {
            LogService.info("GpuBatchService", "Bypassing JSON for Large Upload " + job.id + ". Using Arrow.");
            // Use same serializer (merges features + labels into Arrow Table with 'target'
            // column)
            SerializedData data = arrowSerializer
                    .serializeTrainingRequest(job.features, job.labels, java.util.Collections.emptyMap());

            // Post to /upload_arrow
            // No headers needed for upload (params not used)
            String responseStr = sendRequest(SERVER_URL + "/upload_arrow", data.getBody(), data.getContentType(),
                    java.util.Collections.emptyMap());

            ObjectNode response = (ObjectNode) mapper.readTree(responseStr);
            if (response.has("id")) {
                String dsId = response.get("id").asText();
                job.future.complete(dsId);
            } else {
                job.future.completeExceptionally(new RuntimeException(
                        response.has("error") ? response.get("error").asText() : "Unknown Upload Error"));
            }

        } catch (Exception e) {
            LogService.error("GpuBatchService", "Arrow Upload Failed: " + e.getMessage());
            TelemetryService.getInstance().recordError("ArrowUpload");
            job.future.completeExceptionally(e);
        }
    }

    private void sendBatchRequest(List<GpuJob> jobs, String action) {
        try {
            ObjectNode request = mapper.createObjectNode();
            request.put("action", action);
            ArrayNode jobArray = request.putArray("jobs");

            for (GpuJob job : jobs) {
                ObjectNode jNode = jobArray.addObject();
                jNode.put("id", job.id);
                if (action.equals("TRAIN_BATCH")) {
                    jNode.put("type", job.algoType);

                    if (job.datasetId != null) {
                        // Cached Mode
                        jNode.put("dataset_id", job.datasetId);
                        if (job.featureIndices != null) {
                            ArrayNode idxs = jNode.putArray("feature_indices");
                            for (int idx : job.featureIndices)
                                idxs.add(idx);
                        }
                    } else {
                        // Legacy Mode (Direct Transfer)
                        ArrayNode feats = jNode.putArray("features");
                        for (double[] row : job.features) {
                            ArrayNode r = feats.addArray();
                            for (double val : row)
                                r.add(val);
                        }
                        ArrayNode labs = jNode.putArray("labels");
                        for (int l : job.labels)
                            labs.add(l);
                    }

                    // Params
                    if (job.params != null) {
                        ObjectNode pNode = jNode.putObject("params");
                        job.params.forEach((k, v) -> pNode.put(k, v.toString()));
                    }
                } else if (action.equals("PREDICT_BATCH")) {
                    // Strip ONNX path if present (format: "uuid|path")
                    String rawId = job.modelId;
                    String cleanId = (rawId != null && rawId.contains("|")) ? rawId.split("\\|")[0] : rawId;
                    jNode.put("model_id", cleanId);

                    if (job.datasetId != null) {
                        // Cached Mode
                        jNode.put("dataset_id", job.datasetId);
                        if (job.rowIndices != null) {
                            ArrayNode rows = jNode.putArray("row_indices");
                            for (int idx : job.rowIndices)
                                rows.add(idx);
                        }
                        if (job.featureIndices != null) {
                            ArrayNode feats = jNode.putArray("feature_indices");
                            for (int idx : job.featureIndices)
                                feats.add(idx);
                        }
                    } else {
                        // Legacy/Direct Mode
                        if (job.features != null) {
                            ArrayNode feats = jNode.putArray("features");
                            for (double[] row : job.features) {
                                ArrayNode r = feats.addArray();
                                for (double val : row)
                                    r.add(val);
                            }
                        }
                    }
                } else if (action.equals("CLEANUP_BATCH")) {
                    // Strip ONNX path if present (format: "uuid|path")
                    String rawId = job.modelId;
                    String cleanId = (rawId != null && rawId.contains("|")) ? rawId.split("\\|")[0] : rawId;
                    jNode.put("model_id", cleanId);
                }
            }

            // Handle UPLOAD_DATASET payload (Single job expected)
            if (action.equals("UPLOAD_DATASET")) {
                GpuJob job = jobs.get(0);
                ArrayNode feats = request.putArray("upload_features");
                for (double[] row : job.features) {
                    ArrayNode r = feats.addArray();
                    for (double val : row)
                        r.add(val);
                }
                ArrayNode labs = request.putArray("upload_labels");
                for (int l : job.labels)
                    labs.add(l);
            }

            // Sync Send with Retry Logic
            String responseStr;
            try {
                responseStr = sendPost(SERVER_URL, request.toString());
            } catch (java.net.ConnectException ce) {
                LogService.warn("GpuBatchService", "Connection refused. Attempting to restart GPU Server...");
                com.example.automl_prototype_1.gpu.PythonExecutionService.getInstance().restartGpuServer();
                try {
                    Thread.sleep(3000); // Wait for restart
                } catch (InterruptedException ie) {
                    Thread.currentThread().interrupt();
                }
                // Retry once
                responseStr = sendPost(SERVER_URL, request.toString());
            }

            // Handle UPLOAD response (which is direct, not in 'results' array usually?
            // Wait, gpu_server return UPLOAD response directly as Dict.
            // But handle_train_batch returns {"results": [...]}.
            // My python code for UPLOAD returned {"id": ..., "status": ...}. It did NOT
            // wrap in "results" unless I updated it.
            // Looking at gpu_server: return {"id": dataset_id, ...}.
            // So we need special handling here.

            ObjectNode response = (ObjectNode) mapper.readTree(responseStr);

            if (action.equals("UPLOAD_DATASET")) {
                GpuJob job = jobs.get(0);
                if (response.has("id")) {
                    job.future.complete(response.get("id").asText());
                } else {
                    job.future.completeExceptionally(new RuntimeException("Upload failed: " + responseStr));
                }
                return;
            }

            if (action.equals("CLEAR_CACHE")) {
                boolean cleared = response.has("status") && response.get("status").asText().equals("cleared");
                for (GpuJob j : jobs) {
                    j.future.complete(cleared);
                }
                return;
            }

            // Standard Batch Response
            if (response.has("results")) {
                ArrayNode results = (ArrayNode) response.get("results");
                for (com.fasterxml.jackson.databind.JsonNode rNode : results) {
                    String jId = rNode.get("id").asText();
                    boolean success = rNode.get("success").asBoolean();

                    GpuJob job = jobs.stream().filter(j -> j.id.equals(jId)).findFirst().orElse(null);
                    if (job != null) {
                        if (success) {
                            if (action.equals("TRAIN_BATCH")) {
                                String mId = rNode.get("model_id").asText();
                                if (rNode.has("onnx_path") && !rNode.get("onnx_path").isNull()) {
                                    mId += "|" + rNode.get("onnx_path").asText();
                                }
                                job.future.complete(mId);
                            } else if (action.equals("PREDICT_BATCH")) {
                                // Parse predictions
                                ArrayNode predsNode = (ArrayNode) rNode.get("predictions");
                                int[] preds = new int[predsNode.size()];
                                for (int i = 0; i < predsNode.size(); i++)
                                    preds[i] = predsNode.get(i).asInt();
                                job.future.complete(preds);
                            } else if (action.equals("CLEANUP_BATCH")) {
                                // Cleanup successful
                                job.future.complete(true);
                            }
                        } else {
                            if (job.future != null)
                                job.future.completeExceptionally(new RuntimeException(
                                        rNode.has("error") ? rNode.get("error").asText() : "Unknown Error"));
                        }
                    }
                }
            }

        } catch (Exception e) {
            // Fail all jobs in this sub-batch
            for (GpuJob j : jobs) {
                j.future.completeExceptionally(e);
            }
            LogService.error("GpuBatchService", "Network/Server Failed: " + e.getMessage());
            TelemetryService.getInstance().recordError("BatchProcess");
        }
    }

    private String sendPost(String urlStr, String jsonBody) throws Exception {
        return sendRequest(urlStr, jsonBody.getBytes("utf-8"), "application/json", null);
    }

    private String sendRequest(String urlStr, byte[] body, String contentType, java.util.Map<String, String> headers)
            throws Exception {
        URL url = java.net.URI.create(urlStr).toURL();
        HttpURLConnection conn = (HttpURLConnection) url.openConnection();
        conn.setRequestMethod("POST");
        conn.setRequestProperty("Content-Type", contentType);
        if (headers != null) {
            for (java.util.Map.Entry<String, String> entry : headers.entrySet()) {
                conn.setRequestProperty(entry.getKey(), entry.getValue());
            }
        }
        conn.setFixedLengthStreamingMode(body.length);
        conn.setRequestProperty("Content-Length", String.valueOf(body.length));
        // Disable Expect: 100-continue to prevent deadlock with simple Python server
        conn.setRequestProperty("Expect", "");
        conn.setDoOutput(true);
        conn.setConnectTimeout(30000); // 30s timeout
        conn.setReadTimeout(300000); // 300s (5 min) timeout for processing

        try (OutputStream os = conn.getOutputStream()) {
            os.write(body, 0, body.length);
        }

        int status = conn.getResponseCode();
        java.io.InputStream is = (status >= 400) ? conn.getErrorStream() : conn.getInputStream();

        if (is == null) {
            throw new java.io.IOException("Server returned HTTP " + status + " but no error stream.");
        }

        try (java.io.ByteArrayOutputStream buffer = new java.io.ByteArrayOutputStream()) {
            int nRead;
            byte[] data = new byte[4096];
            while ((nRead = is.read(data, 0, data.length)) != -1) {
                buffer.write(data, 0, nRead);
            }
            String responseStr = new String(buffer.toByteArray(), java.nio.charset.StandardCharsets.UTF_8);

            if (status >= 400) {
                throw new java.io.IOException("Server Error " + status + ": " + responseStr);
            }

            return responseStr;
        }

    }

    public void shutdown() {
        running = false;
        dispatcherThread.interrupt();
    }

    static class GpuJob {

        enum Type {
            TRAIN, PREDICT, CLEANUP, UPLOAD, CLEAR_CACHE
        }

        Type type;
        String id = java.util.UUID.randomUUID().toString();
        CompletableFuture<Object> future = new CompletableFuture<>();

        // Train fields
        String algoType;
        double[][] features;
        int[] labels;
        java.util.Map<String, Object> params;

        // Caching fields
        String datasetId;
        int[] featureIndices;
        int[] rowIndices;

        // Predict/Cleanup fields
        String modelId;

        public GpuJob(Type type) {
            this.type = type;
        }

        // Constructor for compatibility if needed, or just use setters
        public GpuJob(Type type, double[][] features, int[] labels, java.util.Map<String, Object> params,
                String modelId) {
            this.type = type;
            this.features = features;
            this.labels = labels;
            this.params = params;
            this.modelId = modelId;
        }
    }
}
