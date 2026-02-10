package com.example.automl_prototype_1.service;

import com.example.automl_prototype_1.service.agentic.LocalLlmService;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.nio.file.StandardOpenOption;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.Arrays;
import java.util.Comparator;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * Manages run history, ensuring only the last 10 logs are kept.
 * Orchestrates Offline LLM summarization of completed runs.
 */
public class HistoryService {

    private static final HistoryService instance = new HistoryService();
    private static final int MAX_HISTORY_FILES = 10;
    private static final String HISTORY_DIR = "run_history";
    private static final String SUMMARY_FILE = "run_history/summary.md";

    // Executor for async summarization so we don't block the UI
    private final ExecutorService executor = Executors.newSingleThreadExecutor();

    public static HistoryService getInstance() {
        return instance;
    }

    private HistoryService() {
        File dir = new File(HISTORY_DIR);
        if (!dir.exists()) {
            dir.mkdirs();
        }
    }

    /**
     * Called when a job finishes. Archives the log and triggers summarization.
     */
    public void archiveCurrentRun(String algoName, double bestScore) {
        executor.submit(() -> {
            try {
                // 1. Archive execution.log
                File currentLog = new File("execution.log");
                if (!currentLog.exists())
                    return;

                String timestamp = LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyyMMdd_HHmmss"));
                String archiveName = String.format("run_%s_%s.log", timestamp, algoName.replaceAll(" ", "_"));
                File archiveFile = new File(HISTORY_DIR, archiveName);

                Files.copy(currentLog.toPath(), archiveFile.toPath(), StandardCopyOption.REPLACE_EXISTING);
                LogService.info("History", "Archived run to " + archiveFile.getName());

                // 2. Rotate Old Files
                rotateHistory();

                // 3. Smart Summarization (Offline LLM)
                summarizeRun(archiveFile, algoName, bestScore);

            } catch (Exception e) {
                LogService.error("History", "Failed to archive run: " + e.getMessage());
            }
        });
    }

    private void rotateHistory() {
        File dir = new File(HISTORY_DIR);
        File[] files = dir.listFiles((d, name) -> name.endsWith(".log"));

        if (files != null && files.length > MAX_HISTORY_FILES) {
            Arrays.sort(files, Comparator.comparingLong(File::lastModified));
            // Delete oldest until we are back to limit
            for (int i = 0; i < files.length - MAX_HISTORY_FILES; i++) {
                files[i].delete();
                LogService.info("History", "Deleted old log: " + files[i].getName());
            }
        }
    }

    private void summarizeRun(File logFile, String algoName, double score) {
        LocalLlmService llm = new LocalLlmService();
        if (!llm.isAvailable()) {
            LogService.warn("History", "Local LLM offline. Skipping smart summary.");
            return;
        }

        try {
            // Read the last 50 lines of the log for context (avoid reading 10MB file)
            String tail = readLastLines(logFile, 50);

            String prompt = String.format(
                    "Analyze this execution log for '%s' which achieved score %.4f. " +
                            "Summarize key events, errors, and performance details in 3 bullet points. " +
                            "Log Tail:\n%s",
                    algoName, score, tail);

            String summary = llm.queryOllama(prompt, "active");
            if (summary != null) {
                appendSummary(algoName, score, summary);
            }

        } catch (Exception e) {
            LogService.error("History", "Summarization failed: " + e.getMessage());
        }
    }

    private void appendSummary(String algoName, double score, String summary) {
        try {
            String entry = String.format(
                    "## Run: %s\n**Date:** %s\n**Score:** %.4f\n\n%s\n\n---\n",
                    algoName, LocalDateTime.now().toString(), score, summary);
            Files.writeString(Path.of(SUMMARY_FILE), entry, StandardOpenOption.CREATE, StandardOpenOption.APPEND);
            LogService.info("History", "Smart summary appended to history.");
        } catch (IOException e) {
            LogService.error("History", "Failed to write summary: " + e.getMessage());
        }
    }

    private String readLastLines(File file, int numLines) throws IOException {
        java.util.List<String> lines = Files.readAllLines(file.toPath());
        int start = Math.max(0, lines.size() - numLines);
        StringBuilder sb = new StringBuilder();
        for (int i = start; i < lines.size(); i++) {
            sb.append(lines.get(i)).append("\n");
        }
        return sb.toString();
    }
}
