package com.example.automl_prototype_1.service.agentic;

import com.example.automl_prototype_1.model.SystemProcess;
import com.example.automl_prototype_1.service.LogService;
import com.example.automl_prototype_1.service.OllamaService;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Agentic OS Optimizer Service.
 * Scans Windows processes, filters out non-essential heavy processes,
 * uses LLM to explain the impact, and surgically kills them upon user approval.
 */
public class SystemOptimizerService {

    // Hardcoded Ironclad Blacklist of Windows System Processes that should NEVER be
    // killed
    private static final List<String> BLACKLIST = Arrays.asList(
            "system", "system idle process", "smss.exe", "csrss.exe", "wininit.exe",
            "winlogon.exe", "services.exe", "lsass.exe", "svchost.exe", "explorer.exe",
            "spoolsv.exe", "dwm.exe", "taskmgr.exe", "conhost.exe", "fontdrvhost.exe",
            "sihost.exe", "taskhostw.exe", "searchui.exe", "searchapp.exe", "cmd.exe",
            "powershell.exe", "java.exe", "javaw.exe", "ollama.exe", "python.exe" // Protect ourselves and our backend
    );

    private final OllamaService llmService;

    public SystemOptimizerService() {
        this.llmService = OllamaService.getInstance();
    }

    /**
     * Executes Windows tasklist and returns processes using more than given MB
     * threshold,
     * ignoring anything in the safe blacklist.
     */
    public List<SystemProcess> getHeavyBackgroundProcesses(int thresholdMb) {
        List<SystemProcess> heavyProcesses = new ArrayList<>();

        try {
            // Run tasklist, output as CSV with no headers (NH)
            ProcessBuilder pb = new ProcessBuilder("tasklist", "/FO", "CSV", "/NH");
            Process process = pb.start();

            BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()));
            String line;

            while ((line = reader.readLine()) != null) {
                // Example CSV line: "chrome.exe","1234","Console","1","512,123 K"
                String[] parts = line.split("\",\"");
                if (parts.length >= 5) {
                    String procName = parts[0].replace("\"", "").trim();
                    String pid = parts[1].replace("\"", "").trim();
                    String memStrRaw = parts[4].replace("\"", "").replace("K", "").replace(",", "").trim();

                    try {
                        long memKb = Long.parseLong(memStrRaw);
                        long memMb = memKb / 1024;

                        if (memMb > thresholdMb && !isBlacklisted(procName)) {
                            // Filter out duplicates (some apps spawn many sub-processes, e.g. chrome)
                            // To keep list clean, we might just list the massive individual PIDs
                            heavyProcesses.add(new SystemProcess(procName, pid, memMb + " MB", memMb));
                        }
                    } catch (NumberFormatException nfe) {
                        // ignore unparseable memory values
                    }
                }
            }
            process.waitFor();

        } catch (Exception e) {
            LogService.error("SystemOptimizer", "Failed to fetch tasklist: " + e.getMessage());
        }

        // Sort by heaviest memory usage first
        heavyProcesses.sort((p1, p2) -> Long.compare(p2.getMemoryUsageMb(), p1.getMemoryUsageMb()));

        // Cap to top 15 results to not overwhelm user or LLM
        if (heavyProcesses.size() > 15) {
            return heavyProcesses.subList(0, 15);
        }
        return heavyProcesses;
    }

    private boolean isBlacklisted(String procName) {
        String lowerName = procName.toLowerCase();
        for (String blackItem : BLACKLIST) {
            if (lowerName.equals(blackItem)) {
                return true;
            }
        }
        // Also protect any python/java stuff generically
        if (lowerName.contains("idea") || lowerName.contains("studio"))
            return true;
        return false;
    }

    /**
     * Agentic Call: Asks the LLM to explain the purpose and impact of killing the
     * process.
     */
    public void analyzeProcessImpact(SystemProcess process) {
        if (!llmService.isAvailable()) {
            process.setAiAnalysis(
                    "AI Offline. " + process.getName() + " is using " + process.getMemoryUsageMb() + "MB.");
            return;
        }

        String prompt = "You are a Windows OS optimizer assistant helping a user free up RAM for Machine Learning tasks. "
                +
                "The process '" + process.getName() + "' is currently using " + process.getMemoryUsageMb()
                + " MB of RAM. " +
                "In EXACTLY one short sentence, explain what this program likely is, and what happens if the user force-quits it. "
                +
                "Do not use markdown, be very concise.";

        String analysis = llmService.queryOllama(prompt, "active");

        if (analysis != null && !analysis.isBlank()) {
            process.setAiAnalysis(analysis.replace("\n", "").trim());
        } else {
            process.setAiAnalysis("Unknown background application.");
        }
    }

    /**
     * Executes the taskkill command on approved PIDs.
     */
    public boolean killProcesses(List<String> pids) {
        if (pids == null || pids.isEmpty())
            return true;

        try {
            List<String> command = new ArrayList<>();
            command.add("taskkill");
            command.add("/F"); // Force kill

            for (String pid : pids) {
                command.add("/PID");
                command.add(pid);
            }

            ProcessBuilder pb = new ProcessBuilder(command);
            Process process = pb.start();
            int exitCode = process.waitFor();

            if (exitCode == 0 || exitCode == 128) { // 128 means no process found, treated as success if it already died
                LogService.info("SystemOptimizer", "Successfully killed PIDs: " + pids);
                return true;
            } else {
                LogService.error("SystemOptimizer", "Taskkill failed with exit code: " + exitCode);
                return false;
            }
        } catch (Exception e) {
            LogService.error("SystemOptimizer", "Failed to execute taskkill: " + e.getMessage());
            return false;
        }
    }
}
