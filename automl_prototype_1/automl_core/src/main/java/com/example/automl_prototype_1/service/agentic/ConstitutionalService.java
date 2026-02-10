package com.example.automl_prototype_1.service.agentic;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.example.automl_prototype_1.service.LogService;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

/**
 * Manages the "Constitution" of the AI - a set of dynamic, editable rules
 * that govern algorithm selection to prevent failures and ensure best
 * practices.
 */
public class ConstitutionalService {

    private static final String CONSTITUTION_FILE = "constitution.json";
    private final Path configPath;
    private final ObjectMapper mapper;
    private List<Rule> rules;

    private final SystemMonitorService systemMonitor;
    private boolean strictMode = false; // "Absolute Rules Abider" mode

    public ConstitutionalService() {
        this.mapper = new ObjectMapper();
        this.configPath = Paths.get(System.getProperty("user.dir"), "automl_core", "src", "main", "resources",
                CONSTITUTION_FILE);
        this.systemMonitor = new SystemMonitorService();
        // Fallback for production/runtime if src doesn't exist (e.g. running from jar)
        // For prototype, we prefer src to allow persistence across rebuilds.

        loadRules();
    }

    public void setStrictMode(boolean strict) {
        this.strictMode = strict;
        LogService.info("ConstitutionalService", "Strict Mode set to: " + strict);
    }

    public boolean isStrictMode() {
        return strictMode;
    }

    /**
     * Loads rules from JSON file.
     */
    private void loadRules() {
        try {
            if (Files.exists(configPath)) {
                rules = mapper.readValue(configPath.toFile(), new TypeReference<List<Rule>>() {
                });
                LogService.info("ConstitutionalService", "Loaded " + rules.size() + " rules from " + configPath);
            } else {
                // Load default from classpath if file missing
                try (InputStream is = getClass().getClassLoader().getResourceAsStream(CONSTITUTION_FILE)) {
                    if (is != null) {
                        rules = mapper.readValue(is, new TypeReference<List<Rule>>() {
                        });
                        saveRules(); // Persist to disk
                        LogService.info("ConstitutionalService", "Initialized default constitution.");
                    } else {
                        rules = new ArrayList<>();
                        LogService.warn("ConstitutionalService", "No constitution found. Starting empty.");
                    }
                }
            }
        } catch (Exception e) {
            LogService.error("ConstitutionalService", "Failed to load constitution: " + e.getMessage());
            rules = new ArrayList<>();
        }
    }

    private void saveRules() {
        try {
            Files.createDirectories(configPath.getParent());
            mapper.writerWithDefaultPrettyPrinter().writeValue(configPath.toFile(), rules);
        } catch (IOException e) {
            LogService.error("ConstitutionalService", "Failed to save constitution: " + e.getMessage());
        }
    }

    /**
     * Evaluates a specific scenario against the Constitution.
     */
    public Verdict evaluate(MetaFeatures meta, String algorithm, String classifier) {
        for (Rule rule : rules) {
            if (checkCondition(rule.condition, meta, algorithm, classifier)) {
                if ("VETO".equalsIgnoreCase(rule.verdict)) {
                    return Verdict.veto(rule.reason);
                }
                if ("WARN".equalsIgnoreCase(rule.verdict)) {
                    // In STRICT MODE, Warnings can become Vetoes if configured,
                    // or we just respect VETO more aggressively.
                    // For now, Strict Mode implies we might VETO even on strong warnings
                    // if they relate to Resources.
                    if (strictMode && rule.warning.contains("System Freeze")) {
                        return Verdict.veto("[STRICT] " + rule.warning);
                    }
                    return Verdict.warn(rule.warning);
                }
            }
        }
        return Verdict.allow();
    }

    /**
     * Primitive Expression Evaluator for Rules.
     * Supports: &&, ||, >, <, contains, !, ==
     * Variables: meta.numInstances, meta.numFeatures, meta.sparsity,
     * meta.maxSkewness, classifier, algorithm
     * New Variables: system.freeRam, system.totalRam, system.cpuLoad, system.hasGpu
     */
    private boolean checkCondition(String condition, MetaFeatures meta, String algorithm, String classifier) {
        try {
            // Replace Variables with values
            String eval = condition
                    .replace("meta.numInstances", String.valueOf(meta.getNumInstances()))
                    .replace("meta.numFeatures", String.valueOf(meta.getNumFeatures()))
                    .replace("meta.sparsity", String.valueOf(meta.getSparsity()))
                    .replace("meta.maxSkewness", String.valueOf(meta.getMaxSkewness()))
                    .replace("algorithm", "'" + algorithm + "'")
                    .replace("classifier", "'" + classifier + "'")
                    // System Context
                    .replace("system.freeRam", String.valueOf(systemMonitor.getFreeRamGb()))
                    .replace("system.totalRam", String.valueOf(systemMonitor.getTotalRamGb()))
                    .replace("system.cpuLoad", String.valueOf(systemMonitor.getCpuLoad()))
                    .replace("system.hasGpu", String.valueOf(systemMonitor.hasGpu()));

            // Very simple parser logic (Prototype level)
            // We split by AND / OR and evaluate atoms
            // WARNING: This is a simplified evaluator for the specific rules we generated.
            // It is not a full boolean algebra parser.

            // Handle simple AND logic (which covers all our current rules)
            if (eval.contains("&&")) {
                String[] parts = eval.split("&&");
                for (String part : parts) {
                    if (!evaluateAtom(part.trim()))
                        return false;
                }
                return true;
            }

            return evaluateAtom(eval);

        } catch (Exception e) {
            LogService.error("ConstitutionalService", "Rule Eval Error: " + e.getMessage());
            return false;
        }
    }

    private boolean evaluateAtom(String atom) {
        atom = atom.trim();
        // Remove parens
        if (atom.startsWith("(") && atom.endsWith(")")) {
            atom = atom.substring(1, atom.length() - 1);
        }

        // OR logic
        if (atom.contains("||")) {
            String[] parts = atom.split("\\|\\|");
            for (String part : parts) {
                if (evaluateAtom(part.trim()))
                    return true;
            }
            return false;
        }

        // Numeric Checks
        if (atom.contains(">")) {
            String[] p = atom.split(">");
            return Double.parseDouble(p[0].trim()) > Double.parseDouble(p[1].trim());
        }
        if (atom.contains("<")) {
            String[] p = atom.split("<");
            return Double.parseDouble(p[0].trim()) < Double.parseDouble(p[1].trim());
        }

        // String Checks
        if (atom.contains("contains")) {
            // Format: 'String'.contains('Sub') or ! contains
            boolean not = atom.startsWith("!");
            String core = atom.replace("!", "").trim();
            String[] p = core.split("\\.contains");
            String target = p[0].replace("'", "").trim();
            String query = p[1].replace("(", "").replace(")", "").replace("'", "").trim();
            boolean result = target.contains(query);
            return not ? !result : result;
        }

        // Equality
        if (atom.contains("==")) {
            String[] p = atom.split("==");
            String left = p[0].trim().replace("'", "");
            String right = p[1].trim().replace("'", "");
            return left.equals(right);
        }

        return false;
    }

    // --- Dynamic Editing ---

    public void addRule(String id, String condition, String verdict, String reasonOrWarn) {
        Rule r = new Rule();
        r.id = id;
        r.condition = condition;
        r.verdict = verdict;
        if ("VETO".equalsIgnoreCase(verdict))
            r.reason = reasonOrWarn;
        else
            r.warning = reasonOrWarn;

        // Remove existing with same ID
        rules.removeIf(Existing -> Existing.id.equals(id));
        rules.add(r);
        saveRules();
    }

    public static class Rule {
        public String id;
        public String condition;
        public String verdict; // VETO, WARN
        public String reason;
        public String warning;
    }

    public static class Verdict {
        public boolean allowed;
        public String reason;
        public String warning;

        static Verdict veto(String r) {
            Verdict v = new Verdict();
            v.allowed = false;
            v.reason = r;
            return v;
        }

        static Verdict warn(String w) {
            Verdict v = new Verdict();
            v.allowed = true;
            v.warning = w;
            return v;
        }

        static Verdict allow() {
            Verdict v = new Verdict();
            v.allowed = true;
            return v;
        }
    }
}
