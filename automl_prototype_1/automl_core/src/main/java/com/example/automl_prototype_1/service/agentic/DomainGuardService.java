package com.example.automl_prototype_1.service.agentic;

import java.util.*;

/**
 * The Gatekeeper Service.
 * <p>
 * Enforces strict domain boundaries to ensure the AI only processes
 * Computer Science related queries.
 * <p>
 * Strategies:
 * 1. Keyword-Weighted Scoring: Validates prompts against known CS subdomains.
 * 2. Hard Rejection: Blocks queries that do not meet the threshold.
 */
public class DomainGuardService {

    private static final double VALIDATION_THRESHOLD = 1.0; // Minimum score to pass
    private final Map<String, List<String>> domainKeywords;
    private final Set<String> ignoreWords;
    private final Set<String> negativeKeywords;

    public DomainGuardService() {
        this.domainKeywords = new HashMap<>();
        this.ignoreWords = new HashSet<>(Arrays.asList(
                "the", "is", "at", "which", "on", "a", "an", "and", "or", "but", "if", "then", "else", "when", "how",
                "what", "why"));
        this.negativeKeywords = new HashSet<>(Arrays.asList(
                "recipe", "bake", "cook", "food", "cake", "bread",
                "dog", "puppy", "kitten", "bird", "fish", // 'cat' omitted due to 'cat' command
                "flower", "garden", "plant", "nature",
                "sports", "football", "soccer", "baseball", "basketball", "game", "player", "score", // 'score' is
                                                                                                     // risky?
                "movie", "film", "music", "song", "artist",
                "history", "war", "king", "queen", "president", "politics",
                "medical", "doctor", "health", "symptom", "pain", "medicine"));
        initializeKeywords();
    }

    private void initializeKeywords() {
        // 1. ALGORITHMS & DATA STRUCTURES
        domainKeywords.put("ALGORITHMS", Arrays.asList(
                "algorithm", "sort", "search", "graph", "tree", "binary", "heap", "stack", "queue",
                "hashing", "map", "set", "list", "array", "matrix", "recursion", "iteration",
                "complexity", "big-o", "time", "space", "optimization", "dynamic", "greedy",
                "divide", "conquer", "backtracking", "bfs", "dfs", "dijkstra", "astar", "pathfinding"));

        // 2. MACHINE LEARNING & AI
        domainKeywords.put("ML", Arrays.asList(
                "machine", "learning", "ai", "artificial", "intelligence", "neural", "network",
                "deep", "model", "training", "inference", "prediction", "classification", "regression",
                "clustering", "supervised", "unsupervised", "reinforcement", "dataset", "feature",
                "label", "accuracy", "precision", "recall", "f1", "confusion", "matrix", "optimization",
                "gradient", "descent", "loss", "function", "activation", "layer", "neuron", "transformer",
                "llm", "gpt", "bert", "diffusion", "gan", "automl", "genetic", "evolutionary", "swarm",
                "particle", "optimization", "knn", "svm", "random", "forest", "decision", "tree", "xgboost",
                "lightgbm", "catboost", "sklearn", "pytorch", "tensorflow", "keras", "pandas", "numpy"));

        // 3. SYSTEMS & ARCHITECTURE
        domainKeywords.put("SYSTEMS", Arrays.asList(
                "system", "architecture", "design", "pattern", "component", "module", "service",
                "microservice", "monolith", "client", "server", "database", "sql", "nosql", "cache",
                "redis", "kafka", "mq", "message", "queue", "api", "rest", "graphql", "grpc", "soap",
                "http", "tcp", "udp", "ip", "network", "protocol", "distributed", "cloud", "aws",
                "azure", "gcp", "docker", "kubernetes", "container", "virtualization", "os", "kernel",
                "linux", "windows", "process", "thread", "concurrency", "parallelism", "synchronization",
                "lock", "mutex", "semaphore", "monitor", "shell", "bash", "terminal", "cli", "command",
                "select", "insert", "update", "delete", "table", "row", "column", "query", "schema",
                "sudo", "rm", "mkdir", "ssh", "ftp", "grep", "curl", "wget", "ps", "kill", "top"));

        // 4. PROGRAMMING & LANGUAGES
        domainKeywords.put("CODING", Arrays.asList(
                "code", "programming", "language", "java", "python", "c++", "c#", "javascript",
                "typescript", "go", "rust", "swift", "kotlin", "scala", "ruby", "php", "html", "css",
                "sql", "shell", "bash", "script", "function", "class", "object", "interface", "abstract",
                "implementation", "inheritance", "polymorphism", "encapsulation", "variable", "type",
                "int", "string", "boolean", "float", "double", "compiler", "interpreter", "debug",
                "test", "unit", "integration", "mock", "stub", "exception", "error", "log",
                "bug", "issue", "fix", "patch", "git", "commit", "push", "pull", "merge", "branch",
                "repo", "repository", "clone", "diff", "version", "control", "ide", "editor", "syntax"));

        // 5. SECURITY
        domainKeywords.put("SECURITY", Arrays.asList(
                "security", "encryption", "decryption", "hashing", "cryptography", "auth", "authentication",
                "authorization", "exploit", "vulnerability", "attack", "defense", "firewall", "token", "jwt",
                "oauth", "sso", "audit", "compliance", "privacy", "gdpr", "hipaa", "pci",
                "virus", "malware", "ransomware", "trojan", "phishing", "xss", "injection", "dos", "ddos"));
    }

    /**
     * Validates if the prompt belongs to a supported CS domain.
     *
     * @param prompt The user query or prompt.
     * @return true if valid, false if rejected.
     */
    public boolean validate(String prompt) {
        if (prompt == null || prompt.trim().isEmpty()) {
            return false;
        }

        // Short-circuit for very common code-related symbols
        if (prompt.contains("{") || prompt.contains("}") || prompt.contains(";") || prompt.contains("()")) {
            return true;
        }

        ClassificationResult result = classify(prompt);
        return result.score >= VALIDATION_THRESHOLD;
    }

    /**
     * Classifies the prompt into a domain and score.
     */
    public ClassificationResult classify(String prompt) {
        String normalized = prompt.toLowerCase().replaceAll("[^a-z0-9\\s]", " ");
        String[] tokens = normalized.split("\\s+");

        Map<String, Double> domainScores = new HashMap<>();

        for (String token : tokens) {
            if (ignoreWords.contains(token) || token.length() < 2)
                continue;

            // Penalty for negative keywords
            if (negativeKeywords.contains(token)) {
                for (String d : domainKeywords.keySet()) {
                    domainScores.put(d, domainScores.getOrDefault(d, 0.0) - 5.0);
                }
                // Ensure pure negative prompts fail
                domainScores.put("PENALTY", domainScores.getOrDefault("PENALTY", 0.0) - 5.0);
            }

            for (Map.Entry<String, List<String>> entry : domainKeywords.entrySet()) {
                String domain = entry.getKey();
                List<String> keywords = entry.getValue();

                if (keywords.contains(token)) {
                    domainScores.put(domain, domainScores.getOrDefault(domain, 0.0) + 1.0);
                }
                // Partial match bonus (e.g. "algorithmic" matches "algorithm")
                else {
                    for (String k : keywords) {
                        if (token.contains(k) || k.contains(token)) {
                            // Only count if significant overlap length
                            if (Math.min(token.length(), k.length()) > 3) {
                                domainScores.put(domain, domainScores.getOrDefault(domain, 0.0) + 0.5);
                                break; // Count once per token per domain
                            }
                        }
                    }
                }
            }
        }

        // Find best domain
        String bestDomain = "UNKNOWN";
        double maxScore = 0.0;

        for (Map.Entry<String, Double> entry : domainScores.entrySet()) {
            if (entry.getKey().equals("PENALTY"))
                continue;

            if (entry.getValue() > maxScore) {
                maxScore = entry.getValue();
                bestDomain = entry.getKey();
            }
        }

        return new ClassificationResult(bestDomain, maxScore);
    }

    public static class ClassificationResult {
        public final String domain;
        public final double score;

        public ClassificationResult(String domain, double score) {
            this.domain = domain;
            this.score = score;
        }

        @Override
        public String toString() {
            return String.format("Domain: %s, Score: %.2f", domain, score);
        }
    }
}
