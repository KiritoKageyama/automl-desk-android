package com.example.automl_prototype_1.service;

import java.io.PrintWriter;
import java.io.StringWriter;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.time.ZoneOffset;
import java.time.format.DateTimeFormatter;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Production crash/error sink: logs + persisted reports under ~/.automl/crashes.
 */
public final class CrashReportService {

    private static final CrashReportService INSTANCE = new CrashReportService();
    private static final DateTimeFormatter TS =
            DateTimeFormatter.ofPattern("yyyyMMdd-HHmmss-SSS").withZone(ZoneOffset.UTC);
    private final AtomicInteger reportCount = new AtomicInteger();

    private CrashReportService() {
    }

    public static CrashReportService getInstance() {
        return INSTANCE;
    }

    public Path crashDir() {
        return Path.of(System.getProperty("user.home"), ".automl", "crashes");
    }

    public int getReportCount() {
        return reportCount.get();
    }

    /**
     * Record a fatal/unexpected failure. Never throws.
     */
    public Path report(String source, Thread thread, Throwable error) {
        Objects.requireNonNull(source, "source");
        try {
            Path dir = crashDir();
            Files.createDirectories(dir);
            String stamp = TS.format(Instant.now());
            Path file = dir.resolve("crash-" + stamp + "-" + reportCount.incrementAndGet() + ".txt");

            StringWriter sw = new StringWriter();
            PrintWriter pw = new PrintWriter(sw);
            pw.println("AutoML Crash Report");
            pw.println("time_utc=" + Instant.now());
            pw.println("source=" + source);
            pw.println("thread=" + (thread != null ? thread.getName() : "unknown"));
            pw.println("java=" + System.getProperty("java.version"));
            pw.println("os=" + System.getProperty("os.name") + " " + System.getProperty("os.version"));
            pw.println("---");
            if (error != null) {
                error.printStackTrace(pw);
            } else {
                pw.println("(no throwable)");
            }
            pw.flush();

            Files.writeString(file, sw.toString(), StandardCharsets.UTF_8);
            LogService.error("CrashReport", source + ": " + (error != null ? error.getMessage() : "unknown")
                    + " → " + file);
            return file;
        } catch (Throwable t) {
            System.err.println("[CrashReport] Failed to persist report: " + t.getMessage());
            if (error != null) {
                error.printStackTrace(System.err);
            }
            return null;
        }
    }

    public Path report(String source, Throwable error) {
        return report(source, Thread.currentThread(), error);
    }
}
