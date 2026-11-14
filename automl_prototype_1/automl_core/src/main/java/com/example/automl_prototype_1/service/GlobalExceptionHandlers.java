package com.example.automl_prototype_1.service;

/**
 * Installs JVM uncaught exception handlers that persist crash reports.
 * Call {@link #installJavaFxThreadHandler()} from the JavaFX application thread.
 */
public final class GlobalExceptionHandlers {

    private static volatile boolean installed;

    private GlobalExceptionHandlers() {
    }

    public static synchronized void install() {
        if (installed) {
            return;
        }
        Thread.UncaughtExceptionHandler handler = (thread, error) ->
                CrashReportService.getInstance().report("uncaught:" + thread.getName(), thread, error);
        Thread.setDefaultUncaughtExceptionHandler(handler);
        Thread.currentThread().setUncaughtExceptionHandler(handler);
        installed = true;
        LogService.info("CrashReport", "Global exception handlers installed.");
    }

    /** Invoke on the JavaFX application thread after toolkit start. */
    public static void installJavaFxThreadHandler() {
        Thread.UncaughtExceptionHandler handler = (thread, error) ->
                CrashReportService.getInstance().report("javafx:" + thread.getName(), thread, error);
        Thread.currentThread().setUncaughtExceptionHandler(handler);
    }
}
