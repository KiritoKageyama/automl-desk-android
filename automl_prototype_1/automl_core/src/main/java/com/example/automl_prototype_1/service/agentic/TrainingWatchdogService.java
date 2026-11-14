package com.example.automl_prototype_1.service.agentic;

import com.example.automl_prototype_1.service.LogService;

import java.time.Instant;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ScheduledFuture;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicReference;

/**
 * Phase 9.2 — Training Watchdog Service.
 *
 * Monitors training jobs and triggers
 * SelfCorrectionLoopService.triggerCorrection()
 * when a job appears stuck (no progress heartbeat for > 5 minutes).
 *
 * Usage:
 * TrainingWatchdogService.getInstance().beginWatch("XGBoost");
 * TrainingWatchdogService.getInstance().heartbeat(); // call periodically from
 * training thread
 * TrainingWatchdogService.getInstance().endWatch();
 */
public class TrainingWatchdogService {

    private static final long STUCK_THRESHOLD_MS = 5 * 60 * 1000L; // 5 minutes
    private static final long CHECK_INTERVAL_S = 30; // check every 30 seconds

    private static TrainingWatchdogService instance;

    private final ScheduledExecutorService scheduler;
    private ScheduledFuture<?> watchdogFuture;

    // Current watch state
    private final AtomicBoolean watching = new AtomicBoolean(false);
    private final AtomicLong lastBeat = new AtomicLong(0);
    private final AtomicReference<String> currentJob = new AtomicReference<>("Unknown");

    private TrainingWatchdogService() {
        this.scheduler = Executors.newSingleThreadScheduledExecutor(r -> {
            Thread t = new Thread(r, "TrainingWatchdog");
            t.setDaemon(true);
            return t;
        });
    }

    public static synchronized TrainingWatchdogService getInstance() {
        if (instance == null) {
            instance = new TrainingWatchdogService();
        }
        return instance;
    }

    /**
     * Starts the background watchdog loop.
     * Safe to call multiple times (idempotent).
     */
    public synchronized void start() {
        if (watchdogFuture != null && !watchdogFuture.isDone())
            return;

        watchdogFuture = scheduler.scheduleAtFixedRate(this::checkForStuck,
                CHECK_INTERVAL_S, CHECK_INTERVAL_S, TimeUnit.SECONDS);
        LogService.info("TrainingWatchdog", "Watchdog started. Stuck threshold: 5 minutes.");
    }

    /**
     * Registers a new training job to watch.
     *
     * @param jobName Human-readable name (e.g. "XGBoost", "DistilBERT Fine-Tune")
     */
    public void beginWatch(String jobName) {
        currentJob.set(jobName);
        lastBeat.set(System.currentTimeMillis());
        watching.set(true);
        LogService.info("TrainingWatchdog", "Watching job: " + jobName);
    }

    /**
     * Called by training threads periodically to signal progress.
     * Resets the stuck timer.
     */
    public void heartbeat() {
        lastBeat.set(System.currentTimeMillis());
    }

    /**
     * Signals that the current training job has finished (success or failure).
     * Stops the stuck-detection timer.
     */
    public void endWatch() {
        if (watching.getAndSet(false)) {
            LogService.info("TrainingWatchdog", "Job completed: " + currentJob.get());
        }
    }

    private void checkForStuck() {
        if (!watching.get())
            return;

        long elapsed = System.currentTimeMillis() - lastBeat.get();
        if (elapsed > STUCK_THRESHOLD_MS) {
            String jobName = currentJob.get();
            LogService.error("TrainingWatchdog",
                    String.format(
                            "🚨 STUCK DETECTED: '%s' has not made progress for %.1f minutes. Triggering correction.",
                            jobName, elapsed / 60000.0));

            // Trigger self-correction
            SelfCorrectionLoopService.getInstance().triggerCorrection(
                    "TIMEOUT stuck_training=" + jobName);

            // Prevent repeated triggers for same event
            endWatch();
        }
    }

    /** Returns true if a training job is currently being watched. */
    public boolean isWatching() {
        return watching.get();
    }

    /** Returns the name of the currently watched job, or null. */
    public String getCurrentJob() {
        return watching.get() ? currentJob.get() : null;
    }

    /** Returns ms since last heartbeat (0 if not watching). */
    public long getMsSinceLastHeartbeat() {
        if (!watching.get())
            return 0;
        return System.currentTimeMillis() - lastBeat.get();
    }
}
