package com.example.automl_prototype_1.service;

import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Kill-switch for java.awt.Robot desktop control (LCA, RL Lab, exploration).
 * Default OFF — must be explicitly enabled by the user.
 */
public final class DesktopAutomationGate {

    private static final DesktopAutomationGate INSTANCE = new DesktopAutomationGate();
    private final AtomicBoolean enabled = new AtomicBoolean(false);

    private DesktopAutomationGate() {
    }

    public static DesktopAutomationGate getInstance() {
        return INSTANCE;
    }

    public boolean isEnabled() {
        return enabled.get();
    }

    public void setEnabled(boolean on) {
        enabled.set(on);
        LogService.info("DesktopGate", on ? "Robot automation ENABLED by user." : "Robot automation DISABLED.");
    }

    public boolean requireEnabled(String source) {
        if (enabled.get()) {
            return true;
        }
        LogService.warn("DesktopGate", "Blocked Robot action from " + source + " (gate is OFF).");
        return false;
    }
}
