package com.example.automl_prototype_1.controller;

import com.example.automl_prototype_1.service.LogService;
import com.example.automl_prototype_1.service.agentic.BehavioralCloningService;
import com.github.kwhat.jnativehook.GlobalScreen;
import com.github.kwhat.jnativehook.NativeHookException;
import com.github.kwhat.jnativehook.keyboard.NativeKeyEvent;
import com.github.kwhat.jnativehook.keyboard.NativeKeyListener;
import com.github.kwhat.jnativehook.mouse.NativeMouseEvent;
import com.github.kwhat.jnativehook.mouse.NativeMouseListener;
import com.github.kwhat.jnativehook.mouse.NativeMouseMotionListener;

import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Global Keyboard Hook to capture user input for Behavioral Cloning
 * while playing external games (e.g. NFS).
 */
public class GlobalInputListener implements NativeKeyListener, NativeMouseListener, NativeMouseMotionListener {

    private final BehavioralCloningService cloningService;
    private boolean isListening = false;
    private String profile = "WASD"; // Default

    public GlobalInputListener(BehavioralCloningService cloningService) {
        this.cloningService = cloningService;

        // Disable JNativeHook's noisy logger
        Logger logger = Logger.getLogger(GlobalScreen.class.getPackage().getName());
        logger.setLevel(Level.OFF);
        logger.setUseParentHandlers(false);
    }

    public void start() {
        if (isListening)
            return;
        try {
            GlobalScreen.registerNativeHook();
            GlobalScreen.addNativeKeyListener(this);
            GlobalScreen.addNativeMouseListener(this);
            GlobalScreen.addNativeMouseMotionListener(this);
            isListening = true;
            LogService.info("GlobalHook", "Global Input Listener Started. Focus your game now.");
        } catch (NativeHookException e) {
            LogService.error("GlobalHook", "Failed to start global hook: " + e.getMessage());
        }
    }

    public void stop() {
        if (!isListening)
            return;
        try {
            GlobalScreen.removeNativeKeyListener(this);
            GlobalScreen.removeNativeMouseListener(this);
            GlobalScreen.removeNativeMouseMotionListener(this);
            GlobalScreen.unregisterNativeHook();
            isListening = false;
            LogService.info("GlobalHook", "Global Input Listener Stopped.");
        } catch (NativeHookException e) {
            LogService.error("GlobalHook", "Failed to stop global hook: " + e.getMessage());
        }
    }

    // --- Safety / Hand-off Logic ---
    private java.util.function.Consumer<String> safetyCallback;

    public void setSafetyCallback(java.util.function.Consumer<String> callback) {
        this.safetyCallback = callback;
    }

    @Override
    public void nativeKeyPressed(NativeKeyEvent e) {
        // Safety mode (agent is driving): any human key = hand-off.
        if (safetyCallback != null) {
            safetyCallback.accept("KEYBOARD_INTERRUPT");
            return;
        }

        // Recording mode: capture human game controls for behavioral cloning.
        int action = mapKeyToAction(e.getKeyCode());
        if (action != 0) {
            cloningService.setCurrentAction(action);
        }
    }

    // --- Mouse Listeners ---
    @Override
    public void nativeMousePressed(NativeMouseEvent e) {
        if (safetyCallback != null) {
            safetyCallback.accept("MOUSE_INTERRUPT");
            return;
        }
        cloningService.setCurrentMouseButton(e.getButton());
    }

    @Override
    public void nativeMouseMoved(NativeMouseEvent e) {
        // Only yield during autonomous agent control, not while recording demos.
        if (safetyCallback != null) {
            safetyCallback.accept("MOUSE_MOVE_INTERRUPT");
        }
    }

    @Override
    public void nativeMouseReleased(NativeMouseEvent e) {
        cloningService.setCurrentMouseButton(0);
    }

    @Override
    public void nativeMouseDragged(NativeMouseEvent e) {
    }

    public void setProfile(String profile) {
        this.profile = profile;
    }

    @Override
    public void nativeKeyReleased(NativeKeyEvent e) {
        // Simple discrete action logic: Release = Action 0 (No-Op)
        // For more complex continuous control, we'd track key state.
        int action = mapKeyToAction(e.getKeyCode());
        if (action != 0) {
            // Only reset if the released key was the one triggering the action
            cloningService.setCurrentAction(0);
        }
    }

    @Override
    public void nativeKeyTyped(NativeKeyEvent e) {
        // Unused
    }

    private int mapKeyToAction(int keyCode) {
        if ("ARROWS".equals(profile)) {
            switch (keyCode) {
                case NativeKeyEvent.VC_UP:
                    return 1;
                case NativeKeyEvent.VC_LEFT:
                    return 2;
                case NativeKeyEvent.VC_RIGHT:
                    return 3;
                case NativeKeyEvent.VC_DOWN:
                    return 4;
                case NativeKeyEvent.VC_SPACE:
                    return 5;
            }
        } else {
            // WASD
            switch (keyCode) {
                case NativeKeyEvent.VC_W:
                    return 1;
                case NativeKeyEvent.VC_A:
                    return 2;
                case NativeKeyEvent.VC_D:
                    return 3;
                case NativeKeyEvent.VC_S:
                    return 4;
                case NativeKeyEvent.VC_SPACE:
                    return 5;
            }
        }
        return 0; // Unknown/Ignored
    }
}
