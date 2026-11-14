package com.example.automl_prototype_1.service.agentic;

import com.example.automl_prototype_1.service.LogService;
import com.sun.jna.Native;
import com.sun.jna.platform.win32.User32;
import com.sun.jna.platform.win32.WinDef.HWND;

public class WindowFocusService {

    private static final int MAX_TITLE_LENGTH = 1024;

    /**
     * Checks if the currently focused window title contains the given partial name.
     * Case-insensitive.
     */
    public boolean isAppFocused(String partialName) {
        if (partialName == null || partialName.isEmpty())
            return true; // No restriction

        char[] buffer = new char[MAX_TITLE_LENGTH];
        HWND hwnd = User32.INSTANCE.GetForegroundWindow();
        if (hwnd == null)
            return false;

        User32.INSTANCE.GetWindowText(hwnd, buffer, MAX_TITLE_LENGTH);
        String title = Native.toString(buffer);

        if (title == null || title.isEmpty())
            return false;

        boolean match = title.toLowerCase().contains(partialName.toLowerCase());
        // Simple optimization to avoid spamming logs, maybe only log on change?
        // For now, minimal logging.
        return match;
    }

    public String getForegroundWindowTitle() {
        char[] buffer = new char[MAX_TITLE_LENGTH];
        HWND hwnd = User32.INSTANCE.GetForegroundWindow();
        if (hwnd == null)
            return "";
        User32.INSTANCE.GetWindowText(hwnd, buffer, MAX_TITLE_LENGTH);
        return Native.toString(buffer);
    }
}
