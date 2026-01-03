package com.example.automl_prototype_1.service;

/**
 * Service to detect the runtime platform (Desktop vs Android).
 * Used to conditionally enable features like Python Bridge or heavy GPU
 * operations
 * that are not supported on mobile.
 */
public class PlatformService {

    private static Boolean isAndroid = null;

    /**
     * Checks if the application is running on Android.
     * Uses the "java.vm.name" system property to detect Dalvik.
     *
     * @return true if running on Android, false otherwise.
     */
    public static boolean isAndroid() {
        if (isAndroid == null) {
            String vmName = System.getProperty("java.vm.name");
            isAndroid = vmName != null && vmName.toLowerCase().contains("dalvik");
            if (isAndroid) {
                LogService.info("Platform", "Android Environment Detected (" + vmName + ").");
            } else {
                LogService.info("Platform", "Desktop Environment Detected (" + vmName + ").");
            }
        }
        return isAndroid;
    }

    /**
     * Checks if the application is running on a Desktop environment
     * (Windows/Linux/Mac).
     *
     * @return true if running on Desktop.
     */
    public static boolean isDesktop() {
        return !isAndroid();
    }
}
